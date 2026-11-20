# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
import hashlib
import json
import math
import weakref
import torch
from .core import gkd_loss, validate_layer_weights
GKD_WEIGHT = 0.125

def validate_kd_weight(value):
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError('kd_weight must be finite and nonnegative')
    return value

class GKDCapture:
    """Independent Detect arguments, not total derivatives at neck nodes.

    P3 also feeds P4/P5 in YOLO's neck. Cloning at the Detect boundary makes the
    three gradient targets independent while preserving all student feature
    paths for the subsequent first-order KD backward.
    """
    indices = [16, 19, 22]

    def __init__(self, student, teacher, kd_weight=GKD_WEIGHT, layer_weights=(1.0, 1.0, 1.0)):
        self.kd_weight = validate_kd_weight(kd_weight)
        self.layer_weights = validate_layer_weights(layer_weights)
        (self._student_ref, self._teacher_ref) = (weakref.ref(student), weakref.ref(teacher))
        for model in (student, teacher):
            head = model.model[-1]
            if len(model.model) != 24 or list(head.f) != self.indices:
                raise ValueError('Expected native YOLO11 P3/P4/P5 Detect inputs')
            if head.nc != student.model[-1].nc or getattr(head, 'end2end', False):
                raise ValueError('Expected ordinary matching-class Detect head')
        (self._hook, self.enabled, self.features) = (None, False, None)
        self.last_shapes = {}

    def install_hooks(self):
        if self._hook is not None:
            return
        owner = weakref.ref(self)

        def capture(_module, inputs):
            instance = owner()
            if instance is None or not instance.enabled:
                return
            if instance.features is not None:
                raise RuntimeError('Duplicate student Detect input capture')
            features = inputs[0]
            if len(features) != 3 or any((x.ndim != 4 for x in features)):
                raise ValueError('GKD needs three BCHW Detect input tensors')
            instance.features = [x.clone() for x in features]
            instance.last_shapes['student'] = [list(x.shape) for x in instance.features]
            return (instance.features.copy(), *inputs[1:])
        self._hook = self._student_ref().model[-1].register_forward_pre_hook(capture)

    def remove_hooks(self):
        self.disable_capture()
        if self._hook is not None:
            self._hook.remove()
            self._hook = None

    def enable_capture(self):
        if self._hook is None:
            raise RuntimeError('Install capture only after EMA construction')
        (self.features, self.last_shapes, self.enabled) = (None, {}, True)

    def disable_capture(self):
        (self.features, self.enabled) = (None, False)

    def teacher_head_inputs(self, images):
        """Exact native backbone/neck forward, excluding Detect, under no_grad."""
        teacher = self._teacher_ref()
        if teacher.training or any((p.requires_grad for p in teacher.parameters())):
            raise RuntimeError('Teacher must remain frozen and eval')
        (saved, value) = ([], images)
        with torch.no_grad():
            for module in teacher.model[:-1]:
                if module.f != -1:
                    value = saved[module.f] if isinstance(module.f, int) else [value if index == -1 else saved[index] for index in module.f]
                value = module(value)
                saved.append(value if module.i in teacher.save else None)
            features = [saved[index] for index in teacher.model[-1].f]
        leaves = [x.detach().requires_grad_(True) for x in features]
        self.last_shapes['teacher'] = [list(x.shape) for x in leaves]
        return leaves
