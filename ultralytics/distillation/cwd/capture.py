# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
import hashlib
import json
import math
import weakref
import torch
from torch import nn
from .loss import CWDAdapters
CWD_WEIGHT = 5.0

def validate_kd_weight(value):
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError('kd_weight must be finite and nonnegative')
    return value

def validate_tau(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError('tau must be finite and positive')
    return value

class CWDCapture(CWDAdapters):
    """Own only auxiliary projections; detectors are weak references."""
    indices = [16, 19, 22]

    def __init__(self, student, teacher, kd_weight=CWD_WEIGHT, tau=1.0):
        for model in (student, teacher):
            head = model.model[-1]
            if len(model.model) != 24 or list(head.f) != self.indices or head.nc != student.model[-1].nc or getattr(head, 'end2end', False):
                raise ValueError('Expected native YOLO11 Detect inputs at 16/19/22')
        if teacher.training or any((p.requires_grad for p in teacher.parameters())):
            raise ValueError('CWD teacher must be frozen and eval')
        (shapes, hooks) = ({}, [])
        modes = {module: module.training for model in (student, teacher) for module in model.modules()}
        try:
            for (role, model) in (('student', student), ('teacher', teacher)):
                model.eval()
                for i in self.indices:

                    def capture(_module, _inputs, output, key=(role, i)):
                        if not isinstance(output, torch.Tensor) or output.ndim != 4:
                            raise ValueError('Expected BCHW feature')
                        shapes[key] = tuple(output.shape)
                    hooks.append(model.model[i].register_forward_hook(capture))
            dummy = torch.zeros(1, 3, 64, 64, device=next(student.parameters()).device)
            with torch.no_grad():
                student(dummy)
                teacher(dummy)
        finally:
            for hook in hooks:
                hook.remove()
            for (module, training) in modes.items():
                module.training = training
        if any((shapes['student', i][2:] != shapes['teacher', i][2:] for i in self.indices)):
            raise ValueError('CWD spatial levels differ; interpolation is forbidden')
        super().__init__([shapes['student', i][1] for i in self.indices], [shapes['teacher', i][1] for i in self.indices], tau=tau)
        object.__setattr__(self, '_student_ref', weakref.ref(student))
        object.__setattr__(self, '_teacher_ref', weakref.ref(teacher))
        self.kd_weight = validate_kd_weight(kd_weight)
        self.tau = validate_tau(tau)
        (self._hook, self.enabled, self.features) = (None, False, None)
        self.last_shapes = {}

    def install_hooks(self):
        if self._hook is not None:
            return
        owner_ref = weakref.ref(self)

        def capture(_module, inputs):
            owner = owner_ref()
            if owner is None or not owner.enabled:
                return
            if owner.features is not None:
                raise RuntimeError('Duplicate CWD Detect-input capture')
            features = inputs[0]
            if len(features) != 3 or any((x.ndim != 4 for x in features)):
                raise ValueError('CWD requires all three BCHW Detect-input features')
            owner.features = list(features)
            owner.last_shapes['student'] = [list(x.shape) for x in features]
        self._hook = self._student_ref().model[-1].register_forward_pre_hook(capture)

    def remove_hooks(self):
        self.disable_capture()
        if self._hook is not None:
            self._hook.remove()
            self._hook = None

    def enable_capture(self):
        if self._hook is None:
            raise RuntimeError('Capture hooks must be installed after EMA construction')
        (self.features, self.last_shapes, self.enabled) = (None, {}, True)

    def disable_capture(self):
        (self.features, self.enabled) = (None, False)

    def teacher_head_inputs(self, images):
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
            features = [saved[index].detach() for index in teacher.model[-1].f]
        self.last_shapes['teacher'] = [list(x.shape) for x in features]
        return features
