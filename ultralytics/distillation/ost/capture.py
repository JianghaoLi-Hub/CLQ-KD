# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
import hashlib
import json
import weakref
import torch
from torch import nn
from .core import OSTLoss, compact_feature
OST_WEIGHT = 5.0

class OSTCapture(nn.Module):
    """Compact each of all 23 stages immediately; never register detectors here."""

    def __init__(self, student, teacher, qk_dim=128, temperature=1.0, threshold=0.5):
        super().__init__()
        object.__setattr__(self, '_student_ref', weakref.ref(student))
        object.__setattr__(self, '_teacher_ref', weakref.ref(teacher))
        if teacher.training or any((p.requires_grad for p in teacher.parameters())):
            raise ValueError('OST requires frozen eval-mode teacher')
        self.indices = list(range(23))
        if len(student.model) != 24 or len(teacher.model) != 24:
            raise ValueError('Expected native YOLO11 with 23 stages before Detect')
        for model in (student, teacher):
            head = model.model[-1]
            if head.nc != student.model[-1].nc or getattr(head, 'end2end', False) or list(head.f) != [16, 19, 22]:
                raise ValueError('Expected native matching-class YOLO11 Detect head')
        self.stages = []
        for i in self.indices:
            (s, t) = (student.model[i], teacher.model[i])
            if type(s).__name__ != type(t).__name__ or s.f != t.f:
                raise ValueError(f'Nonhomologous top-level stage {i}')
            self.stages.append({'index': i, 'type': type(s).__name__, 'from': s.f})
        shapes = {}
        handles = []
        modes = {module: module.training for model in (student, teacher) for module in model.modules()}
        try:
            for (role, model) in (('student', student), ('teacher', teacher)):
                model.eval()
                for i in self.indices:

                    def capture_shape(_module, _inputs, out, key=(role, i)):
                        if not isinstance(out, torch.Tensor) or out.ndim != 4:
                            raise ValueError(f'Stage {key} must return BCHW tensor')
                        shapes[key] = tuple(out.shape)
                    handles.append(model.model[i].register_forward_hook(capture_shape))
            dummy = torch.zeros(1, 3, 64, 64, device=next(student.parameters()).device)
            with torch.no_grad():
                student(dummy)
                teacher(dummy)
        finally:
            for handle in handles:
                handle.remove()
            for (module, training) in modes.items():
                module.training = training
        self.student_channels = [shapes['student', i][1] for i in self.indices]
        self.teacher_channels = [shapes['teacher', i][1] for i in self.indices]
        for i in self.indices:
            if shapes['student', i][2:] != shapes['teacher', i][2:]:
                raise ValueError(f'OST stage {i} spatial geometry differs')
            self.stages[i].update(student_channels=self.student_channels[i], teacher_channels=self.teacher_channels[i], spatial_at_64=list(shapes['student', i][2:]))
        self.loss = OSTLoss(self.teacher_channels, self.student_channels, qk_dim=qk_dim, temperature=temperature, threshold=threshold)
        (self._hooks, self._captures) = ([], {})
        self._capture_enabled = False
        self.current_stats = None
        self.last_shapes = {}

    def _models(self):
        models = (self._student_ref(), self._teacher_ref())
        if any((model is None for model in models)):
            raise RuntimeError('OST detector reference has been released')
        return models

    def install_hooks(self):
        if self._hooks:
            return
        owner_ref = weakref.ref(self)
        for (role, model) in zip(('student', 'teacher'), self._models()):
            for i in self.indices:

                def capture(_module, _inputs, out, key=(role, i)):
                    owner = owner_ref()
                    if owner is None or not owner._capture_enabled:
                        return
                    if key in owner._captures:
                        raise RuntimeError(f'Duplicate OST stage capture: {key}')
                    if not isinstance(out, torch.Tensor) or out.ndim != 4:
                        raise ValueError(f'Expected BCHW stage output: {key}')
                    if key[0] == 'teacher':
                        out = out.detach()
                    rep = compact_feature(out)
                    owner._captures[key] = rep
                    owner.last_shapes.setdefault(key[0], {})[key[1]] = list(out.shape)
                self._hooks.append(model.model[i].register_forward_hook(capture))

    def remove_hooks(self):
        self.disable_capture()
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def enable_capture(self):
        if not self._hooks:
            raise RuntimeError('OST capture hooks must be installed after EMA construction')
        self._captures.clear()
        self.last_shapes = {}
        self.current_stats = {}
        self._capture_enabled = True

    def disable_capture(self):
        self._capture_enabled = False
        self._captures.clear()

    def forward(self, *, noise=None, gate_override=None):
        expected = {(role, i) for role in ('student', 'teacher') for i in self.indices}
        if set(self._captures) != expected:
            raise RuntimeError('Incomplete 23-stage student/teacher capture')
        student = [self._captures['student', i] for i in self.indices]
        teacher = [self._captures['teacher', i] for i in self.indices]
        raw = self.loss(student, teacher, noise=noise, gate_override=gate_override)
        return raw
