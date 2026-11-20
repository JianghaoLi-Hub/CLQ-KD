# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Capture P3/P4/P5 features for the four-term FGD objective.

Student and teacher are weak references, not registered submodules. All channel
adapters and both sides of the global-relation transform belong to FeatureLoss
and are optimized; the frozen teacher detector itself is never optimized.
"""
import weakref
import torch
from torch import nn
from .feature_loss import FeatureLoss

class FGD(nn.Module):
    OFFICIAL_COMMIT = 'c03a6cf3b3cb3cbfd5554dbf4766e7135747d22d'
    COEFFICIENTS = dict(temp=0.5, alpha_fgd=0.001, beta_fgd=0.0005, gamma_fgd=0.0005, lambda_fgd=5e-06)

    def __init__(self, student, teacher, **coefficients):
        super().__init__()
        object.__setattr__(self, '_student_ref', weakref.ref(student))
        object.__setattr__(self, '_teacher_ref', weakref.ref(teacher))
        if teacher.training or any((p.requires_grad for p in teacher.parameters())):
            raise ValueError('FGD requires a frozen eval-mode teacher detector')
        heads = [model.model[-1] for model in (student, teacher)]
        for head in heads:
            if getattr(head, 'end2end', False) or len(head.f) != 3 or len(head.cv2) != 3:
                raise ValueError('Only native three-scale YOLO11 detection is supported')
            if list(map(float, head.stride)) != [8.0, 16.0, 32.0]:
                raise ValueError('FGD taps must correspond to P3/P4/P5')
        if heads[0].nc != heads[1].nc:
            raise ValueError('Teacher and student class counts must match')
        (self.student_indices, self.teacher_indices) = [list(map(int, h.f)) for h in heads]
        (self.student_channels, self.teacher_channels) = [[branch[0].conv.in_channels for branch in h.cv2] for h in heads]
        self.loss = FeatureLoss(self.student_channels, self.teacher_channels, **{**self.COEFFICIENTS, **coefficients})
        (self._hooks, self._captures) = ([], {})
        self._capture_enabled = False
        self.last_shapes = {}
        self.last_total_loss = None

    def _models(self):
        models = (self._student_ref(), self._teacher_ref())
        if any((model is None for model in models)):
            raise RuntimeError('FGD teacher or student has been released')
        return models

    def install_hooks(self):
        if self._hooks:
            return
        owner_ref = weakref.ref(self)
        for (role, model, indices) in zip(('student', 'teacher'), self._models(), (self.student_indices, self.teacher_indices)):
            for (level, index) in enumerate(indices):
                key = (role, level)

                def capture(_module, _inputs, output, key=key):
                    owner = owner_ref()
                    if owner is None or not owner._capture_enabled:
                        return
                    if key in owner._captures:
                        raise RuntimeError(f'Duplicate pyramid forward during one capture: {key}')
                    if not isinstance(output, torch.Tensor) or output.ndim != 4:
                        raise ValueError('Pyramid tap is not a BCHW tensor')
                    owner._captures[key] = output.clone() if key[0] == 'student' else output.detach().clone()
                self._hooks.append(model.model[index].register_forward_hook(capture))

    def remove_hooks(self):
        self.disable_capture()
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def enable_capture(self):
        if not self._hooks:
            raise RuntimeError('Install pyramid capture hooks after constructing EMA')
        self._captures.clear()
        self._capture_enabled = True

    def disable_capture(self):
        self._capture_enabled = False
        self._captures.clear()

    def forward(self, batch):
        expected = {(role, level) for role in ('student', 'teacher') for level in range(3)}
        if set(self._captures) != expected:
            raise RuntimeError('Student/teacher pyramid capture is incomplete')
        student = [self._captures['student', i] for i in range(3)]
        teacher = [self._captures['teacher', i] for i in range(3)]
        self.last_shapes = {role: [list(self._captures[role, i].shape) for i in range(3)] for role in ('student', 'teacher')}
        loss = self.loss(student, teacher, batch['bboxes'], {'batch_idx': batch['batch_idx']})
        self.last_total_loss = loss.detach()
        return loss
