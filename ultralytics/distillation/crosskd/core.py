# CrossKD YOLO11 adaptation; source: https://github.com/jbwang1997/CrossKD.
# Upstream reference: CC BY-NC 4.0. See THIRD_PARTY.md and licenses/.
"""Response-based CrossKD for native Ultralytics 8.3.178 YOLO11 detection heads.

Algorithm reference: jbwang1997/CrossKD, commit
87e156a46259e3a516b65fc6a049b200b9ab0d54, CrossKDGFL configuration.
This is an architecture adaptation, not an exact reproduction of the GFL study.
The teacher and student are deliberately NOT registered children of this module.
Only channel adapters are persisted.
"""
from __future__ import annotations
import weakref
from typing import Any
import torch
from torch import nn
from torch.nn import functional as F

class CrossKD(nn.Module):
    """Reuse a frozen teacher suffix on actual student-forward preactivations.

    Lifecycle: construct before the optimizer, install_hooks after EMA creation,
    enable_capture before each native student/teacher forward, cross_forward,
    loss, and disable_capture(clear=True) in a finally block. Remove hooks around
    student deepcopy/checkpoint serialization. Inference never needs this module.
    """
    OFFICIAL_COMMIT = '87e156a46259e3a516b65fc6a049b200b9ab0d54'

    def __init__(self, student, teacher, beta=1.0, temperature=1.0, cls_weight=1.0, reg_weight=4.0):
        super().__init__()
        object.__setattr__(self, '_student_ref', weakref.ref(student))
        object.__setattr__(self, '_teacher_ref', weakref.ref(teacher))
        (student_head, teacher_head) = (student.model[-1], teacher.model[-1])
        self._validate_head(student_head)
        self._validate_head(teacher_head)
        if student_head.nc != teacher_head.nc or student_head.reg_max != teacher_head.reg_max:
            raise ValueError('Teacher/student class counts and distribution-bin counts must match')
        if len(student_head.cv2) != len(teacher_head.cv2):
            raise ValueError('Teacher/student prediction-scale counts must match')
        if len(student_head.cv2) != 3:
            raise ValueError('CrossKD requires precisely P3, P4 and P5')
        if teacher.training or any((parameter.requires_grad for parameter in teacher.parameters())):
            raise ValueError('Teacher must be eval() and fully frozen before CrossKD construction')
        self.nc = int(teacher_head.nc)
        self.reg_max = int(teacher_head.reg_max)
        self.no = 4 * self.reg_max + self.nc
        (self.beta, self.temperature) = (float(beta), float(temperature))
        (self.cls_weight, self.reg_weight) = (float(cls_weight), float(reg_weight))
        if self.beta < 0 or self.temperature < 1 or min(self.cls_weight, self.reg_weight) < 0:
            raise ValueError('Invalid CrossKD hyperparameters')
        self.reg_adapters = self._make_adapters(student_head.cv2, teacher_head.cv2)
        self.cls_adapters = self._make_adapters(student_head.cv3, teacher_head.cv3)
        self._hooks = []
        self._capture_enabled = False
        self._captures = {}

    @staticmethod
    def _validate_head(head):
        if getattr(head, 'end2end', False):
            raise ValueError('End-to-end/one-to-one YOLO heads are not supported by this adaptation')
        if len(head.cv2) != len(head.cv3):
            raise ValueError('Paired regression and classification branches are required')
        for branch in (*head.cv2, *head.cv3):
            if len(branch) != 3:
                raise ValueError('Expected two hidden blocks followed by the prediction convolution')
            CrossKD._terminal_conv(branch[0])

    @staticmethod
    def _terminal_conv(first_block):
        terminal = first_block
        while isinstance(terminal, nn.Sequential):
            terminal = terminal[-1]
        if not all((hasattr(terminal, name) for name in ('conv', 'bn', 'act'))):
            raise ValueError('Expected unfused Conv/BN/activation at the first-block output')
        if not isinstance(terminal.bn, nn.modules.batchnorm._BatchNorm):
            raise ValueError('First-block tap must be an unfused BatchNorm preactivation')
        return terminal

    @classmethod
    def _make_adapters(cls, student_branches, teacher_branches):
        return nn.ModuleList((nn.Identity() if cls._terminal_conv(student[0]).conv.out_channels == cls._terminal_conv(teacher[0]).conv.out_channels else nn.Conv2d(cls._terminal_conv(student[0]).conv.out_channels, cls._terminal_conv(teacher[0]).conv.out_channels, 1, bias=False) for (student, teacher) in zip(student_branches, teacher_branches)))

    def _models(self):
        (student, teacher) = (self._student_ref(), self._teacher_ref())
        if student is None or teacher is None:
            raise RuntimeError('CrossKD student or teacher model has been released')
        return (student, teacher)

    def install_hooks(self):
        """Install hooks once; callers must create the inference EMA first."""
        if self._hooks:
            return
        self_ref = weakref.ref(self)
        for (role, model) in zip(('student', 'teacher'), self._models()):
            head = model.model[-1]
            for branch_name in ('cv2', 'cv3'):
                for (level, branch) in enumerate(getattr(head, branch_name)):
                    key = (role, branch_name, level)

                    def save_tap(_module, _inputs, output, key=key):
                        owner = self_ref()
                        if owner is None or not owner._capture_enabled:
                            return
                        if key in owner._captures:
                            raise RuntimeError(f'Duplicate native head forward during one capture: {key}')
                        owner._captures[key] = output.clone() if key[0] == 'student' else output.detach().clone()
                    self._hooks.append(self._terminal_conv(branch[0]).bn.register_forward_hook(save_tap))

    def remove_hooks(self):
        self.disable_capture(clear=True)
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def enable_capture(self):
        if not self._hooks:
            raise RuntimeError('install_hooks() is required before enabling native-forward capture')
        self._captures.clear()
        self._capture_enabled = True

    def disable_capture(self, clear=True):
        self._capture_enabled = False
        if clear:
            self._captures.clear()

    @staticmethod
    def _align_statistics(student_feature, teacher_feature, eps=1e-06):
        """Official per-channel N,H,W sample mean/std alignment (not a loss)."""
        if student_feature.shape != teacher_feature.shape:
            raise ValueError('Statistics alignment requires equal adapted feature shapes')
        if student_feature.numel() // student_feature.shape[1] < 2:
            raise ValueError('Official sample-standard-deviation alignment requires at least two samples')
        dims = (0, 2, 3)
        teacher_feature = teacher_feature.detach()
        student_mean = student_feature.mean(dim=dims, keepdim=True)
        student_std = student_feature.std(dim=dims, keepdim=True, correction=1)
        teacher_mean = teacher_feature.mean(dim=dims, keepdim=True)
        teacher_std = teacher_feature.std(dim=dims, keepdim=True, correction=1)
        return (student_feature - student_mean) / (student_std + eps) * teacher_std + teacher_mean

    def cross_forward(self):
        """Run teacher second hidden block and predictor, without no_grad()."""
        (_, teacher) = self._models()
        if teacher.training or any((parameter.requires_grad for parameter in teacher.parameters())):
            raise RuntimeError('Teacher must remain frozen and eval throughout CrossKD')
        if len(self._captures) != 12:
            raise RuntimeError(f'Expected 12 student/teacher preactivation taps, got {len(self._captures)}')
        head = teacher.model[-1]
        outputs = []
        for level in range(3):
            results = []
            for (branch_name, adapters) in (('cv2', self.reg_adapters), ('cv3', self.cls_adapters)):
                student_tap = self._captures['student', branch_name, level]
                teacher_tap = self._captures['teacher', branch_name, level]
                branch = getattr(head, branch_name)[level]
                feature = self._align_statistics(adapters[level](student_tap), teacher_tap)
                feature = self._terminal_conv(branch[0]).act(feature)
                feature = branch[2](branch[1](feature))
                results.append(feature)
            outputs.append(torch.cat(results, dim=1))
        return outputs

    @staticmethod
    def _levels(predictions):
        levels = predictions[1] if isinstance(predictions, tuple) else predictions
        if not isinstance(levels, (list, tuple)) or len(levels) != 3:
            raise ValueError('Expected three raw P3-P5 prediction tensors, not decoded predictions')
        return levels

    def forward(self, cross_predictions, teacher_predictions, positive_mask):
        """Unscaled KD scalar. Trainer applies global alpha and batch factor once."""
        (cross_levels, teacher_levels) = (self._levels(cross_predictions), self._levels(teacher_predictions))
        batch = cross_levels[0].shape[0]
        anchors = sum((level.shape[-2] * level.shape[-1] for level in cross_levels))
        if positive_mask.dtype != torch.bool or tuple(positive_mask.shape) != (batch, anchors):
            raise ValueError(f'Expected a boolean TAA mask of shape {(batch, anchors)}')
        cls_numerator = cross_levels[0].new_zeros((), dtype=torch.float32)
        reg_reduced_numerator = cross_levels[0].new_zeros((), dtype=torch.float32)
        confidence_sum = cross_levels[0].new_zeros((), dtype=torch.float32)
        eps = torch.finfo(torch.float32).eps
        for (cross, teacher) in zip(cross_levels, teacher_levels):
            if cross.shape != teacher.shape or cross.ndim != 4 or cross.shape[:2] != (batch, self.no):
                raise ValueError(f'Raw prediction shapes disagree: {tuple(cross.shape)}, {tuple(teacher.shape)}')
            if not torch.isfinite(cross).all() or not torch.isfinite(teacher).all():
                raise FloatingPointError('Nonfinite CrossKD raw predictions')
            (cross_reg, cross_cls) = cross.float().split((4 * self.reg_max, self.nc), dim=1)
            (teacher_reg, teacher_cls) = teacher.detach().float().split((4 * self.reg_max, self.nc), dim=1)
            quality = teacher_cls.sigmoid()
            focal = (cross_cls.sigmoid() - quality).abs().pow(self.beta)
            cls_numerator = cls_numerator + (F.binary_cross_entropy_with_logits(cross_cls, quality, reduction='none') * focal).sum()
            (height, width) = cross_reg.shape[-2:]
            cross_reg = cross_reg.reshape(batch, 4, self.reg_max, height, width)
            teacher_reg = teacher_reg.reshape(batch, 4, self.reg_max, height, width)
            distribution_kl = F.kl_div(F.log_softmax(cross_reg / self.temperature, dim=2), F.softmax(teacher_reg / self.temperature, dim=2), reduction='none').sum(dim=2) * self.temperature ** 2
            confidence = quality.max(dim=1, keepdim=True).values
            reg_reduced_numerator = reg_reduced_numerator + (distribution_kl * confidence).sum() / (4.0 + eps)
            confidence_sum = confidence_sum + confidence.sum()
        positive_counts = positive_mask.detach().sum(dim=1)
        positive_count = positive_counts.clamp_min(1).sum().float()
        cls_loss = cls_numerator / (positive_count + eps)
        if bool(confidence_sum.detach() == 0):
            reg_loss = reg_reduced_numerator * 0.0
        else:
            reg_loss = reg_reduced_numerator / confidence_sum
        total = self.cls_weight * cls_loss + self.reg_weight * reg_loss
        if not torch.isfinite(total):
            raise FloatingPointError('Nonfinite CrossKD loss')
        return total
