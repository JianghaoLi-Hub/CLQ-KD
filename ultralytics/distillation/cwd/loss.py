# Copyright 2018-2019 Open-MMLab. All rights reserved.
# Apache-2.0; see licenses/CWD-Apache-2.0.txt.
# Modified: YOLO11 feature adaptation and shared training integration.
"""Channel-wise spatial KL distillation and training-only YOLO adapters."""
import math
import torch
from torch import nn
from torch.nn import functional as F

class CWDLoss(nn.Module):
    """Full KL(P_T || P_S), spatial softmax per sample/channel, summed over levels.

    Each level is reduced by its *aligned* B*C, not by H*W. The official
    detection code omits teacher entropy; subtracting that fixed entropy
    gives this loss and leaves student/adapter gradients unchanged.
    """

    def __init__(self, loss_weight=1.0, tau=1.0):
        super().__init__()
        self.loss_weight = float(loss_weight)
        self.tau = float(tau)
        if not math.isfinite(self.loss_weight) or self.loss_weight < 0:
            raise ValueError('CWD internal loss weight must be finite and nonnegative')
        if not math.isfinite(self.tau) or self.tau <= 0:
            raise ValueError('CWD temperature must be finite and positive')

    def forward(self, preds_S, preds_T):
        if isinstance(preds_S, torch.Tensor):
            (preds_S, preds_T) = ((preds_S,), (preds_T,))
        if len(preds_S) != len(preds_T) or not preds_S:
            raise ValueError('CWD requires nonempty corresponding student/teacher levels')
        losses = []
        for (student, teacher) in zip(preds_S, preds_T):
            if student.ndim != 4 or teacher.ndim != 4 or any((size <= 0 for size in student.shape)):
                raise ValueError('CWD requires nonempty BCHW feature tensors')
            if student.shape != teacher.shape:
                raise ValueError(f'CWD aligned feature shapes differ: {student.shape} vs {teacher.shape}')
            (batch, channels, height, width) = student.shape
            student_log_prob = F.log_softmax(student.reshape(batch * channels, height * width) / self.tau, dim=1)
            teacher_prob = F.softmax(teacher.detach().reshape(batch * channels, height * width) / self.tau, dim=1)
            losses.append(F.kl_div(student_log_prob, teacher_prob, reduction='sum') * (self.tau ** 2 / (batch * channels)))
        return sum(losses) * self.loss_weight

class CWDAdapters(nn.Module):
    """Training-only 1x1 biased projections; not attached to student or EMA."""

    def __init__(self, student_channels, teacher_channels, tau=1.0):
        super().__init__()
        if len(student_channels) != 3 or len(teacher_channels) != 3:
            raise ValueError('CWD requires three feature levels: P3, P4 and P5')
        self.student_channels = list(student_channels)
        self.teacher_channels = list(teacher_channels)
        self.adapters = nn.ModuleList([nn.Conv2d(student, teacher, 1, bias=True) if student != teacher else nn.Identity() for (student, teacher) in zip(student_channels, teacher_channels)])
        self.loss = CWDLoss(loss_weight=1.0, tau=tau)

    def forward(self, features):
        if len(features) != 3:
            raise ValueError('CWD adapter needs all three Detect-input features')
        return [adapter(feature) for (adapter, feature) in zip(self.adapters, features)]
