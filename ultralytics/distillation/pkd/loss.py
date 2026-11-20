# Copyright (c) OpenMMLab. All rights reserved.
# Apache-2.0; see licenses/PKD-Apache-2.0.txt.
# Modified: YOLO11 channel adapters and shared training integration.
"""PKD normalization and reduction with YOLO channel adapters."""
import torch
from torch import nn
from torch.nn import functional as F

def norm(feature):
    """Official per-channel BHW standardization using sample standard deviation."""
    if feature.ndim != 4 or feature.shape[0] * feature.shape[2] * feature.shape[3] <= 1:
        raise ValueError('PKD needs BCHW tensors with more than one BHW observation')
    (n, c, h, w) = feature.shape
    values = feature.permute(1, 0, 2, 3).reshape(c, -1)
    mean = values.mean(dim=-1, keepdim=True)
    std = values.std(dim=-1, keepdim=True)
    return ((values - mean) / (std + 1e-06)).reshape(c, n, h, w).permute(1, 0, 2, 3)

class PKDLoss(nn.Module):
    """No masks, no temperatures, no layer averaging or hidden batch multiplier."""

    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.loss_weight = float(loss_weight)

    @staticmethod
    def norm(feature):
        return norm(feature)

    def forward(self, preds_S, preds_T):
        if isinstance(preds_S, torch.Tensor):
            (preds_S, preds_T) = ((preds_S,), (preds_T,))
        if len(preds_S) != len(preds_T) or not preds_S:
            raise ValueError('PKD requires nonempty corresponding student/teacher levels')
        losses = []
        for (student, teacher) in zip(preds_S, preds_T):
            if student.shape != teacher.shape:
                raise ValueError(f'PKD aligned feature shapes differ: {student.shape} vs {teacher.shape}')
            losses.append(F.mse_loss(norm(student), norm(teacher)) / 2)
        return sum(losses) * self.loss_weight

class PKDAdapters(nn.Module):
    """Training-only 1x1 no-bias projections; not attached to student or EMA."""

    def __init__(self, student_channels, teacher_channels):
        super().__init__()
        if len(student_channels) != 3 or len(teacher_channels) != 3:
            raise ValueError('PKD requires three feature levels: P3, P4 and P5')
        self.student_channels = list(student_channels)
        self.teacher_channels = list(teacher_channels)
        self.adapters = nn.ModuleList([nn.Conv2d(student, teacher, 1, bias=False) if student != teacher else nn.Identity() for (student, teacher) in zip(student_channels, teacher_channels)])
        self.loss = PKDLoss(loss_weight=1.0)

    def forward(self, features):
        if len(features) != 3:
            raise ValueError('PKD adapter needs all three Detect-input features')
        return [adapter(feature) for (adapter, feature) in zip(self.adapters, features)]
