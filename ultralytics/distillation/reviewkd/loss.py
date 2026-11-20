# ReviewKD YOLO11 adaptation; source: https://github.com/JIA-Lab-research/ReviewKD.
# See THIRD_PARTY.md. The framework license does not relicense this reference.
"""Detection ReviewKD ABF and hierarchical context loss, without extra normalization."""
import torch
from torch import nn
from torch.nn import functional as F

class ABF(nn.Module):
    """Fuse the deeper pre-conv2 residual into the current student feature."""

    def __init__(self, in_channel, mid_channel, out_channel, fuse):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channel, mid_channel, kernel_size=1, bias=False), nn.BatchNorm2d(mid_channel))
        self.conv2 = nn.Sequential(nn.Conv2d(mid_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(out_channel))
        self.att_conv = nn.Sequential(nn.Conv2d(mid_channel * 2, 2, kernel_size=1), nn.Sigmoid()) if fuse else None
        nn.init.kaiming_uniform_(self.conv1[0].weight, a=1)
        nn.init.kaiming_uniform_(self.conv2[0].weight, a=1)

    def forward(self, x, y=None):
        (n, _, h, w) = x.shape
        x = self.conv1(x)
        if self.att_conv is not None:
            if y is None:
                raise ValueError('A shallower ABF requires its deeper residual')
            y = F.interpolate(y, size=(h, w), mode='nearest')
            gates = self.att_conv(torch.cat([x, y], dim=1))
            x = x * gates[:, 0].view(n, 1, h, w) + y * gates[:, 1].view(n, 1, h, w)
        elif y is not None:
            raise ValueError('The deepest ABF must not receive a residual')
        return (self.conv2(x), x)

class ReviewKDLoss(nn.Module):
    """Single-level HCL: full map and 4/2/1 pooled contexts, weighted and normalized."""

    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.loss_weight = float(loss_weight)

    def forward(self, student, teacher):
        if not isinstance(student, torch.Tensor) or not isinstance(teacher, torch.Tensor):
            raise TypeError('ReviewKDLoss expects one aligned tensor pair per call')
        if student.ndim != 4 or student.shape != teacher.shape:
            raise ValueError(f'ReviewKD aligned BCHW shapes differ: {student.shape} vs {teacher.shape}')
        loss = F.mse_loss(student, teacher, reduction='mean')
        (weight, normalizer) = (1.0, 1.0)
        for size in (4, 2, 1):
            if size >= student.shape[2]:
                continue
            pooled_student = F.adaptive_avg_pool2d(student, (size, size))
            pooled_teacher = F.adaptive_avg_pool2d(teacher, (size, size))
            weight /= 2.0
            loss = loss + F.mse_loss(pooled_student, pooled_teacher, reduction='mean') * weight
            normalizer += weight
        return loss / normalizer * self.loss_weight

class ReviewKDAdapters(nn.Module):
    """Train-only ABFs with official detection mid-channel width 256.

    The mathematical module accepts paired pyramids for official five-level
    equivalence tests; the YOLO capture/training layer separately enforces three.
    """
    mid_channel = 256

    def __init__(self, student_channels, teacher_channels, mid_channels=256):
        super().__init__()
        self.mid_channel = int(mid_channels)
        if not student_channels or len(student_channels) != len(teacher_channels):
            raise ValueError('ReviewKD requires nonempty paired feature channel lists')
        if any((int(c) != c or c <= 0 for c in [*student_channels, *teacher_channels])):
            raise ValueError('Feature channel counts must be positive integers')
        self.student_channels = list(student_channels)
        self.teacher_channels = list(teacher_channels)
        blocks = [ABF(sc, self.mid_channel, tc, index < len(student_channels) - 1) for (index, (sc, tc)) in enumerate(zip(student_channels, teacher_channels))]
        self.abfs = nn.ModuleList(blocks[::-1])
        self.loss = ReviewKDLoss(loss_weight=1.0)

    def forward(self, features):
        if len(features) != len(self.abfs):
            raise ValueError('ReviewKD requires all configured feature levels')
        for (feature, channels) in zip(features, self.student_channels):
            if feature.ndim != 4 or feature.shape[1] != channels:
                raise ValueError('ReviewKD student feature channel count differs from construction')
        if any((feature.shape[0] != features[0].shape[0] for feature in features)):
            raise ValueError('ReviewKD feature levels must share one batch')
        if any((a.shape[2] < b.shape[2] or a.shape[3] < b.shape[3] for (a, b) in zip(features, features[1:]))):
            raise ValueError('ReviewKD expects shallow-to-deep feature ordering')
        (outputs, residual) = ([], None)
        for (abf, feature) in zip(self.abfs, reversed(features)):
            (output, residual) = abf(feature, residual)
            outputs.insert(0, output)
        return outputs
