# MGD-derived loss, Apache-2.0; see licenses/MGD-Apache-2.0.txt.
# Modified: YOLO11 feature adaptation and shared training integration.
"""Masked generative distillation: author detection SSE/B, no internal alpha."""
import math
import torch
from torch import nn

def validate_mask_ratio(value):
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError('MGD mask_ratio must be finite in [0, 1]')
    return value

class MGDLoss(nn.Module):
    """Full reconstructed-teacher squared error, summed over all entries / B.

    No reduction by channels, pixels or number of levels. Alpha belongs only
    in the trainer's explicit outer B*alpha scaling. Teacher targets detach.
    """

    def forward(self, preds_S, preds_T):
        if isinstance(preds_S, torch.Tensor):
            (preds_S, preds_T) = ((preds_S,), (preds_T,))
        if len(preds_S) != len(preds_T) or not preds_S:
            raise ValueError('MGD requires nonempty corresponding reconstruction/teacher levels')
        losses = []
        for (reconstruction, teacher) in zip(preds_S, preds_T):
            if reconstruction.ndim != 4 or teacher.ndim != 4 or any((n <= 0 for n in reconstruction.shape)):
                raise ValueError('MGD requires nonempty BCHW features')
            if reconstruction.shape != teacher.shape:
                raise ValueError(f'MGD feature geometry mismatch: {reconstruction.shape} vs {teacher.shape}')
            losses.append(nn.functional.mse_loss(reconstruction, teacher.detach(), reduction='sum') / reconstruction.shape[0])
        return sum(losses)

class MGDModules(nn.Module):
    """Train-only biased alignment + independent Conv3/ReLU/Conv3 generators.

    Masks are per sample and spatial position, shared across channels. The
    author's CPU torch.rand -> device -> torch.where construction is retained.
    Fixed binary masks support deterministic loss and gradient tests.
    """

    def __init__(self, student_channels, teacher_channels, mask_ratio=0.65):
        super().__init__()
        if len(student_channels) != len(teacher_channels) or not student_channels:
            raise ValueError('MGD requires corresponding nonempty channel lists')
        if any((not isinstance(ch, int) or ch <= 0 for ch in [*student_channels, *teacher_channels])):
            raise ValueError('MGD channel counts must be positive integers')
        (self.student_channels, self.teacher_channels) = (list(student_channels), list(teacher_channels))
        self.mask_ratio = validate_mask_ratio(mask_ratio)
        self.adapters = nn.ModuleList([nn.Conv2d(student, teacher, kernel_size=1, stride=1, padding=0, bias=True) if student != teacher else nn.Identity() for (student, teacher) in zip(student_channels, teacher_channels)])
        self.generators = nn.ModuleList([nn.Sequential(nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True), nn.ReLU(inplace=True), nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)) for channels in teacher_channels])
        self.loss = MGDLoss()

    def make_mask(self, feature):
        if feature.ndim != 4 or any((n <= 0 for n in feature.shape)):
            raise ValueError('MGD mask requires nonempty BCHW features')
        (batch, _, height, width) = feature.shape
        random = torch.rand((batch, 1, height, width)).to(feature.device)
        return torch.where(random > 1.0 - self.mask_ratio, 0, 1)

    def forward(self, student_features, teacher_features, masks=None, return_parts=False):
        if isinstance(student_features, torch.Tensor):
            (student_features, teacher_features) = ((student_features,), (teacher_features,))
        levels = len(self.adapters)
        if len(student_features) != levels or len(teacher_features) != levels:
            raise ValueError('MGD requires every configured student/teacher level')
        if isinstance(masks, torch.Tensor):
            masks = (masks,)
        if masks is not None and len(masks) != levels:
            raise ValueError('MGD fixed masks must cover every configured level')
        parts = []
        for (index, (student, teacher, adapter, generator)) in enumerate(zip(student_features, teacher_features, self.adapters, self.generators)):
            if student.ndim != 4 or teacher.ndim != 4 or any((n <= 0 for n in student.shape)):
                raise ValueError('MGD requires nonempty BCHW features')
            if student.shape[1] != self.student_channels[index] or teacher.shape[1] != self.teacher_channels[index]:
                raise ValueError('MGD feature channels do not match registered projections')
            if student.shape[0] != teacher.shape[0] or student.shape[2:] != teacher.shape[2:]:
                raise ValueError('MGD spatial/batch mismatch; interpolation is forbidden')
            aligned = adapter(student)
            mask = self.make_mask(aligned) if masks is None else masks[index]
            expected = (aligned.shape[0], 1, aligned.shape[2], aligned.shape[3])
            if not isinstance(mask, torch.Tensor) or tuple(mask.shape) != expected:
                raise ValueError(f'MGD mask must be Bx1xHxW: expected {expected}')
            if mask.device != aligned.device:
                raise ValueError('MGD fixed mask device must match its feature')
            if masks is not None:
                if mask.requires_grad or not bool(torch.isfinite(mask).all()) or (not bool(((mask == 0) | (mask == 1)).all())):
                    raise ValueError('MGD fixed masks must be finite binary tensors without gradients')
            reconstruction = generator(aligned * mask)
            parts.append(self.loss(reconstruction, teacher))
        raw = sum(parts)
        return (raw, parts) if return_parts else raw
