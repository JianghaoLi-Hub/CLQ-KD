# CSAKD YOLO11 adaptation; source: https://github.com/KefanZhan/YOLOv8-KD.
# See THIRD_PARTY.md. The framework license does not relicense this reference.
"""CSAKD author executable loss, with exact actual-batch AAWM aggregation.

Reference: KefanZhan/YOLOv8-KD at 3433559ab6c9cfdef3b4ac099f1bb2feddd42dc8.
The author uses BCHW matmul (B,C,H,H attention), channel normalization,
mean squared errors, and batch-mean dot BEFORE sigmoid. Alpha/beta and the
chosen outer actual-B factor belong to the trainer, not this module.
"""
import torch
from torch import nn
from torch.nn import functional as F

class CSAKDLoss(nn.Module):
    """Pure aligned-feature author core, exposing sufficient sample statistics."""
    normalization_eps = 1e-12

    def per_sample_parts(self, student_features, teacher_features, cross_features):
        if isinstance(student_features, torch.Tensor):
            (student_features, teacher_features, cross_features) = ((student_features,), (teacher_features,), (cross_features,))
        if not student_features or not len(student_features) == len(teacher_features) == len(cross_features):
            raise ValueError('CSAKD requires corresponding nonempty feature levels')
        (agka_parts, cross_parts, dots) = ([], [], [])
        batch = student_features[0].shape[0]
        for (student, teacher, cross) in zip(student_features, teacher_features, cross_features):
            if any((x.ndim != 4 or any((n <= 0 for n in x.shape)) for x in (student, teacher, cross))):
                raise ValueError('CSAKD requires nonempty BCHW tensors')
            if student.shape != teacher.shape or student.shape != cross.shape or student.shape[0] != batch:
                raise ValueError('CSAKD aligned feature geometry mismatch; interpolation is forbidden')
            if student.device != teacher.device or student.device != cross.device:
                raise ValueError('CSAKD feature devices must match')
            target = teacher.detach()
            (q, k) = (F.relu(cross), F.relu(target))
            attention = F.softmax(torch.matmul(q, k.transpose(-2, -1)) / k.shape[1] ** 0.5, dim=-1)
            aligned = (student + torch.matmul(attention, student)) / 2
            norm = lambda tensor: F.normalize(tensor, p=2, dim=1, eps=self.normalization_eps)
            agka_parts.append((norm(aligned) - norm(target)).pow(2).mean(1).mean((1, 2)))
            cross_parts.append((norm(student) - norm(cross)).pow(2).mean(1).mean((1, 2)))
            (tea_gap, stu_gap) = (target.mean(3).mean(2), student.mean(3).mean(2))
            denominator = torch.sqrt(torch.tensor(tea_gap.shape[1], dtype=torch.float32, device=tea_gap.device))
            dots.append((tea_gap * stu_gap).sum(dim=1) / denominator)
        return (torch.stack(agka_parts, dim=1), torch.stack(cross_parts, dim=1), torch.stack(dots, dim=1))

    @staticmethod
    def aggregate_parts(agka, cross_unweighted, dots):
        if agka.ndim != 2 or agka.shape != cross_unweighted.shape or agka.shape != dots.shape or (min(agka.shape) <= 0):
            raise ValueError('CSAKD sample statistics must be identical nonempty BxL tensors')
        (agka_mean, cross_mean, dot_mean) = (agka.mean(dim=0), cross_unweighted.mean(dim=0), dots.mean(dim=0))
        weights = 1 - torch.sigmoid(dot_mean)
        cross = cross_mean * weights
        return (agka_mean.sum(), cross.sum(), dict(agka=agka_mean, cross_unweighted=cross_mean, dot=dot_mean, weights=weights, cross=cross))

    def forward(self, student_features, teacher_features, cross_features, return_parts=False):
        (agka, cross, parts) = self.aggregate_parts(*self.per_sample_parts(student_features, teacher_features, cross_features))
        return (agka, cross, parts) if return_parts else (agka, cross)

class CSAKDModules(CSAKDLoss):
    """Same-architecture identity alignment; no trainable auxiliary modules.

    The author's peer-model path uses the original student features directly.
    Keeping identity alignment explicit shares exactly the same Z_i tensor
    between the cross graph and loss, without adding projections or parameters.
    """

    def __init__(self, student_channels, teacher_channels):
        super().__init__()
        if not student_channels or len(student_channels) != len(teacher_channels):
            raise ValueError('CSAKD requires corresponding nonempty channel lists')
        if any((not isinstance(ch, int) or ch <= 0 for ch in [*student_channels, *teacher_channels])):
            raise ValueError('CSAKD channel counts must be positive integers')
        (self.student_channels, self.teacher_channels) = (list(student_channels), list(teacher_channels))
        if self.student_channels != self.teacher_channels:
            raise ValueError('Peer CSAKD requires identical channels; learned projection is forbidden')
        self.adapters = nn.ModuleList([nn.Identity() for _ in student_channels])

    def align(self, student_features):
        if len(student_features) != len(self.adapters):
            raise ValueError('CSAKD requires all configured student levels')
        aligned = []
        for (index, (student, adapter)) in enumerate(zip(student_features, self.adapters)):
            if student.ndim != 4 or student.shape[1] != self.student_channels[index]:
                raise ValueError('CSAKD student feature does not match registered identity alignment')
            aligned.append(adapter(student))
        return aligned
