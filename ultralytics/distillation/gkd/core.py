# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Differentiable first-order gradient-guided distillation maps.

This is a paper-guided YOLO adaptation, not an end-to-end reproduction of the
official MMDetection CAM pipeline. Gradient weights are signed spatial means
and are treated as constants. Only the student's feature path is differentiated.
The trainer, not this module, applies the actual batch size and outer coefficient.
"""
from __future__ import annotations
from collections.abc import Sequence
import math
import torch
from torch import Tensor

def validate_layer_weights(values):
    """Exactly three positive finite scalar weights in P3/P4/P5 order; no rescaling."""
    values = tuple((float(value) for value in values))
    if len(values) != 3 or any((not math.isfinite(value) or value <= 0 for value in values)):
        raise ValueError('layer_weights must contain three positive finite P3/P4/P5 values')
    return values

def gradient_guided_map(feature: Tensor, gradient: Tensor, eps: float=1e-06) -> Tensor:
    """Return one BxHxW normalized map, preserving the feature autograd path."""
    if feature.ndim != 4 or feature.numel() == 0:
        raise ValueError('Features must be nonempty BxCxHxW tensors')
    if gradient.shape != feature.shape:
        raise ValueError('Each feature gradient must have exactly its feature shape')
    if feature.device != gradient.device or feature.dtype != gradient.dtype:
        raise ValueError('Features and their gradients must share device and dtype')
    if not feature.is_floating_point() or not math.isfinite(eps) or eps <= 0:
        raise ValueError('Floating-point features and a positive finite eps are required')
    weights = gradient.detach().mean(dim=(-2, -1), keepdim=True)
    weighted = feature * weights
    channel_sum = weighted.sum(dim=1, dtype=torch.float64).to(feature.dtype)
    absolute_map = channel_sum.abs()
    minimum = absolute_map.amin(dim=(-2, -1), keepdim=True)
    maximum = absolute_map.amax(dim=(-2, -1), keepdim=True)
    return (absolute_map - minimum) / (maximum - minimum).clamp_min(eps)

def gkd_loss(student_features: Sequence[Tensor], teacher_features: Sequence[Tensor], student_gradients: Sequence[Tensor], teacher_gradients: Sequence[Tensor], eps: float=1e-06, layer_weights: Sequence[float]=(1.0, 1.0, 1.0)) -> tuple[Tensor, dict[str, Tensor]]:
    """Sum three layer-wise mean(B,H,W) map L1 losses without outer scaling.

    P3/P4/P5 must match in batch and spatial dimensions, but need not match in
    channel count. Teacher features and both gradient-weight paths are detached.
    """
    layer_weights = validate_layer_weights(layer_weights)
    groups = (student_features, teacher_features, student_gradients, teacher_gradients)
    if any((len(group) != 3 for group in groups)):
        raise ValueError('Exactly three ordered P3/P4/P5 tensors are required per group')
    terms: list[Tensor] = []
    diagnostics: dict[str, Tensor] = {}
    expected_batch = student_features[0].shape[0] if student_features[0].ndim == 4 else None
    for (level, weight, (student, teacher, grad_s, grad_t)) in zip(('p3', 'p4', 'p5'), layer_weights, zip(*groups)):
        if student.ndim != 4 or teacher.ndim != 4:
            raise ValueError(f'{level}: expected BxCxHxW feature tensors')
        if student.shape[0] != expected_batch or teacher.shape[0] != expected_batch:
            raise ValueError(f'{level}: all layers and both models must use the same batch')
        if student.shape[-2:] != teacher.shape[-2:]:
            raise ValueError(f'{level}: spatial mismatch; interpolation is not permitted')
        if student.device != teacher.device or student.dtype != teacher.dtype:
            raise ValueError(f'{level}: student and teacher must share device and dtype')
        student_map = gradient_guided_map(student, grad_s, eps)
        teacher_map = gradient_guided_map(teacher.detach(), grad_t, eps).detach()
        layer_loss = (student_map - teacher_map).abs().mean()
        terms.append(weight * layer_loss)
    raw_loss = torch.stack(terms).sum()
    return (raw_loss, diagnostics)
