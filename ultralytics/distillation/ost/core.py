# Attention-based Feature-level Distillation
# Copyright (c) 2021-present NAVER Corp.
# Apache License v2.0; see licenses/OST-Apache-2.0.txt.
# Modified: compact YOLO11 features and shared training integration.
"""Numerically faithful OST core using compact, differentiable hook outputs.

Reference: GHOST commit 89b4d9213e082a4de8e3dca1b3a47f5afbb13d1c,
utils/distill/Self_D.py. Spatial values are channel means of squared features,
L2 normalized before comparison. There is no interpolation or layer averaging.
SCM remains a separate train-only module, never registered inside the student.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
STAGE_COUNT = 23
NORMALIZE_EPS = 1e-12

def compact_feature(feature):
    """Reduce BCHW to official GAP and normalized energy, preserving autograd."""
    if not isinstance(feature, torch.Tensor) or feature.ndim != 4:
        raise TypeError('OST stages must produce BCHW tensors')
    (batch, channels, height, width) = feature.shape
    return {'gap': feature.mean(3).mean(2), 'energy': F.normalize(feature.pow(2).mean(1).view(batch, -1), dim=1, eps=NORMALIZE_EPS), 'spatial': (height, width), 'channels': channels}

def _gumbel_sigmoid(logits, tau=1.0, threshold=0.5, training=True, noise=None, gate_override=None):
    """Official single-Gumbel straight-through sigmoid; overrides are test-only."""
    if noise is not None and noise.shape != logits.shape:
        raise ValueError('Fixed Gumbel noise must cover the full B x 23 x 23 matrix')
    if training:
        if noise is None:
            noise = -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
        soft = ((logits + noise) / tau).sigmoid()
    else:
        if noise is not None:
            raise ValueError('Explicit Gumbel noise is only supported in training mode')
        soft = (logits / tau).sigmoid()
    if gate_override is None:
        hard = torch.zeros_like(logits).masked_fill(soft > threshold, 1.0)
    else:
        if gate_override.shape != logits.shape:
            raise ValueError('Test gate override must cover the full SCM matrix')
        hard = gate_override.to(device=logits.device, dtype=logits.dtype)
    return hard - soft.detach() + soft

class nn_bn_relu(nn.Module):

    def __init__(self, nin, nout):
        super().__init__()
        self.linear = nn.Linear(nin, nout)
        self.bn = nn.BatchNorm1d(nout)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x, relu=True):
        x = self.bn(self.linear(x))
        return self.relu(x) if relu else x

class LinearTransformTeacher(nn.Module):

    def __init__(self, channels, qk_dim):
        super().__init__()
        self.query_layer = nn.ModuleList([nn_bn_relu(c, qk_dim) for c in channels])

    def forward(self, features):
        query = torch.stack([layer(f['gap'], relu=False) for (f, layer) in zip(features, self.query_layer)], dim=1)
        return (query, [f['energy'] for f in features])

class LinearTransformStudent(nn.Module):

    def __init__(self, channels, qk_dim):
        super().__init__()
        self.qk_dim = qk_dim
        self.relu = nn.ReLU(inplace=False)
        self.key_layer = nn.ModuleList([nn_bn_relu(c, qk_dim) for c in channels])

    def forward(self, features):
        key = torch.stack([layer(f['gap']) for (f, layer) in zip(features, self.key_layer)], dim=1)
        return (key, [f['energy'] for f in features])

class Attention(nn.Module):

    def __init__(self, teacher_channels, student_channels, qk_dim, temperature, threshold):
        super().__init__()
        self.qk_dim = qk_dim
        self.temperature = temperature
        self.threshold = threshold
        self.linear_trans_s = LinearTransformStudent(student_channels, qk_dim)
        self.linear_trans_t = LinearTransformTeacher(teacher_channels, qk_dim)

    def forward(self, student, teacher, noise=None, gate_override=None):
        (query, teacher_values) = self.linear_trans_t(teacher)
        (key, student_values) = self.linear_trans_s(student)
        logits = torch.einsum('bsq,btq->bts', key, query) / math.sqrt(self.qk_dim)
        switches = _gumbel_sigmoid(logits, self.temperature, self.threshold, self.training, noise, gate_override)
        gates = switches.diagonal(dim1=1, dim2=2)
        losses = [((student_value - teacher_value).pow(2).mean(1) * gates[:, i]).mean() for (i, (student_value, teacher_value)) in enumerate(zip(student_values, teacher_values))]
        return (losses, gates)

class OSTLoss(nn.Module):
    """23-stage OST loss with no batch factor and no external loss coefficient.

    The trainer must add ``actual_batch * configured_weight * raw_loss`` exactly once to the
    native batch-scaled detection loss. Eval-mode switches are deterministic
    diagnostics; training reproduces the official stochastic switches.
    """

    def __init__(self, teacher_channels, student_channels, qk_dim=128, temperature=1.0, threshold=0.5):
        super().__init__()
        self.teacher_channels = tuple((int(c) for c in teacher_channels))
        self.student_channels = tuple((int(c) for c in student_channels))
        if len(self.teacher_channels) != STAGE_COUNT or len(self.student_channels) != STAGE_COUNT:
            raise ValueError('This protocol requires all 23 corresponding pre-Detect stages (0-22)')
        if qk_dim < 1 or temperature <= 0 or not 0 < threshold < 1:
            raise ValueError('OST requires positive qk_dim/temperature and threshold in (0, 1)')
        self.guide_layers = list(range(STAGE_COUNT))
        self.hint_layers = list(range(STAGE_COUNT))
        self.attention = Attention(self.teacher_channels, self.student_channels, qk_dim, temperature, threshold)

    def _validate(self, student, teacher):
        if len(student) != STAGE_COUNT or len(teacher) != STAGE_COUNT:
            raise ValueError('Missing or extra OST stages')
        batch = student[0]['gap'].shape[0]
        for (i, (s, t)) in enumerate(zip(student, teacher)):
            if tuple(s['spatial']) != tuple(t['spatial']):
                raise ValueError(f'Stage {i}: unequal spatial shapes; automatic interpolation is prohibited')
            if s['channels'] != self.student_channels[i] or t['channels'] != self.teacher_channels[i]:
                raise ValueError(f'Stage {i}: channel signature changed')
            pixels = math.prod(s['spatial'])
            if s['energy'].shape != (batch, pixels) or t['energy'].shape != (batch, pixels):
                raise ValueError(f'Stage {i}: invalid normalized spatial energy shape')
            if s['gap'].shape != (batch, self.student_channels[i]) or t['gap'].shape != (batch, self.teacher_channels[i]):
                raise ValueError(f'Stage {i}: invalid GAP shape')

    def forward(self, student, teacher, noise=None, gate_override=None):
        self._validate(student, teacher)
        (losses, gates) = self.attention(student, teacher, noise, gate_override)
        total = sum(losses)
        return total

    def forward_features(self, student, teacher, noise=None, gate_override=None):
        """Full-feature convenience path used only by equivalence tests."""
        return self([compact_feature(x) for x in student], [compact_feature(x) for x in teacher], noise, gate_override)
