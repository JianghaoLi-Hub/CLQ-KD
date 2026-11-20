"""Detached teacher targets and same-assignment matched-class filtering."""
import torch
import torch.nn.functional as F
from ultralytics.utils.tal import make_anchors


@torch.no_grad()
def teacher_outputs(predictions, head):
    features = predictions[1] if isinstance(predictions, tuple) else predictions
    if not isinstance(features, list) or len(features) != 3:
        raise ValueError("CLQ-KD requires native three-scale Detect outputs")
    distribution, logits = torch.cat([x.view(x.shape[0], head.no, -1) for x in features], 2).split(
        (4 * head.reg_max, head.nc), 1)
    anchors, strides = make_anchors(features, head.stride, 0.5)
    boxes = head.decode_bboxes(head.dfl(distribution), anchors.T[None], xywh=False) * strides.T
    logits, boxes = logits.transpose(1, 2).contiguous(), boxes.transpose(1, 2).contiguous()
    if logits.dtype != torch.float32 or boxes.dtype != torch.float32:
        raise ValueError("CLQ-KD teacher outputs must be FP32")
    if not torch.isfinite(logits).all() or not torch.isfinite(boxes).all():
        raise FloatingPointError("Nonfinite CLQ-KD teacher output")
    return logits.detach(), boxes.detach()


def filtered_loss(student, teacher, context, settings):
    """Return BCE/A (before alpha, cls and batch gains), plus detached counts."""
    if context is None or teacher is None:
        raise RuntimeError("CLQ-KD requires same-forward TAL context and teacher outputs")
    fg, labels, gt = context
    logits, boxes = teacher
    if student.shape != logits.shape or boxes.shape != (*student.shape[:2], 4):
        raise ValueError("CLQ-KD teacher/student positions or classes mismatch")
    if fg.shape != student.shape[:2] or gt.shape != boxes.shape:
        raise ValueError("CLQ-KD TAL context shape mismatch")
    with torch.no_grad():
        raw = 1 / (1 + torch.exp(-logits.detach()))
        target = 1 / (1 + torch.exp(-logits.detach() / settings['clqkd_teacher_temperature']))
        b, a = torch.where(fg.detach().bool())
        c = labels.detach()[b, a].long()
        t, g = boxes.detach()[b, a], gt.detach()[b, a]
        intersection = (torch.minimum(t[:, 2:], g[:, 2:]) - torch.maximum(t[:, :2], g[:, :2])).clamp_min(0).prod(1)
        union = (t[:, 2:] - t[:, :2]).clamp_min(0).prod(1) + (g[:, 2:] - g[:, :2]).clamp_min(0).prod(1) - intersection
        # Stabilize the IoU denominator for zero-area boxes.
        iou = intersection / (union + 1e-7)
        keep = (raw[b, a, c] > settings['clqkd_prob_threshold']) & (iou >= settings['clqkd_iou_threshold'])
        mask = torch.ones_like(student)
        mask[b[~keep], a[~keep], c[~keep]] = 0
    probability = 1 / (1 + torch.exp(-student / settings['clqkd_student_temperature']))
    loss = (F.binary_cross_entropy(probability, target, reduction='none') * mask).sum() / student.shape[1]
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite CLQ-KD loss")
    return loss, {'foreground': int(b.numel()), 'kept': int(keep.sum()), 'ignored': int((~keep).sum())}
