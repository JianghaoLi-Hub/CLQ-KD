# FGD / OpenMMLab-derived feature loss, Apache-2.0.
# Copyright 2018-2019 Open-MMLab. All rights reserved.
# Upstream Bytedance modifications: Copyright 2021 Bytedance Inc.
# Modified: YOLO11 boxes, feature adapters and shared training integration.
# See licenses/FGD-Apache-2.0.txt for the complete upstream notices.
"""Focal and Global Knowledge Distillation for YOLO multi-scale features."""
import torch
import torch.nn as nn
import torch.nn.functional as F

def xywh2xyxy(boxes):
    (x, y, width, height) = boxes.unbind(-1)
    return torch.stack((x - width / 2, y - height / 2, x + width / 2, y + height / 2), dim=-1)

class FeatureLoss(nn.Module):
    """Apply the official FGD foreground, background, attention and relation losses."""

    def __init__(self, student_channels, teacher_channels, device=None, temp=0.5, alpha_fgd=0.001, beta_fgd=0.0005, gamma_fgd=0.0005, lambda_fgd=5e-06):
        super().__init__()
        if len(student_channels) != len(teacher_channels):
            raise ValueError('FGD requires paired student and teacher feature scales')
        self.temp = temp
        self.alpha_fgd = alpha_fgd
        self.beta_fgd = beta_fgd
        self.gamma_fgd = gamma_fgd
        self.lambda_fgd = lambda_fgd
        self.last_components = {}
        self.align = nn.ModuleList((nn.Identity() if student == teacher else nn.Conv2d(student, teacher, kernel_size=1, bias=True) for (student, teacher) in zip(student_channels, teacher_channels)))
        self.conv_mask_s = nn.ModuleList((nn.Conv2d(channels, 1, 1) for channels in teacher_channels))
        self.conv_mask_t = nn.ModuleList((nn.Conv2d(channels, 1, 1) for channels in teacher_channels))
        self.channel_add_conv_s = nn.ModuleList((self._context_block(channels) for channels in teacher_channels))
        self.channel_add_conv_t = nn.ModuleList((self._context_block(channels) for channels in teacher_channels))
        self.reset_parameters()
        if device is not None:
            self.to(device)

    @staticmethod
    def _context_block(channels):
        hidden = max(channels // 2, 1)
        return nn.Sequential(nn.Conv2d(channels, hidden, 1), nn.LayerNorm([hidden, 1, 1]), nn.ReLU(inplace=True), nn.Conv2d(hidden, channels, 1))

    @staticmethod
    def _split_boxes(boxes, batch_indices, batch_size):
        boxes = xywh2xyxy(boxes).clamp_(0, 1)
        batch_indices = batch_indices.reshape(-1).long()
        return [boxes[batch_indices == index] for index in range(batch_size)]

    @staticmethod
    def _build_masks(boxes_per_image, height, width, reference):
        foreground = reference.new_zeros((len(boxes_per_image), height, width))
        background = reference.new_ones((len(boxes_per_image), height, width))
        for (image_index, boxes) in enumerate(boxes_per_image):
            if boxes.numel() == 0:
                background[image_index] /= background[image_index].sum().clamp_min(1)
                continue
            x_min = torch.floor(boxes[:, 0] * width).long().clamp_(0, width)
            y_min = torch.floor(boxes[:, 1] * height).long().clamp_(0, height)
            x_max = torch.ceil(boxes[:, 2] * width).long().clamp_(0, width)
            y_max = torch.ceil(boxes[:, 3] * height).long().clamp_(0, height)
            inverse_area = 1.0 / ((x_max - x_min + 1) * (y_max - y_min + 1)).clamp_min(1)
            for box_index in range(boxes.shape[0]):
                region = foreground[image_index, y_min[box_index]:y_max[box_index] + 1, x_min[box_index]:x_max[box_index] + 1]
                foreground[image_index, y_min[box_index]:y_max[box_index] + 1, x_min[box_index]:x_max[box_index] + 1] = torch.maximum(region, inverse_area[box_index].to(region.dtype))
            background[image_index] = (foreground[image_index] == 0).to(background.dtype)
            background[image_index] /= background[image_index].sum().clamp_min(1)
        return (foreground, background)

    def forward(self, preds_S, preds_T, gt_bboxes, img_metas):
        if len(preds_S) != len(self.align) or len(preds_T) != len(self.align):
            raise ValueError(f'FGD expected {len(self.align)} scales, got {len(preds_S)} student and {len(preds_T)} teacher')
        preds_S = [adapter(feature) for (adapter, feature) in zip(self.align, preds_S)]
        preds_T = [feature.detach() for feature in preds_T]
        boxes_per_image = self._split_boxes(gt_bboxes, img_metas['batch_idx'], preds_S[0].shape[0])
        component_losses = {'foreground': preds_S[0].float().new_zeros(()), 'background': preds_S[0].float().new_zeros(()), 'attention': preds_S[0].float().new_zeros(()), 'relation': preds_S[0].float().new_zeros(())}
        for (level, (student, teacher)) in enumerate(zip(preds_S, preds_T)):
            if student.shape != teacher.shape:
                raise ValueError(f'FGD feature shapes differ at level {level}: {student.shape} vs {teacher.shape}')
            (_, _, height, width) = student.shape
            (spatial_t, channel_t) = self.get_attention(teacher, self.temp)
            (spatial_s, channel_s) = self.get_attention(student, self.temp)
            (foreground, background) = self._build_masks(boxes_per_image, height, width, spatial_t)
            (foreground_loss, background_loss) = self.get_feature_loss(student, teacher, foreground, background, channel_t, spatial_t)
            attention_loss = self.get_attention_loss(channel_s, channel_t, spatial_s, spatial_t)
            relation_loss = self.get_relation_loss(student, teacher, level)
            component_losses['foreground'] = component_losses['foreground'] + self.alpha_fgd * foreground_loss
            component_losses['background'] = component_losses['background'] + self.beta_fgd * background_loss
            component_losses['attention'] = component_losses['attention'] + self.gamma_fgd * attention_loss
            component_losses['relation'] = component_losses['relation'] + self.lambda_fgd * relation_loss
        total_loss = sum(component_losses.values())
        self.last_components = {name: value.detach() for (name, value) in component_losses.items()}
        return total_loss

    @staticmethod
    def get_attention(preds, temperature):
        (batch, channels, height, width) = preds.shape
        values = preds.abs().float()
        spatial = (height * width * F.softmax(values.mean(1).reshape(batch, -1) / temperature, dim=1)).reshape(batch, height, width)
        channel = channels * F.softmax(values.mean(dim=(2, 3)) / temperature, dim=1)
        return (spatial, channel)

    @staticmethod
    def get_feature_loss(student, teacher, foreground, background, channel_t, spatial_t):
        (student, teacher) = (student.float(), teacher.float())
        (foreground, background) = (foreground.float(), background.float())
        (channel_t, spatial_t) = (channel_t.float(), spatial_t.float())
        foreground = foreground.unsqueeze(1)
        background = background.unsqueeze(1)
        channel_t = channel_t[:, :, None, None]
        spatial_t = spatial_t.unsqueeze(1)
        attention = torch.sqrt(channel_t.clamp_min(0)) * torch.sqrt(spatial_t.clamp_min(0))
        student = student * attention
        teacher = teacher * attention
        foreground_loss = ((student - teacher) * torch.sqrt(foreground.clamp_min(0))).square().sum() / student.shape[0]
        background_loss = ((student - teacher) * torch.sqrt(background.clamp_min(0))).square().sum() / student.shape[0]
        return (foreground_loss, background_loss)

    @staticmethod
    def get_attention_loss(channel_s, channel_t, spatial_s, spatial_t):
        batch = channel_s.shape[0]
        return ((channel_s.float() - channel_t.float()).abs().sum() + (spatial_s.float() - spatial_t.float()).abs().sum()) / batch

    def spatial_pool(self, feature, level, teacher=False):
        (batch, channels, height, width) = feature.shape
        feature_flat = feature.reshape(batch, channels, height * width).unsqueeze(1)
        mask_module = self.conv_mask_t[level] if teacher else self.conv_mask_s[level]
        mask = F.softmax(mask_module(feature).reshape(batch, 1, height * width).float(), dim=2).to(feature.dtype)
        return torch.matmul(feature_flat, mask.unsqueeze(-1)).reshape(batch, channels, 1, 1)

    def get_relation_loss(self, student, teacher, level):
        (student, teacher) = (student.float(), teacher.float())
        context_s = self.spatial_pool(student, level, teacher=False)
        context_t = self.spatial_pool(teacher, level, teacher=True)
        output_s = student + self.channel_add_conv_s[level](context_s)
        output_t = teacher + self.channel_add_conv_t[level](context_t)
        return F.mse_loss(output_s.float(), output_t.float(), reduction='sum') / student.shape[0]

    @staticmethod
    def _last_zero_init(module):
        nn.init.constant_(module[-1].weight, 0)
        if module[-1].bias is not None:
            nn.init.constant_(module[-1].bias, 0)

    def reset_parameters(self):
        for (student_mask, teacher_mask, student_context, teacher_context) in zip(self.conv_mask_s, self.conv_mask_t, self.channel_add_conv_s, self.channel_add_conv_t):
            nn.init.kaiming_normal_(student_mask.weight, mode='fan_in')
            nn.init.kaiming_normal_(teacher_mask.weight, mode='fan_in')
            if student_mask.bias is not None:
                nn.init.constant_(student_mask.bias, 0)
            if teacher_mask.bias is not None:
                nn.init.constant_(teacher_mask.bias, 0)
            self._last_zero_init(student_context)
            self._last_zero_init(teacher_context)
