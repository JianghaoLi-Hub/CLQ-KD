"""Shared single-device lifecycle; numerical objectives stay in method modules."""
import copy
import json
from contextlib import nullcontext

import torch
from torch import nn
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import attempt_load_one_weight, torch_safe_load
from ultralytics.utils import LOGGER
from . import method_config


def validate_yolo11(model):
    head = model.model[-1]
    if (len(model.model) != 24 or getattr(head, "f", None) != [16, 19, 22]
            or getattr(head, "end2end", False) or type(head).__name__ != "Detect"
            or type(model.model[2]).__name__ != "C3k2"):
        raise ValueError("This release supports native three-scale YOLO11 detection models only")


def validate_teacher(student, teacher, names):
    validate_yolo11(student)
    validate_yolo11(teacher)
    ordered = lambda x: [x[k] for k in sorted(x)] if isinstance(x, dict) else list(x)
    if teacher.model[-1].nc != student.model[-1].nc or ordered(teacher.names) != ordered(names):
        raise ValueError("Teacher/student class count or class order mismatch; provide a matching teacher_model")


class BaselineTrainer(DetectionTrainer):
    """Native detection supervision, with no teacher or KD branches."""
    def _setup_teacher_model(self):
        validate_yolo11(self.model)
        self.teacher_model = None

    def get_teacher_scores(self, imgs):
        return None


class KDTrainer(DetectionTrainer):
    method = None
    slot = None
    grad_clip = 10.0

    def __init__(self, *args, **kwargs):
        self.kd_config = method_config(self.method)
        self.parameters_config = {k: v for k, v in self.kd_config.items() if k != "teacher_model" and not k.startswith("clqkd_")}
        for key, value in self.parameters_config.items():
            setattr(self, key, value)
        super().__init__(*args, **kwargs)
        if getattr(self, '_resume_kd', None):
            self.parameters_config = copy.deepcopy(self._resume_kd['settings'])
            for key, value in self.parameters_config.items():
                setattr(self, key, value)
        from .options import resolve_clqkd
        _, self.clqkd_settings = resolve_clqkd(self.method, vars(self.args))
        self.clqkd_active = bool(self.clqkd_settings)
        self._clqkd_teacher = None
        if self.args.amp or self.args.teacher_amp:
            raise ValueError("KD requires FP32 training (amp=False, teacher_amp=False)")

    def check_resume(self, overrides):
        super().check_resume(overrides)
        self._resume_kd = None
        if self.resume:
            checkpoint, _ = torch_safe_load(self.args.resume)
            state = checkpoint.get('distillation')
            if not state or state.get('method') != self.method:
                raise ValueError('Resume requires a checkpoint for the same distillation method')
            self._resume_kd = state
            self.args.clqkd = state['clqkd'] and self.method != 'clqkd'
            if self.args.clqkd:
                self.args.clqkd_beta = state['clqkd_settings'].get('clqkd_beta', 1.0)
            for key, value in state['clqkd_settings'].items():
                setattr(self.args, key, value)

    def resume_training(self, ckpt):
        """Restore KD state; the online teacher evaluates each new training batch."""
        if ckpt is None or not self.resume:
            return super().resume_training(ckpt)
        state = ckpt['distillation']
        self.model.load_state_dict(state['student'], strict=True)
        if isinstance(self.distiller, nn.Module):
            self.distiller.load_state_dict(state['auxiliary'], strict=True)
        buffers = state.get('teacher_buffers')
        if buffers is not None:
            current = dict(self.teacher_model.named_buffers())
            if current.keys() != buffers.keys():
                raise ValueError('Teacher buffer structure differs from the resume checkpoint')
            with torch.no_grad():
                for name, value in buffers.items():
                    current[name].copy_(value)
        elif self.method == 'clqkd':
            LOGGER.warning('Legacy checkpoint has no teacher BN state; using the supplied original teacher BN.')
        restored = dict(ckpt, optimizer=state['optimizer'])
        # Final inference exports may strip the outer training metadata.
        for key in ('epoch', 'best_fitness', 'updates'):
            if key in state:
                restored[key] = state[key]
        super().resume_training(restored)
        if 'ema' in state and self.ema:
            self.ema.ema.load_state_dict(state['ema'], strict=True)
            self.ema.updates = state['updates']
        self.scheduler.load_state_dict(state['scheduler'])
        self.scaler.load_state_dict(state['scaler'])

    @property
    def distiller(self):
        return getattr(self, self.slot, None) if self.slot else None

    def _setup_teacher_model(self):
        devices = [self.device.index or 0] if self.device.type == "cuda" else []
        # Feature methods isolate teacher/auxiliary initialization RNG. CLQ-KD
        # retains the original native initialization sequence.
        context = torch.random.fork_rng(devices=devices) if self.method != "clqkd" else nullcontext()
        with context:
            teacher, _ = attempt_load_one_weight(self.args.teacher_model, device=self.device, fuse=False)
            self.teacher_model = teacher.float().requires_grad_(False)
            validate_teacher(self.model, teacher, self.data["names"])
            teacher.train(self.method == "clqkd")
            if self.slot:
                module = self.build_distiller()
                if isinstance(module, nn.Module):
                    module = module.to(self.device)
                setattr(self, self.slot, module)
        self.teacher_amp = False

    def _setup_train(self, world_size):
        if world_size > 1:
            raise ValueError("KD supports one training device only")
        super()._setup_train(world_size)
        if self.distiller is not None:
            self.distiller.install_hooks()  # EMA must be created without hooks.
        (self.save_dir / "distillation.json").write_text(json.dumps({
            "method": self.method, "settings": self.parameters_config, "clqkd": self.clqkd_active,
            "clqkd_settings": self.clqkd_settings, "objective_version": 3,
            "teacher_model": str(self.args.teacher_model), "joint_grad_clip": self.grad_clip,
            "teacher_mode": "train" if self.method == "clqkd" else "eval", "resume_supported": True,
        }, indent=2), encoding="utf-8")

    def train(self):
        try:
            return super().train()
        finally:
            if self.distiller is not None:
                self.distiller.remove_hooks()

    def build_optimizer(self, model, **kwargs):
        if isinstance(self.distiller, nn.Module):
            combined = nn.ModuleList([model, self.distiller])
            combined.nc = model.nc
            return super().build_optimizer(combined, **kwargs)
        return super().build_optimizer(model, **kwargs)

    def _model_train(self):
        super()._model_train()
        self.teacher_model.train(self.method == "clqkd")
        if isinstance(self.distiller, nn.Module):
            self.distiller.train()

    def get_teacher_scores(self, imgs):
        raise RuntimeError("Only CLQ-KD may replace classification targets")

    def compute_training_loss(self, batch):
        active = getattr(self, 'clqkd_active', False)
        if active:
            self.prepare_clqkd()
        try:
            native, items, raw, _ = self.compute_loss_components(batch)
            extra = self.clqkd_extra(batch) if active else None
            if active:
                native, items = self.apply_clqkd_beta(native, items)
        finally:
            if active:
                self.clear_clqkd()
        weight = getattr(self, "kd_weight", getattr(self, "mgd_alpha", 1.0))
        weighted = batch["img"].shape[0] * weight * raw
        total = native if self.method == "mgd" and weight == 0 else native + weighted
        if active:
            original = total
            total = total + extra[1] if self.clqkd_settings['clqkd_alpha'] else total
            self.record_clqkd(original, extra[0], extra[1], total, extra[2])
        self.last_native_loss, self.last_kd_loss = native.detach(), raw.detach()
        self.last_weighted_kd_loss, self.last_total_loss = weighted.detach(), total.detach()
        if not torch.isfinite(total):
            raise FloatingPointError(f"Nonfinite {self.method} loss")
        return total, items

    def prepare_clqkd(self):
        if not hasattr(self.model, 'criterion'):
            self.model.criterion = self.model.init_criterion()
        self.model.criterion.capture_clqkd = True
        self.model.criterion.clqkd_context = None
        self.model.criterion.clqkd_native_loss = None
        self._clqkd_teacher = None
        return self.model.criterion

    def apply_clqkd_beta(self, native, items):
        """Scale only native classification, after the original method computes its KD."""
        beta = self.clqkd_settings['clqkd_beta']
        if beta == 1.0:
            return native, items
        parts = self.model.criterion.clqkd_native_loss
        if parts is None or parts.shape != (3,):
            raise RuntimeError('Missing same-forward native box/classification/DFL losses')
        adjusted = parts[0] + beta * parts[1] + parts[2]
        items = items.clone()
        items[1] *= beta
        return adjusted, items

    def capture_clqkd_teacher(self, predictions):
        if not self.clqkd_active:
            return
        if self._clqkd_teacher is not None:
            raise RuntimeError('Duplicate CLQ-KD teacher capture')
        from .clqkd.core import teacher_outputs
        self._clqkd_teacher = teacher_outputs(predictions, self.teacher_model.model[-1])

    def capture_clqkd_features(self, features):
        if self.clqkd_active:
            with torch.no_grad():
                self.capture_clqkd_teacher(self.teacher_model.model[-1](list(features)))

    def clqkd_extra(self, batch):
        from .clqkd.core import filtered_loss
        context = self.model.criterion.clqkd_context
        if context is None:
            raise RuntimeError('Missing same-forward CLQ-KD TAL context')
        raw, counts = filtered_loss(context[0], self._clqkd_teacher, context[1], self.clqkd_settings)
        weighted = raw * len(batch['img']) * self.args.cls * self.clqkd_settings['clqkd_alpha']
        return raw, weighted, counts

    def clear_clqkd(self):
        criterion = self.model.criterion
        criterion.capture_clqkd = False
        criterion.clqkd_context = criterion.clqkd_teacher = criterion.clqkd_settings = None
        criterion.clqkd_native_loss = None
        self._clqkd_teacher = None

    def record_clqkd(self, original, raw, weighted, total, counts, supervised=None):
        record = dict(epoch=self.epoch + 1, original_method_loss=float(original.detach()),
                      clqkd_beta=self.clqkd_settings['clqkd_beta'],
                      clqkd_raw=float(raw.detach()), clqkd_weighted=float(weighted.detach()),
                      total_loss=float(total.detach()), **counts)
        if supervised is not None:
            record['native_classification_raw'] = float(supervised)
        with (self.save_dir / 'clqkd_batches.jsonl').open('a', encoding='utf8') as f:
            f.write(json.dumps(record) + '\n')

    def optimizer_step(self):
        self.scaler.unscale_(self.optimizer)
        params = [p for group in self.optimizer.param_groups for p in group["params"]]
        torch.nn.utils.clip_grad_norm_(params, self.grad_clip, error_if_nonfinite=True)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        if self.ema:
            self.ema.update(self.model)

    def checkpoint_extras(self):
        auxiliary = self.distiller.state_dict() if isinstance(self.distiller, nn.Module) else {}
        return {"distillation": {
            "method": self.method, "settings": self.parameters_config, "clqkd": self.clqkd_active,
            "clqkd_settings": self.clqkd_settings, "objective_version": 3,
            "auxiliary": {k: v.detach().cpu().clone() for k, v in auxiliary.items()},
            "student": {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()},
            "optimizer": copy.deepcopy(self.optimizer.state_dict()),
            "scheduler": self.scheduler.state_dict(), "scaler": self.scaler.state_dict(),
            "epoch": self.epoch, "best_fitness": self.best_fitness,
            "ema": {k: v.detach().cpu().clone() for k, v in self.ema.ema.state_dict().items()},
            "updates": self.ema.updates,
            "teacher_buffers": {k: v.detach().cpu().clone() for k, v in self.teacher_model.named_buffers()},
            "resume_supported": True,
        }}

    def save_model(self):
        if self.distiller is None:
            return super().save_model()
        self.distiller.remove_hooks()
        try:
            return super().save_model()
        finally:
            self.distiller.install_hooks()
