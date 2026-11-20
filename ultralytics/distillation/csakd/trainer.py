# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
from ..base import KDTrainer
from .capture import *

class Trainer(KDTrainer):
    method = 'csakd'
    slot = 'csakd'

    def build_distiller(self):
        return CSAKDCapture(self.model, self.teacher_model, **self.parameters_config)

    def compute_training_loss(self, batch):
        active = getattr(self, 'clqkd_active', False)
        if active:
            self.prepare_clqkd()
            self.csakd.normal_target_callback = self.capture_clqkd_features
        try:
            native, items, agka, cross, _ = self.compute_loss_components(batch)
            extra = self.clqkd_extra(batch) if active else None
            if active:
                native, items = self.apply_clqkd_beta(native, items)
        finally:
            if active:
                self.csakd.normal_target_callback = None
                self.clear_clqkd()
        total = 2.0 * native
        if self.csakd_alpha != 0:
            total = total + self.csakd_alpha * agka
        if self.csakd_beta != 0:
            total = total + self.csakd_beta * cross
        if active:
            original = total
            total = total + extra[1] if self.clqkd_settings['clqkd_alpha'] else total
            self.record_clqkd(original, extra[0], extra[1], total, extra[2])
        if not torch.isfinite(torch.stack((native, agka, cross, total))).all():
            raise FloatingPointError('Nonfinite CSAKD loss')
        self.last_native_loss = native.detach()
        self.last_kd_loss = (self.csakd_alpha * agka + self.csakd_beta * cross).detach()
        self.last_total_loss = total.detach()
        return total, items

    def compute_loss_components(self, batch, use_checkpoint=False):
        if use_checkpoint is not False:
            raise ValueError('Peer CSAKD requires full-batch computation without activation checkpointing')
        if 'teacher_scores' in batch:
            raise RuntimeError('Teacher response must not replace native classification loss')
        self.teacher_model.eval()
        self.csakd.enable_capture()
        try:
            (native, items) = self.model(batch)
            student = self.csakd.features
            if student is None or any((feature is None for feature in student)):
                raise RuntimeError('Student 23-node capture absent or incomplete')
            self.csakd.disable_capture()
            agka, cross = self.csakd(batch['img'], student, use_checkpoint=use_checkpoint)
            return (native.sum(), items, agka, cross, {})
        finally:
            self.csakd.disable_capture()
