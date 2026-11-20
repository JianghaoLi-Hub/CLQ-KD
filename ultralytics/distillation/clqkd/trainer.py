"""Standalone CLQ-KD; frozen train-mode teacher, native validation loss."""
from ..base import KDTrainer
import torch


class Trainer(KDTrainer):
    method = 'clqkd'

    def compute_training_loss(self, batch):
        criterion = self.prepare_clqkd()
        try:
            with torch.no_grad():
                self.capture_clqkd_teacher(self.teacher_model(batch['img']))
            criterion.clqkd_teacher = self._clqkd_teacher
            criterion.clqkd_settings = self.clqkd_settings
            loss, items = self.model(batch)
            total = loss.sum()
            if not torch.isfinite(total):
                raise FloatingPointError('Nonfinite CLQ-KD total loss')
            raw, counts, supervised = criterion.clqkd_diagnostics
            weighted = raw * len(batch['img']) * self.args.cls * self.clqkd_settings['clqkd_alpha']
            self.record_clqkd(total - weighted, raw, weighted, total, counts, supervised)
            return total, items
        finally:
            self.clear_clqkd()
