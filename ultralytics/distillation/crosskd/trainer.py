import torch
from ..base import KDTrainer
from .core import CrossKD


class Trainer(KDTrainer):
    method = "crosskd"
    slot = "crosskd"

    def build_distiller(self):
        return CrossKD(self.model, self.teacher_model, **self.parameters_config)

    def compute_loss_components(self, batch):
        self.teacher_model.eval()
        self.crosskd.enable_capture()
        try:
            native, items = self.model(batch)
            with torch.no_grad():
                predictions = self.teacher_model(batch["img"])
                self.capture_clqkd_teacher(predictions)
                raw = predictions[1] if isinstance(predictions, tuple) else predictions
            kd = self.crosskd(self.crosskd.cross_forward(), raw, self.model.criterion.last_fg_mask)
            return native.sum(), items, kd, {}
        finally:
            self.crosskd.disable_capture()
