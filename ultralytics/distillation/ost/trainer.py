import torch
from ..base import KDTrainer
from .capture import OSTCapture


class Trainer(KDTrainer):
    method = "ost"
    slot = "ost"

    def build_distiller(self):
        params = {k: v for k, v in self.parameters_config.items() if k != "kd_weight"}
        return OSTCapture(self.model, self.teacher_model, **params)

    def compute_loss_components(self, batch):
        self.teacher_model.eval()
        self.ost.enable_capture()
        try:
            native, items = self.model(batch)
            with torch.no_grad():
                self.capture_clqkd_teacher(self.teacher_model(batch["img"]))
            return native.sum(), items, self.ost(), {}
        finally:
            self.ost.disable_capture()
