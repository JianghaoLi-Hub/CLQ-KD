import torch
from ..base import KDTrainer
from .core import FGD


class Trainer(KDTrainer):
    method = "fgd"
    slot = "fgd"
    grad_clip = 35.0

    def build_distiller(self):
        return FGD(self.model, self.teacher_model, **self.parameters_config)

    def compute_loss_components(self, batch):
        self.teacher_model.eval()
        self.fgd.enable_capture()
        try:
            native, items = self.model(batch)
            with torch.no_grad():
                self.capture_clqkd_teacher(self.teacher_model(batch["img"]))
            return native.sum(), items, self.fgd(batch), {}
        finally:
            self.fgd.disable_capture()
