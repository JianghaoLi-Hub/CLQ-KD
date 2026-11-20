# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
from ..base import KDTrainer
from .capture import *

class Trainer(KDTrainer):
    method = 'mgd'
    slot = 'mgd'

    def build_distiller(self):
        return MGDCapture(self.model, self.teacher_model, **self.parameters_config)

    def compute_loss_components(self, batch, masks=None):
        if 'teacher_scores' in batch:
            raise RuntimeError('Teacher response must not replace native classification loss')
        self.teacher_model.eval()
        self.mgd.enable_capture()
        try:
            (native, items) = self.model(batch)
            student = self.mgd.features
            if student is None:
                raise RuntimeError('Student Detect-input capture absent')
            teacher = self.mgd.teacher_head_inputs(batch['img'])
            self.capture_clqkd_features(teacher)
            if any((s.shape[0] != t.shape[0] or s.shape[2:] != t.shape[2:] for (s, t) in zip(student, teacher))):
                raise ValueError('Teacher/student feature geometry mismatch; interpolation forbidden')
            (raw, parts) = self.mgd(student, teacher, masks=masks, return_parts=True)
            diagnostics = {f'mgd_level_{index}': value.detach() for (index, value) in zip(self.mgd.indices, parts)}
            return (native.sum(), items, raw, diagnostics)
        finally:
            self.mgd.disable_capture()
