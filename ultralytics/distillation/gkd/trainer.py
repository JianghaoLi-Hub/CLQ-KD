# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
from ..base import KDTrainer
from .capture import *
import copy
from ultralytics.utils.loss import v8DetectionLoss

class Trainer(KDTrainer):
    method = 'gkd'
    slot = 'gkd'

    def build_distiller(self):
        self.teacher_model.args = copy.copy(self.args)
        self.teacher_criterion = v8DetectionLoss(self.teacher_model)
        return GKDCapture(self.model, self.teacher_model, **self.parameters_config)

    def teacher_features_and_gradients(self, batch):
        leaves = self.gkd.teacher_head_inputs(batch['img'])
        with torch.enable_grad():
            predictions = self.teacher_model.model[-1](leaves.copy())
            self.capture_clqkd_teacher(predictions)
            (teacher_loss, _) = self.teacher_criterion(predictions, batch)
            gradients = torch.autograd.grad(teacher_loss.sum(), leaves, create_graph=False)
        return ([x.detach() for x in leaves], [x.detach() for x in gradients])

    def compute_loss_components(self, batch):
        if 'teacher_scores' in batch:
            raise RuntimeError('Teacher classification targets must not replace native detection loss')
        self.teacher_model.eval()
        self.gkd.enable_capture()
        try:
            (native, items) = self.model(batch)
            student_features = self.gkd.features
            if student_features is None:
                raise RuntimeError('Student Detect feature capture absent')
            native_sum = native.sum()
            student_gradients = torch.autograd.grad(native_sum, student_features, retain_graph=True, create_graph=False)
            (teacher_features, teacher_gradients) = self.teacher_features_and_gradients(batch)
            for (s, teacher) in zip(student_features, teacher_features):
                if s.shape[0] != teacher.shape[0] or s.shape[2:] != teacher.shape[2:]:
                    raise ValueError('Teacher/student Detect input geometry mismatch')
            (kd, diagnostics) = gkd_loss(student_features, teacher_features, student_gradients, teacher_gradients, layer_weights=self.layer_weights)
            return (native_sum, items, kd, diagnostics)
        finally:
            self.gkd.disable_capture()
