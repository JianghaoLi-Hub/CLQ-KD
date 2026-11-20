"""Resume routing and exact restoration of saved KD state (CPU)."""
import copy
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault('YOLO_CONFIG_DIR', str(Path(tempfile.gettempdir()) / 'clqkd-tests/settings'))
import torch
from torch import nn
from ultralytics import YOLO
from ultralytics.distillation import trainer_for
from ultralytics.distillation.base import KDTrainer


class ResumeTests(unittest.TestCase):
    def test_resume_routes_saved_combination_and_parameters(self):
        model = YOLO('configs/yolo11n.yaml')
        model.ckpt = {'train_args': {'method': 'cwd', 'clqkd': True}, 'distillation': {
            'method': 'cwd', 'clqkd': True, 'clqkd_settings': {'clqkd_alpha': 3.0}}}
        with patch('ultralytics.engine.model.Model.train') as call:
            model.train(resume=True)
            self.assertIs(call.call_args.kwargs['trainer'], trainer_for('cwd'))
            self.assertTrue(call.call_args.kwargs['clqkd'])
            self.assertEqual(call.call_args.kwargs['clqkd_alpha'], 3.)
            self.assertEqual(call.call_args.kwargs['clqkd_beta'], 1.)
        model.ckpt['distillation']['clqkd_settings']['clqkd_beta']=3.
        with patch('ultralytics.engine.model.Model.train') as call:
            model.train(resume=True)
            self.assertEqual(call.call_args.kwargs['clqkd_beta'],3.)
        with self.assertRaisesRegex(ValueError, 'cannot change'):
            model.train(resume=True, method='clqkd')
        model.ckpt['distillation']['method'] = 'clqkd'
        with patch('ultralytics.engine.model.Model.train') as call:
            model.train(resume=True)
            self.assertIs(call.call_args.kwargs['trainer'], trainer_for('clqkd'))
            self.assertFalse(call.call_args.kwargs['clqkd'])

    def test_resume_flag_allowed_precision_guard_unchanged(self):
        def init(obj, *args, **kwargs):
            obj.args = SimpleNamespace(resume=True, amp=False, teacher_amp=False)
        with patch('ultralytics.models.yolo.detect.DetectionTrainer.__init__', init):
            trainer_for('cwd')()
        def amp_init(obj, *args, **kwargs):
            obj.args = SimpleNamespace(resume=True, amp=True, teacher_amp=False)
        with patch('ultralytics.models.yolo.detect.DetectionTrainer.__init__', amp_init):
            with self.assertRaisesRegex(ValueError, 'FP32'):
                trainer_for('cwd')()

    @staticmethod
    def trainer():
        obj = object.__new__(KDTrainer)
        obj.method, obj.slot, obj.resume = 'cwd', 'aux', True
        obj.model, obj.aux = nn.Linear(2, 2), nn.Linear(2, 2)
        obj.teacher_model = nn.BatchNorm1d(2).requires_grad_(False)
        obj.ema = SimpleNamespace(ema=copy.deepcopy(obj.model), updates=7)
        obj.optimizer = torch.optim.SGD([*obj.model.parameters(), *obj.aux.parameters()], lr=.1, momentum=.9)
        obj.scheduler = torch.optim.lr_scheduler.LambdaLR(obj.optimizer, lambda e: 1 - .1 * e)
        obj.scaler = torch.cuda.amp.GradScaler(enabled=False)
        obj.args = SimpleNamespace(close_mosaic=0, epochs=3, model='test.pt')
        obj.epochs, obj.epoch, obj.start_epoch = 3, 0, 0
        obj.best_fitness = .42
        obj.parameters_config = {'kd_weight': 5.}
        obj.clqkd_active, obj.clqkd_settings = True, {'clqkd_alpha': 3.}
        return obj

    def test_complete_state_and_next_update_survive_serialization(self):
        old = self.trainer()
        x = torch.tensor([[1., 2.], [3., 5.]])
        old.aux(old.model(x)).square().sum().backward()
        old.optimizer.step()
        old.optimizer.zero_grad()
        old.scheduler.step()
        old.teacher_model(x)
        ckpt = dict(old.checkpoint_extras(), epoch=0, best_fitness=.42,
                    ema=copy.deepcopy(old.ema.ema), updates=7)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'last.pt'
            torch.save(ckpt, path)
            ckpt = torch.load(path, weights_only=False)
        new = self.trainer()
        new.resume_training(ckpt)
        self.assertEqual(new.start_epoch, 1)
        self.assertEqual(new.ema.updates, 7)
        self.assertEqual(new.scheduler.state_dict(), old.scheduler.state_dict())
        for attr in ('model', 'aux', 'teacher_model'):
            for name, value in getattr(old, attr).state_dict().items():
                torch.testing.assert_close(value, getattr(new, attr).state_dict()[name], rtol=0, atol=0)
        for trainer in (old, new):
            trainer.aux(trainer.model(x)).square().sum().backward()
            trainer.optimizer.step()
        for attr in ('model', 'aux'):
            for a, b in zip(getattr(old, attr).parameters(), getattr(new, attr).parameters()):
                torch.testing.assert_close(a, b, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
