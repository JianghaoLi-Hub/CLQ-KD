"""CPU-only selected-default and one-time scaling regression tests."""
import os
import tempfile
from pathlib import Path
os.environ.setdefault('YOLO_CONFIG_DIR', str(Path(tempfile.gettempdir()) / 'clqkd-tests/settings'))
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import torch
from ultralytics.distillation import method_config, trainer_for
from ultralytics.distillation.base import KDTrainer
from ultralytics.distillation.gkd.core import gkd_loss
from ultralytics.distillation.csakd.capture import CSAKD_ALPHA
from ultralytics.distillation.ost.capture import OST_WEIGHT


class SelectedDefaults(unittest.TestCase):
    expected = {
        'crosskd': dict(beta=1., temperature=1., cls_weight=1., reg_weight=4.),
        'fgd': dict(temp=.5, alpha_fgd=.001, beta_fgd=.0005, gamma_fgd=.0005, lambda_fgd=.000005),
        'gkd': dict(kd_weight=.125, layer_weights=[1., 1., 1.]),
        'ost': dict(kd_weight=5., qk_dim=128, temperature=1., threshold=.5),
        'pkd': dict(kd_weight=10.), 'reviewkd': dict(kd_weight=2., mid_channels=256),
        'cwd': dict(kd_weight=5., tau=1.), 'mgd': dict(mgd_alpha=.00002, mask_ratio=.65),
        'csakd': dict(csakd_alpha=300., csakd_beta=10., kd_batch_mode='full'),
    }

    def test_nine_configurations_and_explicit_config_priority(self):
        for name, expected in self.expected.items():
            with self.subTest(method=name):
                conf = method_config(name)
                self.assertEqual({k: conf[k] for k in expected}, expected)
                # Supported override path: method config before trainer construction.
                key = next(iter(expected))
                changed = dict(conf, **{key: 7. if key != 'kd_batch_mode' else 'full'})
                def init_base(obj, *args, **kwargs):
                    obj.args = SimpleNamespace(resume=False, amp=False, teacher_amp=False)
                with patch('ultralytics.distillation.base.method_config', return_value=changed), patch('ultralytics.models.yolo.detect.DetectionTrainer.__init__', init_base):
                    trainer = trainer_for(name)()
                    self.assertEqual(getattr(trainer, key), changed[key])
                    self.assertEqual(trainer.parameters_config[key], changed[key])
        self.assertEqual(CSAKD_ALPHA, 300.)
        self.assertEqual(OST_WEIGHT, 5.)
        self.assertFalse(torch.cuda.is_initialized())

    def test_gkd_layer_weights_enter_core_and_outer_scale_once(self):
        torch.manual_seed(17)
        s = [torch.rand(2, 3, 4, 4, requires_grad=True) for _ in range(3)]
        t = [torch.rand_like(x) for x in s]
        gs = [torch.ones_like(x) for x in s]
        gt = [torch.ones_like(x) for x in t]
        equal, _ = gkd_loss(s, t, gs, gt, layer_weights=method_config('gkd')['layer_weights'])
        doubled, _ = gkd_loss(s, t, gs, gt, layer_weights=[2, 2, 2])
        torch.testing.assert_close(doubled, 2 * equal)
        for name in ['gkd', 'ost']:
            weight = method_config(name)['kd_weight']
            obj = SimpleNamespace(method=name, kd_weight=weight, compute_loss_components=lambda b: (torch.tensor(3.), None, equal, {}))
            total, _ = KDTrainer.compute_training_loss(obj, {'img': torch.empty(2, 3, 4, 4)})
            torch.testing.assert_close(total, 3 + 2 * weight * equal)
            obj.kd_weight = 7.
            changed, _ = KDTrainer.compute_training_loss(obj, {'img': torch.empty(2, 3, 4, 4)})
            torch.testing.assert_close(changed, 3 + 2 * 7 * equal)

    def test_csakd_loss_no_extra_batch_or_duplicate_weight(self):
        raw = torch.tensor(2., requires_grad=True)
        for alpha in [300., 7.]:
            obj = SimpleNamespace(csakd_alpha=alpha, csakd_beta=10., compute_loss_components=lambda b: (torch.tensor(3.), None, raw, torch.tensor(4.), {}))
            total, _ = trainer_for('csakd').compute_training_loss(obj, {'img': torch.empty(40, 3, 1, 1)})
            torch.testing.assert_close(total, torch.tensor(6. + alpha * 2 + 40))
            self.assertEqual(torch.autograd.grad(total, raw)[0].item(), alpha)

if __name__ == '__main__':
    unittest.main()
