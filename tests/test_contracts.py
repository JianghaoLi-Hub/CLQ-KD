"""Unit tests for method selection, configuration and model compatibility."""
import os
import tempfile
import copy
from pathlib import Path
import unittest
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(tempfile.gettempdir()) / "clqkd-tests/settings"))

import torch
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.distillation import METHODS, method_config, trainer_for
from ultralytics.distillation.base import validate_teacher, validate_yolo11
from ultralytics.distillation.cwd.loss import CWDLoss
from ultralytics.distillation.mgd.loss import MGDLoss
from ultralytics.distillation.pkd.loss import PKDLoss
from ultralytics.distillation.csakd.capture import validate_peer_architecture
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.checks import check_yaml

torch.set_num_threads(2)


class Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.yolo = YOLO(str(ROOT / "configs/yolo11n.yaml"))

    def test_registry_and_switching(self):
        self.assertEqual(len(METHODS), 11)
        with patch("ultralytics.engine.model.Model.train") as train:
            for name in METHODS:
                self.yolo.train(method=name)
                args = train.call_args.kwargs
                self.assertIs(args["trainer"], trainer_for(name))
                self.assertFalse(args["amp"])
                self.assertEqual(args["teacher_model"], None if name == "baseline" else method_config(name)["teacher_model"])
            self.yolo.train(method="baseline", teacher_model="not-loaded.pt")
            self.assertIsNone(train.call_args.kwargs["teacher_model"])
            self.yolo.train(method="pkd", teacher_model="custom.pt")
            self.assertEqual(train.call_args.kwargs["teacher_model"], "custom.pt")

    def test_invalid_method(self):
        with self.assertRaisesRegex(ValueError, "Unknown distillation method"):
            self.yolo.train(method="typo")

    def test_packaged_coco8_yaml(self):
        self.assertEqual(Path(check_yaml("coco8.yaml")).resolve(),
                         ROOT / "ultralytics/cfg/datasets/coco8.yaml")

    def test_unicode_checkpoint_path(self):
        with tempfile.TemporaryDirectory(prefix="蒸馏-") as directory:
            path = Path(directory) / "权重.pt"
            for keyword in (False, True):
                state = {"tensor": torch.arange(5)}
                if keyword:
                    torch.save(obj=state, f=path)
                else:
                    torch.save(state, path)
                loaded = torch.load(path, weights_only=False)
                torch.testing.assert_close(loaded["tensor"], state["tensor"])

    def test_single_device(self):
        with self.assertRaisesRegex(ValueError, "single-device"):
            self.yolo.train(method="cwd", device="0,1")

    def test_model_and_classes(self):
        student = self.yolo.model
        validate_yolo11(student)
        validate_teacher(student, student, student.names)
        with self.assertRaisesRegex(ValueError, "class count or class order"):
            validate_teacher(student, student, {0: "wrong"})
        head = student.model[-1]
        old = head.end2end
        try:
            head.end2end = True
            with self.assertRaisesRegex(ValueError, "YOLO11"):
                validate_yolo11(student)
        finally:
            head.end2end = old

    def test_peer_architecture_rejection(self):
        other = YOLO("yolo11s.yaml").model.float()
        with self.assertRaisesRegex(ValueError, "identical full internal"):
            validate_peer_architecture(self.yolo.model.float(), other)

    def test_peer_legacy_false_root_flag(self):
        student = self.yolo.model.float()
        teacher = copy.deepcopy(student)
        del teacher.end2end
        validate_peer_architecture(student, teacher)
        teacher.end2end = True
        with self.assertRaisesRegex(ValueError, "identical full internal"):
            validate_peer_architecture(student, teacher)

    def test_pkd_sample_standard_deviation(self):
        torch.manual_seed(1)
        s = torch.randn(2, 3, 4, 4, requires_grad=True)
        t = torch.randn_like(s)
        norm = lambda x: (x - x.mean((0, 2, 3), keepdim=True)) / (x.std((0, 2, 3), keepdim=True) + 1e-6)
        expected = F.mse_loss(norm(s), norm(t)) / 2
        torch.testing.assert_close(PKDLoss()(s, t), expected)

    def test_cwd_kl_and_batch_channel_reduction(self):
        torch.manual_seed(2)
        s = torch.randn(2, 3, 4, 4, requires_grad=True)
        t = torch.randn_like(s)
        p = t.flatten(2).softmax(-1)
        expected = (p * (p.log() - s.flatten(2).log_softmax(-1))).sum() / 6
        torch.testing.assert_close(CWDLoss()(s, t), expected)

    def test_mgd_sum_over_pixels(self):
        s = torch.ones(2, 3, 4, 4, requires_grad=True)
        t = torch.zeros_like(s, requires_grad=True)
        loss = MGDLoss()(s, t)
        self.assertEqual(loss.item(), 48.)
        loss.backward()
        self.assertIsNone(t.grad)
        torch.testing.assert_close(s.grad, torch.ones_like(s))

    def test_removed_acam_is_unavailable(self):
        self.assertNotIn("acamkd", METHODS)
        for entry in (method_config, trainer_for):
            with self.assertRaisesRegex(ValueError, "Unknown distillation method"):
                entry("acamkd")
        with self.assertRaisesRegex(ValueError, "Unknown distillation method"):
            self.yolo.train(method="acamkd")
        self.assertFalse((ROOT / "ultralytics/distillation/acamkd").exists())

    def test_native_classification_and_reject_unfiltered_targets(self):
        model = self.yolo.model
        model.args = get_cfg()
        criterion = v8DetectionLoss(model)
        logits = torch.randn(2, 10, 80, requires_grad=True)
        targets = torch.zeros_like(logits)
        native = criterion.classification_loss(logits, targets, 1.)
        torch.testing.assert_close(native, F.binary_cross_entropy_with_logits(logits, targets, reduction="sum"))
        with self.assertRaises(ValueError):
            criterion.classification_loss(logits, targets, 1., torch.rand_like(logits))


if __name__ == "__main__":
    unittest.main()
