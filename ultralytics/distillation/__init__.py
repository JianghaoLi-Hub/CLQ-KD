"""Lazy method selection; importing YOLO never loads a teacher."""
from importlib import import_module
from pathlib import Path
import yaml

METHODS = ("baseline", "clqkd", "crosskd", "csakd", "cwd", "fgd", "gkd", "mgd", "ost", "pkd", "reviewkd")


def method_config(method):
    if method not in METHODS:
        raise ValueError(f"Unknown distillation method {method!r}. Choose from: {', '.join(METHODS)}")
    if method == "baseline":
        return {}
    path = Path(__file__).resolve().parents[2] / "configs/distillation.yaml"
    with path.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)[method]


def trainer_for(method):
    method_config(method)
    if method == "baseline":
        from .base import BaselineTrainer
        return BaselineTrainer
    return import_module(f"ultralytics.distillation.{method}.trainer").Trainer
