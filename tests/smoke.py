"""Real COCO8 integration tests; not an accuracy benchmark.

Run from the repository root: python -m tests.smoke [method ...]
Outputs (including weights and the JSON report) are ignored by Git.
"""
import csv
import gc
import hashlib
import json
import math
import os
import sys
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path(os.environ.get("KD_TEST_OUTPUT", str(Path(tempfile.gettempdir()) / "clqkd-smoke")))
os.environ.setdefault("YOLO_CONFIG_DIR", str(OUTPUT / "settings"))

import torch
from ultralytics import YOLO, settings
from ultralytics.distillation import METHODS

def fingerprint(tensors):
    digest = hashlib.sha256()
    for name, tensor in tensors:
        digest.update(name.encode())
        digest.update(tensor.detach().float().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def hook_count(model):
    return sum(len(m._forward_hooks) + len(m._forward_pre_hooks) for m in model.modules())


def run(case):
    combination = case.endswith('+clqkd')
    method = case.split('+')[0]
    record = {"method": method, "clqkd": combination, "optimizer_steps": 0, "teacher_head_calls": 0, "student_calls": 0, "tal_calls": 0}
    model = YOLO(str(ROOT / "configs/yolo11n.yaml"))

    def before(trainer):
        record["student_before"] = fingerprint(trainer.model.named_parameters())
        teacher = trainer.teacher_model
        record["has_teacher"] = teacher is not None
        if teacher is not None:
            record["teacher_before"] = fingerprint(teacher.named_parameters())
            record["bn_before"] = fingerprint((n, b) for n, b in teacher.named_buffers() if "running_" in n)
            assert not any(p.requires_grad for p in teacher.parameters())
        aux = getattr(trainer, "distiller", None)
        if isinstance(aux, torch.nn.Module):
            params = list(aux.named_parameters())
            record["auxiliary_parameter_count"] = sum(p.numel() for _, p in params)
            record["auxiliary_before"] = fingerprint(params)
            registered = {id(p) for group in trainer.optimizer.param_groups for p in group["params"]}
            assert all(id(p) in registered for _, p in params if p.requires_grad)
            assert not any(id(p) in registered for p in teacher.parameters())
        def count_student(*args): record['student_calls'] += 1
        def count_tal(*args): record['tal_calls'] += 1
        if not hasattr(trainer.model, 'criterion'): trainer.model.criterion = trainer.model.init_criterion()
        # Count only train forwards; the validator uses EMA.
        record['_hooks'] = [trainer.model.register_forward_hook(count_student), trainer.model.criterion.assigner.register_forward_hook(count_tal)]
        if teacher is not None:
            def count_teacher(*args): record['teacher_head_calls'] += 1
            record['_hooks'].append(teacher.model[-1].register_forward_hook(count_teacher))
        original_step = trainer.optimizer_step

        def step():
            original_step()
            record["optimizer_steps"] += 1
        trainer.optimizer_step = step

    def remove_test_hooks(trainer):
        for h in record.pop('_hooks', []): h.remove()

    def epoch_end(trainer):
        assert torch.isfinite(trainer.loss).all()
        record["student_after"] = fingerprint(trainer.model.named_parameters())
        if trainer.teacher_model is not None:
            record["teacher_after"] = fingerprint(trainer.teacher_model.named_parameters())
            record["bn_after"] = fingerprint((n, b) for n, b in trainer.teacher_model.named_buffers() if "running_" in n)
        aux = getattr(trainer, "distiller", None)
        if isinstance(aux, torch.nn.Module):
            record["auxiliary_after"] = fingerprint(aux.named_parameters())

    model.add_callback("on_train_start", before)
    model.add_callback("on_train_epoch_end", epoch_end)
    model.add_callback("on_train_epoch_end", remove_test_hooks)
    teacher_root = Path(os.environ.get('KD_TEST_TEACHERS', str(ROOT / 'weights')))
    teacher_kw = {} if method == 'baseline' else dict(teacher_model=str(teacher_root / ('yolo11n.pt' if method == 'csakd' else 'yolo11x.pt')))
    model.train(method=method, clqkd=combination, data=str(OUTPUT / 'coco8.yaml'), epochs=3, imgsz=160, batch=2,
                workers=0, amp=False, nbs=2, warmup_epochs=0, device=0,
                project=str(OUTPUT / "runs"), name=case, exist_ok=False,
                plots=False, save=True, val=True, seed=0, **teacher_kw)
    trainer = model.trainer
    assert record["optimizer_steps"] >= 6
    assert record["student_before"] != record["student_after"]
    assert record["has_teacher"] == (method != "baseline")
    if method != "baseline":
        assert record["teacher_before"] == record["teacher_after"]
        assert (record["bn_before"] != record["bn_after"]) == (method == "clqkd")
    if record.get("auxiliary_parameter_count", 0):
        assert record["auxiliary_before"] != record["auxiliary_after"]
    assert record['student_calls'] == record['tal_calls'] == 2
    expected_teacher_calls = 0 if method == 'baseline' else (2 if combination or method in ('clqkd','fgd','ost','crosskd','gkd') else 0)
    assert record['teacher_head_calls'] == expected_teacher_calls, record
    assert hook_count(trainer.model) == 0
    if trainer.teacher_model is not None:
        assert hook_count(trainer.teacher_model) == 0
    rows = list(csv.DictReader(trainer.csv.open(encoding="utf-8")))
    assert [int(r["epoch"].strip()) for r in rows] == [1, 2, 3]
    assert all(math.isfinite(float(v)) for row in rows for v in row.values())
    checkpoint = torch.load(trainer.last, map_location="cpu", weights_only=False)
    assert checkpoint.get("distillation", {}).get("method") == (None if method == "baseline" else method)
    assert hook_count(checkpoint["model"]) == 0
    reloaded = YOLO(str(trainer.last))
    metrics = reloaded.val(data=str(OUTPUT / "coco8.yaml"), imgsz=160, batch=2, workers=0, device=0, plots=False)
    assert all(math.isfinite(float(v)) for v in metrics.results_dict.values())
    image = next(Path(trainer.data["val"]).glob("*.jpg"))
    prediction = reloaded.predict(source=str(image), imgsz=160, device=0, save=False)
    assert len(prediction) == 1 and torch.isfinite(prediction[0].boxes.data).all()
    record.update(passed=True, epochs=len(rows), checkpoint_reload=True, validation=True, prediction=True,
                  save_dir=str(trainer.save_dir))
    del model, reloaded, trainer, checkpoint, prediction
    gc.collect()
    torch.cuda.empty_cache()
    return record


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    dataset = OUTPUT / 'datasets/coco8'
    if not dataset.exists(): shutil.copytree(ROOT / 'datasets/coco8', dataset)
    import yaml
    cfg = yaml.safe_load((ROOT / 'ultralytics/cfg/datasets/coco8.yaml').read_text(encoding='utf-8'))
    cfg['path'] = str(dataset); cfg.pop('download', None)
    (OUTPUT / 'coco8.yaml').write_text(yaml.safe_dump(cfg))
    settings.update({"datasets_dir": str(OUTPUT / 'datasets'), "weights_dir": str(OUTPUT / 'teachers'),
                     "runs_dir": str(OUTPUT / 'runs'), "sync": False})
    output = OUTPUT
    for method in sys.argv[1:] or [*METHODS, *(m+"+clqkd" for m in METHODS if m not in ("baseline", "clqkd"))]:
        try:
            result = run(method)
        except Exception as error:
            (output / f"{method}-FAILED.json").write_text(json.dumps({"method": method, "error": repr(error)}, indent=2), encoding="utf-8")
            raise
        (output / f"{method}-passed.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"SMOKE PASSED: {method}", flush=True)


if __name__ == "__main__":
    main()
