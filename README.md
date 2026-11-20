# YOLO11 Knowledge Distillation

[中文](README.zh-CN.md)

Based on Ultralytics 8.3.178, this repository supports native baseline training, CLQ-KD, and CWD, FGD, PKD, MGD, CrossKD, GKD, OST, ReviewKD and CSAKD, either individually or combined with CLQ-KD.

## Installation

Environment: Python 3.8.20, PyTorch 2.0.1, torchvision 0.15.2 and CUDA 11.7.

```bash
python -m pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu117
python -m pip install -r requirements.txt
```

The repository includes the eight COCO8 images and labels, but no teacher weights. Prepare the teacher weights before running distillation examples. For COCO8 smoke tests, download the official COCO-pretrained `yolo11x.pt`; the same-architecture CSAKD test uses `yolo11n.pt`. Place the files in `weights/`. See [weights/README.md](weights/README.md) for official sources and checksums.

After installation and preparation of the required teacher weights, run a small GPU smoke test; outputs are saved in the system temporary directory:

```bash
python -m tests.smoke baseline clqkd cwd+clqkd
```

To test all entries, run `python -m tests.smoke` without method arguments. COCO8 is for functional testing; use the corresponding full dataset and teacher checkpoint for the paper experiments.

## Datasets

See [dataset descriptions and splits](DATASETS.md) (Chinese) for dataset sources and their use in the experiments. The VOC experiments follow the Ultralytics image splits and retain objects marked as difficult.

## Training

Set the method, teacher weights, dataset path and training parameters in `train.py`, then run:

```bash
python train.py
```

Standalone CLQ-KD with the bundled COCO8 data and an official teacher downloaded separately. Save this script in the repository root:

```python
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent

if __name__ == "__main__":
    model = YOLO(str(ROOT / "configs/yolo11n.yaml"))
    model.train(
        method="clqkd",
        teacher_model=str(ROOT / "weights/yolo11x.pt"),
        clqkd_alpha=5.0,
        clqkd_beta=0.0,
        clqkd_teacher_temperature=1.5,
        clqkd_student_temperature=1.0,
        clqkd_prob_threshold=0.001,
        clqkd_iou_threshold=0.3,
        data=str(ROOT / "datasets/coco8/data.yaml"),
        epochs=100,
        imgsz=640,
        batch=80,
        seed=0,
        amp=False,
    )
```

Select the training mode with the following arguments. Replace `cwd` with another comparison method as needed:

| Training mode | Arguments |
| --- | --- |
| Native baseline | `method="baseline"` |
| Comparison KD method alone | `method="cwd", clqkd=False` |
| Comparison KD method + CLQ-KD | `method="cwd", clqkd=True, clqkd_beta=1.0` |
| Standalone CLQ-KD | `method="clqkd"` |

For baseline training or a comparison method alone, remove all `clqkd_` arguments from the example. Combined training also supports `clqkd_beta`: 0 disables native classification supervision, 1 retains its original strength, and other values scale it proportionally. Box, DFL and the original distillation terms are unchanged. Use `clqkd=True` only for combined training.

### CLQ-KD Parameters

| Parameter | Description | Default |
| --- | --- | ---: |
| `clqkd_alpha` | Classification distillation coefficient | 5.0 |
| `clqkd_beta` | Native classification supervision coefficient | Standalone: 0.0; combined: 1.0 |
| `clqkd_teacher_temperature` | Teacher temperature | 1.5 |
| `clqkd_student_temperature` | Student training temperature | 1.0 |
| `clqkd_prob_threshold` | Unscaled teacher probability threshold for the matched GT class | 0.001 |
| `clqkd_iou_threshold` | IoU threshold between the teacher box and matched GT box | 0.3 |

Explicit CLQ-KD arguments in `train.py` override `configs/distillation.yaml`. The default combined-training beta is set in the `clqkd_combination` section; other KD parameters are configured in their respective method sections.

The teacher's class count and order must match the dataset. CSAKD also requires identical teacher and student architectures; with the default YOLO11n student, set the teacher path to `weights/yolo11n.pt`.

### Parameter Selection and Applicability

See the [CLQ-KD experimental settings table](EXPERIMENT_SETTINGS.en.md) for the KD coefficient α, GT classification coefficient β, and teacher/student temperatures used for each dataset. The table covers the object-detection and instance-segmentation experiments in the paper. The current repository implements training entries for YOLO11 object detection; the instance-segmentation rows document the paper settings only.

Based on our experimental experience, CLQ-KD may use teacher-derived classification supervision alone (*β*=0) when the teacher provides sufficiently reliable responses. Greater caution is needed on small datasets, where teacher generalization should be assessed before relying entirely on teacher-derived classification targets. When teacher reliability is limited, retaining native classification supervision through a positive *β*, such as *β*=1, may be preferable. A smaller *α* may also be considered to limit the influence of unreliable teacher responses or an overly dominant distillation term.

Additionally, the KD loss is normalized by the number of prediction anchors but not by the number of class channels. A larger number of classes can therefore increase the aggregate KD loss, potentially requiring a smaller *α* to maintain an appropriate balance with the other objectives. These recommendations summarize empirical experience rather than establish universal selection rules. The appropriate settings remain dataset dependent and should be validated for the specific teacher–student configuration.

## Comparison Methods

See the [method configuration table](METHOD_COMPARISON.md) (Chinese) for distilled feature levels, parameter settings, correspondence with the original papers and official/public code, and implementation adaptations. The table links to the reference GitHub revisions.

## Resuming Training

```python
from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("runs/detect/train/weights/last.pt")
    model.train(resume=True)
```

The training method, configuration and state are restored from the checkpoint. The teacher continues to evaluate each current batch online; retain the original teacher weights.

## Validation and Prediction

Set the student weights and input paths in `val.py` and `predict.py`, then run. Neither requires a teacher:

```bash
python val.py
python predict.py
```

## References

Please cite the original papers for the methods you use. BibTeX entries are provided in [references.bib](references.bib).

| Method | Original paper | Publication |
| --- | --- | --- |
| CWD | [Channel-wise Knowledge Distillation for Dense Prediction](https://doi.org/10.1109/ICCV48922.2021.00526) | ICCV 2021 |
| FGD | [Focal and Global Knowledge Distillation for Detectors](https://doi.org/10.1109/CVPR52688.2022.00460) | CVPR 2022 |
| PKD | [PKD: General Distillation Framework for Object Detectors via Pearson Correlation Coefficient](https://papers.neurips.cc/paper_files/paper/2022/hash/631ad9ae3174bf4d6c0f6fdca77335a4-Abstract-Conference.html) | NeurIPS 2022; [DOI](https://doi.org/10.52202/068431-1120) |
| MGD | [Masked Generative Distillation](https://doi.org/10.1007/978-3-031-20083-0_4) | ECCV 2022 |
| CrossKD | [CrossKD: Cross-Head Knowledge Distillation for Object Detection](https://doi.org/10.1109/CVPR52733.2024.01563) | CVPR 2024 |
| GKD | [Gradient-Guided Knowledge Distillation for Object Detectors](https://doi.org/10.1109/WACV57701.2024.00049) | WACV 2024 |
| OST | [Guided Hybrid Quantization for Object Detection in Remote Sensing Imagery via One-to-One Self-Teaching](https://doi.org/10.1109/TGRS.2023.3293147) | IEEE TGRS 2023 |
| ReviewKD | [Distilling Knowledge via Knowledge Review](https://doi.org/10.1109/CVPR46437.2021.00497) | CVPR 2021 |
| CSAKD | [Knowledge Distillation via Cross Supervising with Attention for Remote Sensing Object Detection](https://bmvc2025.bmva.org/proceedings/398/) | BMVC 2025 |

OST refers to the One-to-One Self-Teaching distillation component of GHOST.

## License

The Ultralytics framework uses [AGPL-3.0](LICENSE). Third-party components retain their own licenses; CrossKD upstream uses CC BY-NC 4.0. See [THIRD_PARTY.md](THIRD_PARTY.md) for sources, licenses and third-party terms.
