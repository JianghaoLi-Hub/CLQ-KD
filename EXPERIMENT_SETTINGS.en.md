# CLQ-KD Experimental Settings

[中文](EXPERIMENT_SETTINGS.md) · [Back to README](README.md)

The table lists the CLQ-KD settings used for each dataset in this study. See the [README](README.md#parameter-selection-and-applicability) for parameter-selection guidance and applicability.

| Task | Dataset | KD coefficient $\alpha$ | GT classification coefficient $\beta$ | Teacher temperature $T_t$ | Student temperature $T_s$ |
| --- | --- | ---: | ---: | ---: | ---: |
| Object detection | PASCAL VOC | 5 | 0 | 1.5 | 1 |
| Object detection | COCO | 1 | 5 | 1.5 | 1 |
| Object detection | African-Wildlife | 5 | 0 | 1 | 1 |
| Object detection | VisDrone | 1 | 5 | 1.5 | 1 |
| Object detection | HomeObjects-3K | 1 | 1 | 1.5 | 1 |
| Object detection | RUOD | 1 | 1 | 1.5 | 1 |
| Object detection | URPC2020 | 3 | 1 | 1.5 | 1 |
| Instance segmentation | COCO | 1 | 5 | 1.5 | 1 |
| Instance segmentation | Cityscapes | 5 | 1 | 1.5 | 1 |
| Instance segmentation | CarDD | 5 | 0 | 1.5 | 1 |

The four parameter columns correspond to `clqkd_alpha`, `clqkd_beta`, `clqkd_teacher_temperature`, and `clqkd_student_temperature`, respectively. Here, β weights the native classification supervision term.

This table covers the object-detection and instance-segmentation experiments in the paper. The current repository implements training entries for YOLO11 object detection; the instance-segmentation rows document the paper settings only. These settings are not selected automatically by dataset name and must be specified explicitly.
