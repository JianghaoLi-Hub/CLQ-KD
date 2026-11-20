# CLQ-KD实验参数

[English](EXPERIMENT_SETTINGS.en.md) · [返回README](README.zh-CN.md)

下表列出本研究各数据集采用的CLQ-KD参数。参数选择建议及适用范围见[README](README.zh-CN.md#参数选择建议与适用范围)。

| 任务 | 数据集 | KD系数 $\alpha$ | GT分类系数 $\beta$ | 教师温度 $T_t$ | 学生温度 $T_s$ |
| --- | --- | ---: | ---: | ---: | ---: |
| 目标检测 | PASCAL VOC | 5 | 0 | 1.5 | 1 |
| 目标检测 | COCO | 1 | 5 | 1.5 | 1 |
| 目标检测 | African-Wildlife | 5 | 0 | 1 | 1 |
| 目标检测 | VisDrone | 1 | 5 | 1.5 | 1 |
| 目标检测 | HomeObjects-3K | 1 | 1 | 1.5 | 1 |
| 目标检测 | RUOD | 1 | 1 | 1.5 | 1 |
| 目标检测 | URPC2020 | 3 | 1 | 1.5 | 1 |
| 实例分割 | COCO | 1 | 5 | 1.5 | 1 |
| 实例分割 | Cityscapes | 5 | 1 | 1.5 | 1 |
| 实例分割 | CarDD | 5 | 0 | 1.5 | 1 |

四列参数分别对应 `clqkd_alpha`、`clqkd_beta`、`clqkd_teacher_temperature` 和 `clqkd_student_temperature`。其中β控制原生分类监督项的权重。

本表包含论文的目标检测与实例分割实验；当前开源库的训练入口面向YOLO11目标检测，实例分割行仅用于说明论文实验配置。上述设置不随数据集名称自动切换，使用时需显式设置相应参数。
