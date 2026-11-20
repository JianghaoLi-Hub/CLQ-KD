# YOLO11 知识蒸馏

[English](README.md)

基于 Ultralytics 8.3.178，支持普通基线、CLQ-KD，以及 CWD、FGD、PKD、MGD、CrossKD、GKD、OST、ReviewKD、CSAKD 的独立训练和与 CLQ-KD 的组合训练。

## 安装

环境：Python 3.8.20、PyTorch 2.0.1、torchvision 0.15.2、CUDA 11.7。

```bash
python -m pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu117
python -m pip install -r requirements.txt
```

仓库附带COCO8的8张图片和标签，不包含教师权重。运行蒸馏示例前，请自行准备教师权重；COCO8短测可使用官方COCO预训练 `yolo11x.pt`，CSAKD同架构师生短测使用 `yolo11n.pt`，分别放入 `weights/` 目录。官方来源与文件校验信息见 [weights/README.md](weights/README.md)。

安装并准备相应教师权重后，可运行小规模GPU测试，输出保存在系统临时目录：

```bash
python -m tests.smoke baseline clqkd cwd+clqkd
```

不指定方法时测试全部入口：`python -m tests.smoke`。COCO8仅用于流程测试，论文实验请替换为对应完整数据集及教师权重。

## 数据集

数据集介绍、训练／验证／测试划分及来源链接见[数据集说明](DATASETS.md)。VOC实验采用Ultralytics划分，并保留difficult目标。

## 训练

在 `train.py` 中设置方法、教师权重、数据集路径和训练参数，然后运行：

```bash
python train.py
```

以仓库附带的COCO8数据和自行下载的官方教师运行独立CLQ-KD为例，将以下代码保存于仓库根目录：

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

通过以下参数选择训练方式，表中的 `cwd` 可替换为其他对比方法：

| 训练方式 | 调用参数 |
| --- | --- |
| 普通基线 | `method="baseline"` |
| 单独使用对比KD | `method="cwd", clqkd=False` |
| 对比KD＋CLQ-KD | `method="cwd", clqkd=True, clqkd_beta=1.0` |
| 独立CLQ-KD | `method="clqkd"` |

普通基线和单独使用对比KD时，移除示例中的全部 `clqkd_` 参数。组合训练同样支持 `clqkd_beta`：设为0关闭原生分类监督，设为1保留原方法的分类监督强度，其他值按比例缩放；不改变box、DFL和原方法的蒸馏项。`clqkd=True` 仅用于组合训练。

### CLQ-KD参数

| 参数 | 含义 | 默认值 |
| --- | --- | ---: |
| `clqkd_alpha` | 分类蒸馏损失系数 | 5.0 |
| `clqkd_beta` | 原生分类监督系数 | 独立CLQ-KD：0.0；组合：1.0 |
| `clqkd_teacher_temperature` | 教师温度 | 1.5 |
| `clqkd_student_temperature` | 学生训练温度 | 1.0 |
| `clqkd_prob_threshold` | 教师匹配GT类别的原始概率阈值 | 0.001 |
| `clqkd_iou_threshold` | 教师框与匹配GT框的IoU阈值 | 0.3 |

`train.py` 中显式传入的CLQ-KD参数优先于 `configs/distillation.yaml`。组合模式的默认β在 `clqkd_combination` 配置节设置；其他KD方法的参数在各自配置节中修改。

教师的类别数量和顺序须与数据集一致。CSAKD还要求师生模型结构相同；使用默认YOLO11n学生时，将教师路径改为 `weights/yolo11n.pt`。

### 参数选择建议与适用范围

各数据集的KD系数α、GT分类系数β及师生温度见[CLQ-KD实验参数表](EXPERIMENT_SETTINGS.md)。表中包含论文的目标检测和实例分割实验；当前开源库的训练入口面向YOLO11目标检测，实例分割行仅用于说明论文实验配置。

根据我们的实验经验，当教师提供的响应足够可靠时，CLQ-KD可以仅使用教师提供的分类监督（β=0）。对于小规模数据集，应更加谨慎：在完全依赖教师分类目标之前，需要评估教师的泛化能力。当教师可靠性有限时，保留原生分类监督可能更合适，例如设置β=1。也可以考虑减小α，以限制不可靠教师响应或过强蒸馏项的影响。

此外，KD损失按预测锚点数归一化，但不按类别通道数归一化。因此，类别数增加可能使KD损失总量增大，需要考虑减小α，以保持其与其他目标之间的适当平衡。这些建议是实验经验的总结，并非通用的参数选择规则。合适的设置仍取决于数据集，应针对具体的教师–学生组合进行验证。

## 对比方法

各对比方法的蒸馏特征层级、参数设置、与原论文及官方／公开代码的一致性，以及实现适配，见[对比方法配置表](METHOD_COMPARISON.md)。表中提供对应GitHub代码版本链接。

## 断点续训

```python
from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("runs/detect/train/weights/last.pt")
    model.train(resume=True)
```

恢复检查点中的训练方法、配置和状态，教师继续对当前batch在线计算；请保留原教师权重文件。

## 验证与推理

修改 `val.py`、`predict.py` 中的学生权重和输入路径后运行，无需加载教师：

```bash
python val.py
python predict.py
```

## 论文引用

使用以下方法时，请引用对应原始论文。BibTeX见 [references.bib](references.bib)。

| 方法 | 原始论文 | 发表信息 |
| --- | --- | --- |
| CWD | [Channel-wise Knowledge Distillation for Dense Prediction](https://doi.org/10.1109/ICCV48922.2021.00526) | ICCV 2021 |
| FGD | [Focal and Global Knowledge Distillation for Detectors](https://doi.org/10.1109/CVPR52688.2022.00460) | CVPR 2022 |
| PKD | [PKD: General Distillation Framework for Object Detectors via Pearson Correlation Coefficient](https://papers.neurips.cc/paper_files/paper/2022/hash/631ad9ae3174bf4d6c0f6fdca77335a4-Abstract-Conference.html) | NeurIPS 2022；[DOI](https://doi.org/10.52202/068431-1120) |
| MGD | [Masked Generative Distillation](https://doi.org/10.1007/978-3-031-20083-0_4) | ECCV 2022 |
| CrossKD | [CrossKD: Cross-Head Knowledge Distillation for Object Detection](https://doi.org/10.1109/CVPR52733.2024.01563) | CVPR 2024 |
| GKD | [Gradient-Guided Knowledge Distillation for Object Detectors](https://doi.org/10.1109/WACV57701.2024.00049) | WACV 2024 |
| OST | [Guided Hybrid Quantization for Object Detection in Remote Sensing Imagery via One-to-One Self-Teaching](https://doi.org/10.1109/TGRS.2023.3293147) | IEEE TGRS 2023 |
| ReviewKD | [Distilling Knowledge via Knowledge Review](https://doi.org/10.1109/CVPR46437.2021.00497) | CVPR 2021 |
| CSAKD | [Knowledge Distillation via Cross Supervising with Attention for Remote Sensing Object Detection](https://bmvc2025.bmva.org/proceedings/398/) | BMVC 2025 |

OST对应GHOST论文中的One-to-One Self-Teaching蒸馏部分。

## 许可

Ultralytics框架采用 [AGPL-3.0](LICENSE)，第三方组件保留各自许可，其中CrossKD上游采用CC BY-NC 4.0。来源、许可及第三方使用条件见 [THIRD_PARTY.md](THIRD_PARTY.md)。
