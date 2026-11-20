# 数据集与实验划分

本文档介绍论文所用数据集、训练／验证／测试划分及公开来源。表中的数量以实验采用的版本为准，不随公开下载页的后续更新而改变。分割数据集用于说明论文实验；本仓库的CLQ-KD及对比KD训练入口面向目标检测。

## 划分与用途

模型在训练集上训练，在验证集上选择超参数、检查点并报告结果。PASCAL VOC和RUOD分别使用官方VOC2007 test及RUOD test作为验证集，没有另设独立测试集；其他数据集的测试集不参与本研究实验。

论文Table 1的VOC超参数扫描与对应结果报告采用相同的训练／验证划分：训练16,551张、验证4,952张，均使用保留difficult目标的标签。没有为参数搜索另划调参集，因此这些结果属于同一验证集上的模型选择与性能报告，不应表述为独立留出测试集上的评价。

| 数据集／版本 | 任务 | 类别数 | 训练图像 | 验证图像 | 测试划分及使用情况 |
| --- | --- | ---: | ---: | ---: | --- |
| COCO2017 | 目标检测、实例分割 | 80 | 118,287 | 5,000 | test2017共40,670张，未使用 |
| PASCAL VOC，Ultralytics划分 | 目标检测 | 20 | 16,551 | 4,952 | VOC2007 test已作为验证集；无额外独立测试集 |
| VisDrone2019-DET，Ultralytics适配版 | 目标检测 | 10 | 6,471 | 548 | test-dev 1,610张、test-challenge 1,580张，均未使用 |
| African-Wildlife，Ultralytics适配版 | 目标检测 | 4 | 1,052 | 225 | 227张，未使用 |
| HomeObjects-3K，Ultralytics发布版 | 目标检测 | 12 | 2,285 | 404 | 所用版本未提供单独测试集 |
| URPC2020，Kaggle分发版 | 目标检测 | 4 | 5,543 | 1,200 | 800张（按实验所用版本记录），未使用 |
| RUOD | 目标检测 | 10 | 9,800 | 4,200 | 官方4,200张test已作为验证集；无额外独立测试集 |
| Cityscapes，精细标注子集 | 实例分割 | 8 | 2,975 | 500 | 1,525张，未使用 |
| CarDD | 实例分割 | 6 | 2,816 | 810 | 374张，未使用 |

## 数据集介绍与来源

### COCO2017

COCO2017用于日常场景中的目标检测与实例分割，提供80类目标的边界框和实例掩码标注，包含118,287张训练图像、5,000张验证图像和40,670张测试图像。本研究仅使用训练集和验证集。

来源：[COCO官方网站及下载入口](https://cocodataset.org/#download)；[Ultralytics COCO说明](https://docs.ultralytics.com/datasets/detect/coco/)。

### PASCAL VOC

PASCAL VOC提供20类目标的边界框标注。按照Ultralytics VOC配置中的划分，将VOC2007 trainval的5,011张图像与VOC2012 trainval的11,540张图像合并，得到16,551张训练图像。官方VOC2007 test的4,952张图像用于验证，没有额外独立测试集。训练集和验证集在标注转换时均保留difficult目标。

实验标签**保留difficult目标**，训练集共47,223个框，验证集共14,976个框。这些目标作为普通标注参与训练和验证，不采用VOC官方评测中对difficult目标的忽略处理。

图像划分与标签处理是两个不同设置：Ultralytics 8.3.178自带的VOC转换脚本会排除`difficult=1`目标。复现实验应使用上述保留difficult的标签，或在XML转换时保留这些目标，不能直接用默认转换后的标签替代。

来源：[PASCAL VOC官方网站](http://host.robots.ox.ac.uk/pascal/VOC/)；[Ultralytics VOC说明](https://docs.ultralytics.com/datasets/detect/voc/)；[8.3.178版本的VOC划分与转换脚本](https://github.com/ultralytics/ultralytics/blob/v8.3.178/ultralytics/cfg/datasets/VOC.yaml)。

### VisDrone

使用VisDrone2019-DET的Ultralytics适配版开展无人机视角目标检测，提供10类行人及交通目标的边界框标注。官方划分包含6,471张训练图像、548张验证图像、1,610张test-dev图像和1,580张test-challenge图像。本研究仅使用训练集和验证集。

来源：[VisDrone官方数据仓库与下载入口](https://github.com/VisDrone/VisDrone-Dataset)；[Ultralytics VisDrone说明](https://docs.ultralytics.com/datasets/detect/visdrone/)。

### African-Wildlife

使用Ultralytics适配的African-Wildlife数据集开展野生动物检测，提供水牛、大象、犀牛和斑马四类动物的边界框标注。数据集包含1,052张训练图像、225张验证图像和227张测试图像。本研究仅使用训练集和验证集。

来源：[Ultralytics African-Wildlife说明](https://docs.ultralytics.com/datasets/detect/african-wildlife/)；[所用发布系列的数据包](https://github.com/ultralytics/assets/releases/download/v0.0.0/african-wildlife.zip)。

### HomeObjects-3K

HomeObjects-3K由Ultralytics开发并发布，用于室内家居目标检测，提供家具、电器及其他家居物品等12类目标的边界框标注。数据集包含2,285张训练图像和404张验证图像，本研究使用这两个划分。所用发布版本没有单独测试集。

来源：[Ultralytics HomeObjects-3K说明](https://docs.ultralytics.com/datasets/detect/homeobjects-3k/)；[所用发布系列的数据包](https://github.com/ultralytics/assets/releases/download/v0.0.0/homeobjects-3K.zip)。

### URPC2020

使用Kaggle分发的URPC2020版本开展水下目标检测，提供海参、海胆、扇贝和海星四类海洋生物的边界框标注。该版本包含5,543张训练图像、1,200张验证图像和800张测试图像。本研究仅使用训练集和验证集。

来源：[URPC2020 Kaggle发布页（lywang777/urpc2020）](https://www.kaggle.com/datasets/lywang777/urpc2020)。本实验使用该页面分发的数据版本。

### RUOD

RUOD包含多种真实水下场景中的10类目标，共14,000张图像。沿用官方9,800张训练图像和4,200张测试图像的划分，将后者用于验证与结果报告，没有额外独立测试集。

来源：[RUOD官方数据仓库与下载入口](https://github.com/xiaoDetection/RUOD)。

### Cityscapes

使用Cityscapes精细标注子集评估CLQ-KD在城市驾驶场景中的实例分割性能。实验涵盖person、rider、car、truck、bus、train、motorcycle和bicycle八类目标，实例标注转换为带类别的多边形标签，用于YOLO分割训练。该子集包含2,975张训练图像、500张验证图像和1,525张测试图像。本研究仅使用训练集和验证集。

来源：[Cityscapes官方数据集介绍](https://www.cityscapes-dataset.com/dataset-overview/)；[Cityscapes官方下载入口](https://www.cityscapes-dataset.com/downloads/)。

### CarDD

使用Car Damage Detection（CarDD）数据集评估CLQ-KD在车辆损伤实例分割上的性能。数据集共4,000张图像，涵盖dent、scratch、crack、glass shatter、lamp broken和tire flat六类损伤。COCO格式的实例标注转换为YOLO多边形标签。数据集包含2,816张训练图像、810张验证图像和374张测试图像。本研究仅使用训练集和验证集。

来源：[CarDD官方项目与数据下载说明](https://cardd-ustc.github.io/)。

各数据集的下载、使用与再分发遵循其发布方的许可条款。公开原始数据链接不代表同时提供了本研究转换后的多边形标签或VOC difficult标签包。
