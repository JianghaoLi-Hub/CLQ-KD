# 对比蒸馏方法的特征层级、参数设置与实现适配

下表汇总九种对比方法在本仓库中的蒸馏层级、参数设置，以及与原论文和参考代码的对应关系。参数配置见 [configs/distillation.yaml](configs/distillation.yaml)，原始论文见 [README中的论文引用](README.zh-CN.md#论文引用)。GitHub链接指向采用的代码版本。

| 方法 | 参考代码实现（GitHub） | 蒸馏特征层级（本仓库） | 本仓库所用参数 | 与论文参数的一致性 | 与参考代码参数的一致性 | 相对于参考实现的适配与修改 |
| --- | --- | --- | --- | --- | --- | --- |
| **CWD** | [mmdetection-distiller（公开实现）](https://github.com/pppppM/mmdetection-distiller/tree/7a8a1306c35bd0521946870179f0c475db4cacd8) | P3／P4／P5，第16／19／22层 | 权重5，τ=1 | 论文未明确报告对应检测参数 | **一致** | **有**：YOLO接入、通道对齐，采用完整KL |
| **FGD** | [FGD（官方）](https://github.com/yzd-v/FGD/tree/c03a6cf3b3cb3cbfd5554dbf4766e7135747d22d) | P3／P4／P5，第16／19／22层 | T=0.5；α=0.001，β=0.0005，γ=0.0005，λ=0.000005 | **部分一致**：论文对应单阶段配置中γ=0.001 | **一致**：GFL／RetinaNet配置 | **有**：YOLO三尺度接入及通道适配 |
| **PKD** | [MMRazor／PKD（官方）](https://github.com/open-mmlab/mmrazor/tree/90c7af1fdf35a606a2fceaaeb4b6be2f0dac4eb7) | P3／P4／P5，第16／19／22层 | 权重10 | **一致**：单阶段教师设置 | **一致** | **有**：增加学生侧1×1通道对齐 |
| **MGD** | [MGD（官方）](https://github.com/yzd-v/MGD/tree/2c9da0b28625eb948db57afc02c824452c3910fe) | P3／P4／P5，第16／19／22层 | α=2×10⁻⁵，遮罩比例0.65 | **一致**：单阶段检测设置 | **一致** | **有**：YOLO三尺度接入及重建模块适配 |
| **CrossKD** | [CrossKD（官方）](https://github.com/jbwang1997/CrossKD/tree/87e156a46259e3a516b65fc6a049b200b9ab0d54) | P3／P4／P5检测头的分类、回归分支中间特征；蒸馏交叉预测 | 聚焦指数1，温度1，分类／回归权重1／4 | 聚焦指数一致；完整参数设置采用官方GFL配置 | **一致**：GFL配置 | **有**：交叉位置适配为YOLO分支首块BN输出（激活前） |
| **ReviewKD** | [ReviewKD（官方）](https://github.com/JIA-Lab-research/ReviewKD/tree/cede6ea6387ae9b6127de0e561507177bf19c11e) | P3／P4／P5，第16／19／22层 | 权重2，ABF中间通道256 | **权重不同**：论文补充材料为0.5 | **一致**：MV2–R50检测配置 | **有**：ABF／HCL适配YOLO三尺度 |
| **GKD** | [GKD（官方）](https://github.com/lanqz7766/GKD/tree/257eedc3532db676ecc99d4fd4af35fb16dbed54) | P3／P4／P5的梯度引导特征图，第16／19／22层 | 权重0.125；层权重1／1／1 | 论文未给出通用权重；0.125通过本仓库的参数搜索选定 | **不同**：官方GFL／COCO配置为0.5 | **有**：可微一阶YOLO适配，仅使用GKD，不含BMFI |
| **OST** | [GHOST／OST（官方）](https://github.com/icey-zhang/GHOST/tree/89b4d9213e082a4de8e3dca1b3a47f5afbb13d1c) | Detect之前第0–22层，共23个节点 | 权重5；qk_dim=128，τ=1，门控阈值0.5 | **权重不同**：5通过本仓库的参数搜索选定 | **部分一致**：qk_dim、τ及门控阈值一致；官方训练入口默认权重400 | **有**：YOLO节点、FP32训练及损失缩放适配 |
| **CSAKD** | [YOLOv8-KD／CSAKD（官方）](https://github.com/KefanZhan/YOLOv8-KD/tree/3433559ab6c9cfdef3b4ac099f1bb2feddd42dc8) | Detect之前第0–22层，共23个节点及对应交叉特征 | α=300，β=10 | **部分一致**：论文推荐400／10；α通过本仓库的参数搜索选定 | β一致；α通过本仓库的参数搜索选定 | **有**：同架构YOLO11n适配，保留作者代码的损失核心 |

注：层编号采用YOLO11模型的零起始索引；P3／P4／P5分别对应步长8／16／32。参数一致性指表中所列数值，整体训练流程和损失尺度仍取决于具体实现。各方法中的α、β、λ为其自身参数，不等同于CLQ-KD的同名系数。
