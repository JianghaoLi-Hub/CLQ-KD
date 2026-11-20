# Third-Party Sources and Licenses

This repository extends Ultralytics 8.3.178 under [AGPL-3.0](LICENSE) with a KD registry, shared training lifecycle, method modules and PyTorch compatibility support. Third-party components retain their own copyright and license notices.

The root license does not override third-party licenses or grant additional rights to third-party code. License statements below apply only to the sources and components identified.

## Method Sources

| Method | Reference and source revision | Implementation and license |
| --- | --- | --- |
| CLQ-KD | Classification/localization quality-filtering implementation; entry-point style from [LLV-YOLO](https://github.com/JianghaoLi-Hub/LLV-YOLO) | Temperature-scaled classification distillation in the AGPL framework. |
| CrossKD | [Cross-Head Knowledge Distillation](https://github.com/jbwang1997/CrossKD/tree/87e156a46259e3a516b65fc6a049b200b9ab0d54) | YOLO11 response/head adaptation. Upstream [CC BY-NC 4.0](licenses/CrossKD-CC-BY-NC-4.0.txt) permits noncommercial sharing and adaptation with attribution, a license link and an indication of changes. This implementation adapts the cross-head position, channel alignment and training integration to YOLO11. The noncommercial restriction remains applicable; commercial use requires separate permission. |
| CSAKD | [YOLOv8-KD, OtherKD/CSAKD.py](https://github.com/KefanZhan/YOLOv8-KD/tree/3433559ab6c9cfdef3b4ac099f1bb2feddd42dc8) | Author executable-loss reference; full-batch, same-architecture YOLO11 adaptation. |
| CWD | [mmdetection-distiller](https://github.com/pppppM/mmdetection-distiller/tree/7a8a1306c35bd0521946870179f0c475db4cacd8), [ICCV 2021 paper](https://openaccess.thecvf.com/content/ICCV2021/html/Shu_Channel-Wise_Knowledge_Distillation_for_Dense_Prediction_ICCV_2021_paper.html) | Spatial KL and YOLO channel adapters. [Apache-2.0](licenses/CWD-Apache-2.0.txt), Copyright 2018-2019 Open-MMLab. |
| FGD | [Focal and Global Knowledge Distillation](https://github.com/yzd-v/FGD/tree/c03a6cf3b3cb3cbfd5554dbf4766e7135747d22d) | Detection masks, attention and relation terms adapted to YOLO. [Upstream license text](licenses/FGD-Apache-2.0.txt), including its retained notices. |
| GKD | [WACV 2024 paper](https://openaccess.thecvf.com/content/WACV2024/html/Lan_Gradient-Guided_Knowledge_Distillation_for_Object_Detectors_WACV_2024_paper.html), [author reference](https://github.com/lanqz7766/GKD/tree/257eedc3532db676ecc99d4fd4af35fb16dbed54) | Paper-guided first-order YOLO gradient maps, excluding BMFI. |
| MGD | [Masked Generative Distillation](https://github.com/yzd-v/MGD/tree/2c9da0b28625eb948db57afc02c824452c3910fe) | Masked generator and reduction adapted to YOLO. [Apache-2.0](licenses/MGD-Apache-2.0.txt). |
| OST | [GHOST, utils/distill/Self_D.py](https://github.com/icey-zhang/GHOST/blob/89b4d9213e082a4de8e3dca1b3a47f5afbb13d1c/utils/distill/Self_D.py), [paper](https://ieeexplore.ieee.org/abstract/document/10175627) | Attention-based Feature-level Distillation; Copyright (c) 2021-present NAVER Corp. The referenced file declares [Apache-2.0](licenses/OST-Apache-2.0.txt). Modified for compact YOLO stage features and the shared trainer; not the complete quantization system. |
| PKD | [MMRazor PKD loss](https://github.com/open-mmlab/mmrazor/blob/90c7af1fdf35a606a2fceaaeb4b6be2f0dac4eb7/mmrazor/models/losses/pkd_loss.py), [NeurIPS 2022 paper](https://papers.neurips.cc/paper_files/paper/2022/file/631ad9ae3174bf4d6c0f6fdca77335a4-Paper-Conference.pdf) | Copyright (c) OpenMMLab. All rights reserved. [Apache-2.0](licenses/PKD-Apache-2.0.txt); modified with YOLO adapters and shared integration. |
| ReviewKD | [Distilling Knowledge via Knowledge Review](https://github.com/JIA-Lab-research/ReviewKD/tree/cede6ea6387ae9b6127de0e561507177bf19c11e), Detection/model/kd_trans.py | ABF and hierarchical context loss adapted to YOLO. |

## Licensing Notes

Retain the applicable copyright notices and license texts when distributing components. Paper citations identify research sources and do not grant code redistribution rights. Teacher weights and datasets are subject to their own terms.
