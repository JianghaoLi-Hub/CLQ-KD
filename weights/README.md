# Official COCO pretrained weights

Teacher weights are not included in this repository. For COCO8 functional tests, download the unmodified Ultralytics weights below and place them in this directory. They are not the VOC-finetuned teachers used in the paper experiments.

| File | Official source | SHA256 |
| --- | --- | --- |
| `yolo11x.pt` | [Ultralytics assets v8.3.0](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x.pt) | `7bc158aa95c0ebfdd87f70f01653c1131b93e92522dbe15c228bcd742e773a24` |
| `yolo11n.pt` | [Ultralytics assets v8.3.0](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt) | `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1` |

For other datasets, supply a teacher checkpoint with the same class count and class order as the dataset. Weight files are excluded from Git. Upstream licensing information: [Ultralytics](https://www.ultralytics.com/license).
