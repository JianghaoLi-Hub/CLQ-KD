# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent

if __name__ == "__main__":
    model = YOLO(str(ROOT / "configs/yolo11n.yaml"))
    model.train(
        # baseline | clqkd | cwd | fgd | pkd | mgd | crosskd | gkd | ost | reviewkd | csakd
        method="clqkd",
        clqkd=False,  # True adds CLQ-KD to a comparison KD; standalone clqkd needs no switch.
        teacher_model=str(ROOT / "weights/yolo11x.pt"),  # Must match the dataset classes and class order.
        clqkd_alpha=5.0,
        clqkd_beta=0.0,  # Also applies to combinations: 0 disables native classification; 1 retains it.
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
