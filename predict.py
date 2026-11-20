from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent

if __name__ == "__main__":
    YOLO(str(ROOT / "runs/detect/train/weights/best.pt")).predict(
        source=str(ROOT / "datasets/coco8/images/val/000000000036.jpg"), save=True
    )
