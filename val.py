from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent

if __name__ == "__main__":
    YOLO(str(ROOT / "runs/detect/train/weights/best.pt")).val(data=str(ROOT / "datasets/coco8/data.yaml"))
