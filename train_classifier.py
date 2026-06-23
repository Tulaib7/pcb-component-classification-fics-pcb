"""
train_classifier.py
Train and evaluate a YOLOv8 classification model on the prepared dataset.

Usage:
    python train_classifier.py --data cls --epochs 40 --imgsz 128
"""
import argparse
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to the classification dataset folder")
    ap.add_argument("--model", default="yolov8s-cls.pt")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--imgsz", type=int, default=128)
    args = ap.parse_args()

    model = YOLO(args.model)
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz)
    metrics = model.val(split="test")
    print("Top-1 accuracy (test):", round(metrics.top1, 4))


if __name__ == "__main__":
    main()
