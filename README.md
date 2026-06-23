# PCB Component Classification (FICS-PCB)

Image classification of printed circuit board (PCB) component types using YOLOv8
on the **FICS-PCB** dataset from the University of Florida SCAN Lab. The goal is to
recognize component types from cropped PCB imagery, a building block of automated
visual inspection for hardware assurance.

## Motivation

Outsourced PCB fabrication creates a need for assurance, and a growing share of that
work uses machine learning and computer vision on PCB imagery. A recurring bottleneck
is the scarcity and imbalance of labeled data. This project reproduces the component
classification task on the FICS-PCB benchmark and analyzes how class imbalance affects
per-class performance.

## Dataset

FICS-PCB: A Multi-Modal Image Dataset for Automated Printed Circuit Board Visual
Inspection (University of Florida, SCAN Lab). It contains DSLR and microscope images
with component crops sorted into class folders.

- Source: https://www.trust-hub.org/#/data/pcb-images
- Reference paper: https://eprint.iacr.org/2020/366.pdf

The dataset is **not redistributed in this repository**. Download it from the source
above (registration may be required) and follow the steps below.

## Approach

- Used the pre-cropped component images that ship with the dataset, sorted by type.
- Focused on the three well-populated classes present across boards: **capacitor, IC,
  resistor** (the dataset also contains inductor, transistor, and diode, which are too
  sparse for a clean held-out evaluation).
- Split by **board** (whole boards assigned to train / val / test) to measure
  cross-board generalization rather than memorization.
- Trained a YOLOv8 classification model.

## Results

Top-1 accuracy on a held-out board: **82.8%**.

Per-class recall (from the confusion matrix):

| Component | Recall | Notes |
|---|---|---|
| Capacitor | 97.5% | Data-rich class, near perfect |
| Resistor  | 66.8% | Often confused with capacitor |
| IC        | 37.5% | Sparse class, hardest to learn |

The model is strong on the data-rich capacitor class and weaker on the sparse IC class,
with the main error being resistors and ICs predicted as capacitors. This is consistent
with the class imbalance and with the visual similarity between small surface-mount
components noted in the FICS-PCB study, and it illustrates the labeled-data challenge
central to automated optical inspection.

## Reproduce (Google Colab)

1. Download a few board zips (for example s18, s28, s27, s29, s19) from trust-hub and
   unzip them under a `raw/` folder.
2. Build the classification dataset:
   ```bash
   python build_classification_dataset.py --raw raw --out cls --boards s18 s28 s27 s29 s19
   ```
3. Train and evaluate:
   ```bash
   yolo classify train model=yolov8s-cls.pt data=cls epochs=40 imgsz=128
   yolo classify val model=runs/classify/train/weights/best.pt data=cls split=test
   ```

## Files

- `build_classification_dataset.py` builds the train/val/test class-folder layout from
  the FICS-PCB crops, splitting by board.
- `train_classifier.py` optional Python wrapper for training and evaluation.
- `requirements.txt` Python dependencies.

## Tools

Python, YOLOv8 (Ultralytics), OpenCV.
