# Racecar Component Detection Using YOLOv8

**CS5600 / CS6600 — Final Project**  
**Rylei Mindrum**  
Utah State University

## Overview

This repository implements a complete, reproducible object detection pipeline for identifying both **racecars** and **fine-grained automotive components** from RGB imagery using **YOLOv8**.

The project has two primary objectives:

1. Demonstrate a full supervised learning workflow aligned with the data-centric and model-centric AI engineering principles taught in CS5600/6600  
2. Support downstream **aerodynamic simulation research** by automatically identifying vehicle components such as wheels, bumpers, glass panels, doors, and mirrors

To achieve this, the system merges two heterogeneous public datasets, harmonizes labels into a unified taxonomy, and trains a single YOLOv8 model capable of reasoning about both **global vehicle structure** and **component-level detail**.

## Key Features

- Reproducible multi-dataset merging pipeline  
- Unified **16-class YOLOv8 detection model** (15 components + racecar)  
- Quantitative evaluation using mAP, precision, recall, F1, and confusion matrices  
- Inference and visualization utilities  
- Simulation-ready Python interface for aerodynamic analysis  
- Full LaTeX report documenting methodology and results  

## Datasets

This project integrates two datasets from **Roboflow Universe**:

- [Racecars Dataset](https://universe.roboflow.com/tyrone-brock/racecars-n6toi)
- [Car Components Dataset](https://universe.roboflow.com/sammy/car-components-dataset)

### Racecars Dataset

- Provides whole-vehicle bounding boxes and coarse structural context  
- High variation in car types, lighting, motion blur, and occlusion  
- Supplies the **racecar** class and global vehicle geometry  

### Car Components Dataset

- Annotates 15 fine-grained automotive components:
  - wheels  
  - doors (left / right)  
  - headlights and taillights (left / right)  
  - mirrors (left / right)  
  - bumpers  
  - glass panels  
  - hood  
- Small object sizes and high inter-class similarity make detection challenging  

### Merged Dataset Construction

A custom preprocessing pipeline:

- converts all annotations to normalized YOLO format  
- enforces consistent class naming and index ordering  
- prefixes filenames to avoid collisions  
- removes corrupted or duplicate images  
- splits data into train / validation / test sets  

**Final validation set statistics**

- 356 images  
- 2,041 annotated instances  
- 16 total classes  

## Project Structure

```text
racecar-components-yolo/
├── data/
│   ├── raw/
│   │   ├── racecars/
│   │   └── car_components/
│   └── merged/
│       ├── train/
│       ├── val/
│       ├── test/
│       └── merged_data.yaml
│
├── src/
│   ├── prepare_merged_dataset.py
│   ├── train.py
│   ├── infer.py
│   ├── visualize_predictions.py
│   ├── eval.py
│   └── sim_interface.py
│
├── runs/
├── report/
│   └── final_report.tex
└── README.md
```

## Installation

### Create environment

conda create -n racecar-yolo python=3.10 -y  
conda activate racecar-yolo  

### Install dependencies

pip install ultralytics opencv-python numpy matplotlib pyyaml  
pip install jupyterlab ipykernel  

## Build the Merged Dataset

python src/prepare_merged_dataset.py  

This generates YOLO-formatted train, validation, and test directories under data/merged.

## Model Training

Training uses the YOLOv8n (nano) architecture for fast iteration and real-time inference.

yolo task=detect mode=train model=yolov8n.pt data=data/merged/merged_data.yaml epochs=50 imgsz=640 batch=8 project=runs/train name=merged_racecar_components  

Training highlights include pretrained COCO weights, AdamW optimization, cosine learning rate decay, and geometric and photometric augmentations.

## Evaluation Results

Performance on the held-out validation set:

| Metric      | Value  |
|-------------|--------|
| Precision   | 0.674  |
| Recall      | 0.908  |
| mAP@50      | 0.7125 |
| mAP@50–95   | 0.5453 |

High recall is intentional. For aerodynamic simulation, missing a component is more costly than detecting an extra one. Large, high-contrast components such as wheels, glass, and bumpers perform best, while mirrors and side lights remain challenging due to small size and symmetry.

## Inference

python src/infer.py --model runs/train/merged_racecar_components/weights/best.pt --source path/to/image_or_directory  

## Visualization

python src/visualize_predictions.py --model runs/train/merged_racecar_components/weights/best.pt --image example.jpg  

Annotated outputs are saved to the visualizations directory.

## Simulation Integration

The repository includes a clean Python interface for extracting detected components in a simulation-friendly format.

from src.sim_interface import analyze_components  

components = analyze_components("runs/train/merged_racecar_components/weights/best.pt", "frame.jpg")  
print(components)  

Example output:

| Component        | Confidence | Bounding Box (x1, y1, x2, y2) |
|------------------|------------|-------------------------------|
| wheel            | 0.98       | (75, 260, 155, 340)           |
| front_bumper     | 0.94       | (112, 240, 398, 360)          |
| front_glass      | 0.96       | (140, 110, 420, 220)          | 

These detections can be normalized and ingested by aerodynamic surface estimators, drag and lift coefficient models, mesh-mask generators for CFD, and temporal tracking pipelines.

## Report

The full project report is located at:

[Final Project Report (LaTeX)](RacecarComponentDetection.TeX)

The document corresponds directly to the experiments and code in this repository.

## License

Datasets are used under CC BY-4.0 or as permitted by Roboflow Universe.  
This repository is intended for educational and research use.
