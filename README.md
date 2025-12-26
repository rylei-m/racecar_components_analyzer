# Racecar Component Detection Using YOLOv8
CS5600/6600 Final Project — Rylei Mindrum  
Utah State University

---

## Overview
This project develops an intelligent vision model capable of detecting both **racecars** and **fine-grained racecar components** such as wheels, doors, glass panels, light assemblies, and body sections.  
The system merges two heterogeneous public datasets and trains a unified YOLOv8 model using a fully reproducible AI engineering pipeline.

This repository provides:

- A merged multi-dataset construction pipeline  
- YOLOv8 model training scripts  
- Inference, visualization, and evaluation tools  
- A structured LaTeX report template  
- Integration hooks for aerodynamic simulation research  

---

## Project Structure

~~~
racecar-components-yolo/
├── data/
│   ├── raw/
│   │   ├── racecars/              # Roboflow Racecars dataset
│   │   └── car_components/        # Car Components dataset
│   ├── merged/                    # Uniform YOLOv8 dataset
│   │   └── merged_data.yaml
│
├── src/
│   ├── prepare_merged_dataset.py  # Build merged dataset
│   ├── train.py                   # (Optional) training wrapper
│   ├── infer.py                   # Run inference on images/videos
│   ├── visualize_predictions.py   # Save annotated images
│   ├── eval.py                    # Compute metrics (mAP, AP per class)
│   └── sim_interface.py           # Simulation-ready structured outputs
│
├── runs/                          # YOLOv8 training outputs
├── report/
│   └── final_report.tex           # Full LaTeX report template
└── README.md
~~~

---

## Installation

### 1. Create Conda environment

~~~bash
conda create -n racecar-yolo python=3.10 -y
conda activate racecar-yolo
~~~

### 2. Install dependencies

~~~bash
pip install ultralytics opencv-python numpy matplotlib pyyaml
pip install jupyterlab ipykernel
~~~

---

## Building the Merged Dataset

~~~bash
python src/prepare_merged_dataset.py
~~~

This produces:

~~~
data/merged/{train,val,test}/images
data/merged/{train,val,test}/labels
~~~

---

## Training YOLOv8

~~~bash
yolo task=detect mode=train \
    model=yolov8n.pt \
    data=data/merged/merged_data.yaml \
    epochs=50 \
    imgsz=640 \
    batch=8 \
    project=runs/train \
    name=merged_racecar_components
~~~

---

## Running Inference

~~~bash
python src/infer.py --model runs/train/merged_racecar_components/weights/best.pt \
                    --source path/to/image_or_dir
~~~

---

## Visualizing Predictions

~~~bash
python src/visualize_predictions.py \
    --model runs/train/merged_racecar_components/weights/best.pt \
    --image some_image.jpg
~~~

Results saved under: `visualizations/`

---

## Evaluation

~~~bash
python src/eval.py \
    --model runs/train/merged_racecar_components/weights/best.pt \
    --data data/merged/merged_data.yaml
~~~

---

## Simulation Integration

Import from Python:

~~~python
from src.sim_interface import analyze_components
components = analyze_components("best.pt", "frame.jpg")
print(components)
~~~

---

## Report

The LaTeX report is located at:

~~~
report/final_report.tex
~~~

---

## License

This project uses datasets under CC BY-4.0 or as permitted by Roboflow Universe.
