# loads YOLOv8 Model, Runs Inference, Saves annotated Images, Prints out Structured Detection Results

import argparse
from pathlib import Path
from ultralytics import YOLO

def run_inference(model_path: str, source: str, save: bool = True):
    model = YOLO(model_path)

    results = model(source, save=save)

    for r in results:
        print("=" * 60)
        print(f"Image: {r.path}")
        print(f"Detections: {len(r.boxes)}")

        for box in r.boxes:
            cls_id = int(box.cls)
            conf = float(box.conf)
            xyxy = box.xyxy[0].tolist()

            print(f"  Class {cls_id}  Conf: {conf:.3f}  BBox: {xyxy}")

    print("Done. Annotated images saved under 'runs/detect/'.")


def main():
    parser = argparse.ArgumentParser(description="YOLOv8 Inference Utility")
    parser.add_argument("--model", required=True, help="Path to YOLO model (.pt)")
    parser.add_argument("--source", required=True, help="Image file, directory, or video path")
    parser.add_argument("--nosave", action="store_true", help="If set, do not save annotated images")

    args = parser.parse_args()
    run_inference(args.model, args.source, save=not args.nosave)


if __name__ == "__main__":
    main()
