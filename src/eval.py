import argparse
from ultralytics import YOLO

def evaluate(model_path, data_yaml, save_json=False):
    model = YOLO(model_path)
    metrics = model.val(data=data_yaml, save_json=save_json)

    print("\n=== EVALUATION RESULTS ===")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}\n")

    print("Per-class AP:")
    for cls_name, ap in zip(metrics.names.values(), metrics.box.maps):
        print(f"  {cls_name}: {ap:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--save-json", action="store_true")

    args = parser.parse_args()
    evaluate(args.model, args.data, args.save_json)


if __name__ == "__main__":
    main()
