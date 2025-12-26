from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")
    results = model("https://ultralytics.com/images/bus.jpg", save=True)
    print("Inference completed. Check the 'runs/detect' directory for outputs.")

if __name__ == "__main__":
    main()
