from ultralytics import YOLO

class DetectedComponent:
    def __init__(self, cls_name, conf, x1, y1, x2, y2):
        self.cls_name = cls_name
        self.conf = conf
        self.bbox = (x1, y1, x2, y2)

    def __repr__(self):
        return f"{self.cls_name}({self.conf:.2f}): {self.bbox}"


def analyze_components(model_path, image_path):
    model = YOLO(model_path)
    results = model(image_path)[0]

    detected = []
    names = model.names

    for box in results.boxes:
        cls_id = int(box.cls)
        conf = float(box.conf)
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        detected.append(
            DetectedComponent(
                cls_name=names[cls_id],
                conf=conf,
                x1=x1, y1=y1, x2=x2, y2=y2
            )
        )

    return detected
