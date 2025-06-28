# src/detect.py

from ultralytics import YOLO

class detect_players:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def __call__(self, frame):
        results = self.model(frame, verbose=False)[0]
        detections = []

        for box in results.boxes:
            xyxy = box.xyxy[0].tolist()
            x1, y1, x2, y2 = map(int, xyxy)
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            if conf > 0.4 and int(cls) == 0:  # Class 0 = person
                w, h = x2 - x1, y2 - y1
                detections.append([x1, y1, w, h, conf])  # ✅ CORRECT FORMAT

        return detections  # ✅ Final output: list of [x, y, w, h, conf]
