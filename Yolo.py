from ultralytics import YOLO

model = YOLO("yolov8n-oiv7.pt")

model.predict(source="0", show=True, save=True, conf=0.2)