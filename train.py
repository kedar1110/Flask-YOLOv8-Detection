from ultralytics import YOLO

# Load base YOLOv8n
model = YOLO("yolov8n.pt")

# Train on your dataset
model.train(
    data="dataset/data.yaml",
    epochs=100,
    imgsz=640,
    batch=8,
    device='cpu'  # No GPU found, use CPU
)

