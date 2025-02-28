from ultralytics import YOLO

def load_yolo_model(model_path='yolov8.pt'):
    """Load YOLO model for object detection."""
    return YOLO(model_path)