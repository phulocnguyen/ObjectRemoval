import cv2

def detect_objects(image, model):
    results = model(image) 
    detected_objects = []
    
    for box, conf, cls in zip(results[0].boxes.xyxy, results[0].boxes.conf, results[0].boxes.cls):
        x1, y1, x2, y2 = map(int, box.tolist()) 
        detected_objects.append((int(cls.item()), (x1, y1, x2, y2)))
    
    return detected_objects


def draw_detected_objects(image, detected_objects, class_names):
    for cls, (x1, y1, x2, y2) in detected_objects:
        label = class_names[int(cls)]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image