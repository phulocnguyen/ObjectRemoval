import cv2
import argparse
import os
from models.SwinTransformer import load_sam_model
from models.yolo_model import load_yolo_model
from processing.segmentation import segment_object
from processing.object_detection import detect_objects, draw_detected_objects
from processing.inpainting import removeObject
from processing.postprocessing import refine_output
from utils.video_utils import initialize_video_writer, apply_optical_flow

import cv2

def process_image(image_path, output_path):
    image = cv2.imread(image_path)
    predictor = load_sam_model()
    yolo_model = load_yolo_model()
    

    detected_objects = detect_objects(image, yolo_model)
    
    frame_with_boxes = draw_detected_objects(image.copy(), detected_objects, yolo_model.names)
    cv2.imshow("Detected Objects", frame_with_boxes)
    cv2.waitKey(0) 
    
    selected_class = int(input("Enter the class ID of the object to remove: "))
    selected_object = next((obj for obj in detected_objects if obj[0] == selected_class), None)

    if selected_object:
        mask = segment_object(image, predictor, selected_object)
        
        cv2.imshow("Segmentation Mask", mask)
        cv2.waitKey(0)  
        
        inpainted_image = removeObject(image, mask)
        
        final_image = refine_output(inpainted_image)
    else:
        final_image = image
    cv2.imwrite(output_path, final_image)
    print(f"Processed image saved as {output_path}")

    cv2.destroyAllWindows()


def process_video(video_path, output_path, batch_size=10, resume=False):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out = initialize_video_writer(output_path, width, height, fps)
    predictor = load_sam_model()
    yolo_model = load_yolo_model()
    prev_frame = None
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        detected_objects = detect_objects(frame, yolo_model)
        frame_with_boxes = draw_detected_objects(frame, detected_objects, yolo_model.names)
        cv2.imshow("Select Object to Remove", frame_with_boxes)
        cv2.waitKey(0)
        selected_class = int(input("Enter the class ID of the object to remove: "))
        selected_object = next((obj for obj in detected_objects if obj[0] == selected_class), None)
        if selected_object:
            mask = segment_object(frame, predictor, selected_object)
            inpainted_frame = removeObject(frame, mask)
            if prev_frame is not None:
                inpainted_frame = apply_optical_flow(prev_frame, inpainted_frame)
            final_frame = refine_output(inpainted_frame)
        else:
            final_frame = frame
        out.write(final_frame)
        prev_frame = final_frame
    cap.release()
    out.release()
    print(f"Processed video saved as {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Object Removal")
    parser.add_argument("--mode", type=str, required=True, choices=["image", "video"], help="Processing mode")
    parser.add_argument("--input", type=str, required=True, help="Path to input file")
    parser.add_argument("--output", type=str, required=True, help="Path to save output")
    args = parser.parse_args()
    if args.mode == "image":
        process_image(args.input, args.output)
    else:
        process_video(args.input, args.output)
