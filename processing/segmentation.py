import cv2
import numpy as np

def segment_object(image, predictor, selected_object=None):
    predictor.set_image(image)
    masks, _, _ = predictor.predict()
    if not masks:
        return None  # case no mask is found
    
    mask = masks[0]
    kernel = np.ones((5, 5), np.uint8)
    refined_mask = cv2.dilate(mask, kernel, iterations=1)
    return refined_mask
