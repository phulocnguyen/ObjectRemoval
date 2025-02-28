import cv2
import numpy as np

def refine_output(image):
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(blurred, -1, sharpen_kernel)
    return sharpened