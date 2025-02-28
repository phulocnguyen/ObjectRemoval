import cv2
import numpy as np

def initialize_video_writer(output_path, width, height, fps):
    """Initialize video writer for output video."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(output_path, fourcc, fps, (width, height))

def apply_optical_flow(prev_frame, next_frame):
    """Apply optical flow for smoother transitions."""
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    h, w = flow.shape[:2]
    flow_map = np.column_stack((np.repeat(np.arange(h), w), np.tile(np.arange(w), h), flow.reshape(-1, 2)))
    remapped = cv2.remap(next_frame, flow_map[:, 1], flow_map[:, 0], cv2.INTER_LINEAR)
    return remapped