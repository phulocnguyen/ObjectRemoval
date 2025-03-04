# Object Removal with MAT

## Overview

This project implements object removal from images and videos using **YOLOv8** for object detection, **Swin Transformer** for instance segmentation, and **Masked Attention Transformer (MAT)** for high-quality inpainting. The pipeline efficiently detects, segments, and removes objects while reconstructing the missing areas seamlessly.

## Features

- **Object Detection**: Utilizes **YOLOv8** for precise object localization.
- **Instance Segmentation**: Uses **Swin Transformer** for accurate object segmentation.
- **Inpainting with MAT**: Leverages **Masked Attention Transformer (MAT)** to fill in missing regions with high fidelity.
- **User Interaction**: Allows users to choose specific objects for removal from detected objects.
- **Video Processing**: Supports object removal in videos frame-by-frame.

## Installation

### Prerequisites
Ensure you have the following installed:

- **Python 3.8+**
- **PyTorch**
- **OpenCV**
- **CUDA (if using GPU acceleration)**

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Object Removal from Image
```bash
python main.py --mode image --input path/to/image.jpg --output path/to/output.jpg
```

### Object Removal from Video
```bash
python main.py --mode video --input path/to/video.mp4 --output path/to/output.mp4
```

## Example Workflow

1. **Object Detection**: The program detects objects in the input image or video using **YOLOv8**.
2. **User Selection**: The user selects an object class for removal.
3. **Segmentation**: **Swin Transformer** segments the selected object.
4. **Object Removal & Inpainting**:
   - The segmented object is removed.
   - The missing area is reconstructed using **Masked Attention Transformer (MAT)**.
5. **Final Output**: The processed image or video is saved at the specified output path.

## Model Details

### **YOLOv8** (Object Detection)
- Used to detect and localize objects in images and videos before segmentation.
- Provides high accuracy and real-time performance.

### **Swin Transformer** (Instance Segmentation)
- Efficient hierarchical vision transformer model.
- Provides precise segmentation for complex objects.

### **Masked Attention Transformer (MAT)** (Inpainting)
- Advanced transformer-based inpainting model.
- Effectively fills missing regions while maintaining structural coherence.

## Acknowledgments

- **YOLOv8**: Developed by Ultralytics
- **Swin Transformer**: Developed by Microsoft Research
- **Masked Attention Transformer (MAT)**: Cutting-edge image inpainting model

## License

This project is licensed under the **MIT License**.