# Object Removal
## Overview

This project implements object removal from images using YOLOv8 for object detection, SwinTransformer for segmentation and the LaMa model for image inpainting. The pipeline detects objects, allows users to select an object for removal, and seamlessly fills the removed area using advanced inpainting techniques for natural-looking results.

## Features

**Object Detection**: Uses YOLO to detect objects in images.

**Instance Segmentation**: Utilizes Swin Transformer for precise segmentation.

**Inpainting**: Removes selected objects and fills the missing areas using an inpainting algorithm.

**User Interaction**: Allows users to choose which object to remove from the detected objects.

## Installation

### Prerequisites

**Ensure you have the following installed:**

**Python 3.8+**

**PyTorch**

**OpenCV**

### Install Dependencies
```bash
pip install -r requirements.txt
```
### Usage

Run Object Removal
```bash
python main.py --input path/to/image.jpg --output path/to/output.jpg
```
### Example Workflow

1. The program detects objects in the input image.

2. The user selects an object class to remove.

3. Swin Transformer segments the selected object.

4. The segmented area is removed and filled using inpainting.

5. The final image is saved to the output path.


### Model Details

**Swin Transformer** (Used for instance segmentation): Provides accurate and efficient segmentation for complex objects.

**YOLOv8** (Used for object detection): Identifies objects in images before segmentation.

**Inpainting Method**: Uses Lama Model & inpainting algorithm to reconstruct missing regions.

### Acknowledgments

**Swin Transformer**: Microsoft Research

**YOLOv8**: Ultralytics

**Lama model**

### License

This project is licensed under the MIT License.