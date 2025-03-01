import cv2
import torch
import numpy as np
from segment_anything import sam_model_registry, SamPredictor

# Đường dẫn đến checkpoint của SAM
checkpoint = "/Users/phulocnguyen/Documents/Workspace/VideoObjectRemoval/sam_vit_h_4b8939.pth"

# Hàm tải mô hình SAM
def load_sam_model(model_type='vit_h', checkpoint=checkpoint):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[model_type](checkpoint=checkpoint).to(device)
    return SamPredictor(sam)

# Hàm kiểm tra mô hình với một ảnh đầu vào
def test_sam(image_path):
    # Load mô hình
    predictor = load_sam_model()

    # Đọc ảnh
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Không thể đọc ảnh từ {image_path}")
        return
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Chuyển sang RGB
    predictor.set_image(image_rgb)  # Đưa ảnh vào mô hình

    # Chọn một điểm cố định (ví dụ, giữa ảnh)
    height, width, _ = image.shape
    input_point = np.array([[width // 2, height // 2]])
    input_label = np.array([1])  # 1 = foreground

    # Dự đoán mask
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True  # Trả về nhiều mask
    )

    # Hiển thị kết quả
    for i, mask in enumerate(masks):
        mask_image = (mask * 255).astype(np.uint8)  # Chuyển mask sang ảnh
        cv2.imshow(f"Mask {i+1} - Score: {scores[i]:.4f}", mask_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Chạy kiểm tra với ảnh mẫu
test_sam("/Users/phulocnguyen/Documents/Workspace/VideoObjectRemoval/test.jpg")
