import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os
import cv2

# Tham số được lưu trực tiếp
IMAGE_PATH = "â.png"  # Đường dẫn đến hình ảnh cần dự đoán
MODEL_PATH = "save_model/resnet18_trained.pth"  # Đường dẫn đến tệp mô hình đã lưu

# 25 chữ cái từ a đến y
CLASS_NAMES = ['a', 'b', 'c', 'd', 'e', 'g', 'h', 'i', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'x', 'y', 'â', 'ê', 'ô', 'ă', 'đ', 'ơ', 'ư']  


def load_model(model_path, num_classes, device):
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
    return model

def preprocess_image(image_path):
    
    image = Image.open(image_path)
    
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3), 
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    try:
        image = transform(image)
        image = image.unsqueeze(0)
        return image
    except Exception as e:
        raise Exception(f"Không thể xử lý ảnh {image_path}: {e}")

def predict(image_path, model, class_names, device):
    image = preprocess_image(image_path)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)
        print(probabilities)
        predicted_class = class_names[predicted.item()]
        predicted_prob = probabilities[0][predicted.item()].item()
    return predicted_class, predicted_prob

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Số lớp
    num_classes = len(CLASS_NAMES)

    # Kiểm tra tệp mô hình
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Tệp mô hình {MODEL_PATH} không tồn tại")

    # Tải mô hình
    try:
        model = load_model(MODEL_PATH, num_classes, device)
    except RuntimeError as e:
        print(f"Lỗi khi tải mô hình: {e}")
        print("Có thể mô hình đã được huấn luyện trên số lớp khác. Vui lòng huấn luyện lại mô hình với 25 lớp.")
        return

    # Kiểm tra ảnh đầu vào
    if not os.path.exists(IMAGE_PATH):
        raise FileNotFoundError(f"Ảnh {IMAGE_PATH} không tồn tại")

    # Dự đoán
    try:
        predicted_class, predicted_prob = predict(IMAGE_PATH, model, CLASS_NAMES, device)
        true_class = os.path.basename(os.path.dirname(IMAGE_PATH))
        print(f"Hình ảnh: {IMAGE_PATH}")
        print(f"Lớp dự đoán: {predicted_class} (Xác suất: {predicted_prob:.4f})")
    except Exception as e:
        print(f"Lỗi: {e}")

if __name__ == "__main__":
    main()