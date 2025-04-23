import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from tqdm import tqdm

def predict_character_folder(folder_path='./chu', model_path='./models1/new_best_cnn.keras', output_dir='./results'):
    # Các tham số cố định
    IMG_WIDTH = 50
    IMG_HEIGHT = 150
    classes = ['a', 'a1', 'a2', 'b', 'c', 'd', 'd1', 'e', 'e1', 'g',
               'h', 'i', 'k', 'l', 'm', 'n', 'o', 'o1', 'o2', 'p',
               'q', 'r', 's', 't', 'u', 'u1', 'v', 'x', 'y']
    
    vietnamese_map = {
        'a1': 'â', 'a2': 'ă', 'd1': 'đ', 'e1': 'ê',
        'o1': 'ô', 'o2': 'ơ', 'u1': 'ư'
    }
    
    # Kiểm tra file và thư mục
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Không tìm thấy file mô hình: {model_path}")
    
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Không tìm thấy thư mục ảnh: {folder_path}")
    
    # Tạo thư mục đầu ra nếu chưa tồn tại
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Kiểm tra định dạng ảnh hợp lệ
    valid_extensions = ['.png', '.jpg', '.jpeg']
    image_files = [f for f in os.listdir(folder_path) 
                   if any(f.lower().endswith(ext) for ext in valid_extensions)]
    
    if not image_files:
        raise ValueError(f"Không tìm thấy ảnh hợp lệ trong thư mục: {folder_path}")
    
    # Tải mô hình
    try:
        model = load_model(model_path, compile=False)
        model.compile(loss='categorical_crossentropy',
                     optimizer='adam',
                     metrics=['accuracy'])
    except Exception as e:
        raise RuntimeError(f"Lỗi khi tải mô hình: {str(e)}")
    
    # Lưu kết quả tổng hợp
    results = []
    
    # Xử lý từng ảnh với thanh tiến trình
    for image_file in tqdm(image_files, desc="Đang xử lý ảnh"):
        image_path = os.path.join(folder_path, image_file)
        
        try:
            # Đọc và tiền xử lý ảnh
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Cảnh báo: Không thể đọc ảnh {image_path}")
                continue
            
            # Resize và chuẩn hóa
            img_resized = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)
            img_normalized = img_resized / 255.0
            img_input = img_normalized.reshape(1, IMG_HEIGHT, IMG_WIDTH, 1)
            
            # Kiểm tra kích thước đầu vào
            expected_shape = model.input_shape[1:]
            if img_input.shape[1:] != expected_shape:
                img_input = cv2.resize(img_normalized, (expected_shape[1], expected_shape[0]))
                img_input = img_input.reshape(1, expected_shape[0], expected_shape[1], expected_shape[2])
            
            # Dự đoán
            prediction = model.predict(img_input, verbose=0)
            predicted_class = np.argmax(prediction[0])
            confidence = prediction[0][predicted_class]
            
            # Xử lý kết quả
            if predicted_class < len(classes):
                predicted_char = classes[predicted_class]
                display_char = vietnamese_map.get(predicted_char, predicted_char)
            else:
                display_char = f"Unknown ({predicted_class})"
            
            # Lưu ảnh kết quả
            plt.figure(figsize=(6, 4))
            plt.imshow(img, cmap='gray')
            plt.title(f"Dự đoán: {display_char} (độ tin cậy: {confidence:.4f})")
            plt.axis('off')
            output_path = os.path.join(output_dir, f"result_{image_file}")
            plt.savefig(output_path)
            plt.close()
            
            # Lưu kết quả
            results.append({
                'image': image_file,
                'predicted_char': display_char,
                'confidence': confidence
            })
            
            print(f"Ảnh {image_file}: Ký tự dự đoán: {display_char}, Độ tin cậy: {confidence:.4f}")
        
        except Exception as e:
            print(f"Lỗi khi xử lý ảnh {image_path}: {str(e)}")
            continue
    
    # Lưu kết quả tổng hợp vào file
    summary_path = os.path.join(output_dir, 'prediction_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("Kết quả dự đoán ký tự viết tay:\n\n")
        for result in results:
            f.write(f"Ảnh: {result['image']}\n")
            f.write(f"Ký tự dự đoán: {result['predicted_char']}\n")
            #f.write(f"Độ tin cậy: {result['confidence']:.4f}\n")
            f.write("-" * 50 + "\n")
    
    print(f"\nĐã xử lý xong {len(results)} ảnh")
    print(f"Kết quả tổng hợp được lưu tại: {summary_path}")
    
    return results

if __name__ == "__main__":
    try:
        print("Bắt đầu dự đoán ký tự viết tay từ thư mục...")
        predict_character_folder()
    except Exception as e:
        print(f"Lỗi: {str(e)}")