import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import seaborn as sns
import pandas as pd

def evaluate_models_on_dataset():
    # Các tham số cố định
    IMG_WIDTH = 50
    IMG_HEIGHT = 150
    DATA_DIR = "../testHTN"
    MODEL_PATHS = {
        "CNN": "./models/best_cnn.keras",
        "CNN_LSTM": "./models/best_cnn_lstm.keras"
    }
    
    classes = ['a', 'a1', 'a2', 'b', 'c', 'd', 'd1', 'e', 'e1', 'g',
               'h', 'i', 'k', 'l', 'm', 'n', 'o', 'o1', 'o2', 'p',
               'q', 'r', 's', 't', 'u', 'u1', 'v', 'x', 'y']
    
    vietnamese_map = {
        'a1': 'ă', 'a2': 'â', 'd1': 'đ', 'e1': 'ê',
        'o1': 'ô', 'o2': 'ơ', 'u1': 'ư'
    }

    def load_test_dataset(data_dir, classes):
        X, y_true, img_paths = [], [], []
        print("Đang tải dữ liệu kiểm tra...")
        
        for idx, label in enumerate(classes):
            class_dir = os.path.join(data_dir, label)
            if not os.path.exists(class_dir):
                print(f"Thư mục {class_dir} không tồn tại, bỏ qua.")
                continue
            
            for img_name in os.listdir(class_dir):
                if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                
                img_path = os.path.join(class_dir, img_name)
                try:
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        print(f"Lỗi: Không thể đọc ảnh {img_path}")
                        continue
                    
                    img_resized = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)
                    img_normalized = img_resized / 255.0
                    img_input = img_normalized.reshape(IMG_HEIGHT, IMG_WIDTH, 1)
                    
                    X.append(img_input)
                    y_true.append(idx)
                    img_paths.append(img_path)
                except Exception as e:
                    print(f"Lỗi khi xử lý ảnh {img_path}: {e}")
        
        if not X:
            print("Không tìm thấy ảnh nào trong thư mục dữ liệu!")
            return None, None, None
        
        X = np.array(X)
        y_true = np.array(y_true)
        print(f"Đã tải {len(X)} ảnh kiểm tra.")
        return X, y_true, img_paths

    def evaluate_model(model, model_name, X_test, y_true, classes):
        try:
            # Dự đoán
            predictions = model.predict(X_test, verbose=1)
            y_pred = np.argmax(predictions, axis=1)
            
            # Tính các chỉ số
            overall_accuracy = accuracy_score(y_true, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
            
            print(f"\n{model_name} - Kết quả đánh giá:")
            print(f"Độ chính xác tổng thể: {overall_accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1-score: {f1:.4f}")
            
            # Tính độ chính xác cho từng lớp
            class_accuracies = {}
            for idx, class_name in enumerate(classes):
                class_mask = (y_true == idx)
                if np.sum(class_mask) == 0:
                    class_accuracies[class_name] = 0.0
                    continue
                class_accuracy = accuracy_score(y_true[class_mask], y_pred[class_mask])
                class_accuracies[class_name] = class_accuracy
            
            # Vẽ confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(12, 10))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=[vietnamese_map.get(c, c) for c in classes],
                       yticklabels=[vietnamese_map.get(c, c) for c in classes])
            plt.title(f'Confusion Matrix - {model_name}')
            plt.xlabel('Dự đoán')
            plt.ylabel('Thực tế')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f'confusion_matrix_{model_name}.png')
            plt.close()
            
            return y_pred, overall_accuracy, class_accuracies, precision, recall, f1
        except Exception as e:
            print(f"Lỗi khi đánh giá mô hình {model_name}: {e}")
            return None, None, None, None, None, None

    def plot_class_accuracies(all_class_accuracies, classes):
        plt.figure(figsize=(12, 6))
        for model_name, class_accuracies in all_class_accuracies.items():
            accuracies = [class_accuracies.get(cls, 0.0) for cls in classes]
            plt.plot(classes, accuracies, marker='o', label=model_name)
        
        plt.title("Độ chính xác theo từng lớp ký tự")
        plt.xlabel("Ký tự")
        plt.ylabel("Độ chính xác")
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('class_accuracies.png')
        plt.close()

    def plot_metrics(metrics, metric_name, output_path):
        plt.figure(figsize=(8, 6))
        models = list(metrics.keys())
        values = list(metrics.values())
        
        plt.bar(models, values)
        plt.title(f'So sánh {metric_name} giữa các mô hình')
        plt.ylabel(metric_name)
        plt.grid(True, axis='y')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    def plot_training_history(model, model_name, X_test, y_true):
        # Đánh giá trên tập test để lấy loss và accuracy
        y_true_cat = tf.keras.utils.to_categorical(y_true, num_classes=len(classes))
        loss, accuracy = model.evaluate(X_test, y_true_cat, verbose=0)
        
        plt.figure(figsize=(8, 6))
        plt.bar(['Loss', 'Accuracy'], [loss, accuracy])
        plt.title(f'Test Loss và Accuracy - {model_name}')
        plt.ylabel('Giá trị')
        plt.grid(True, axis='y')
        plt.tight_layout()
        plt.savefig(f'test_metrics_{model_name}.png')
        plt.close()

    def save_results(all_class_accuracies, metrics, classes):
        data = {"Class": classes}
        for model_name in all_class_accuracies:
            data[f"{model_name}_Accuracy"] = [all_class_accuracies[model_name].get(cls, 0.0) for cls in classes]
        
        df = pd.DataFrame(data)
        df.loc[len(df)] = ["Overall"] + [metrics[model_name]['accuracy'] for model_name in all_class_accuracies]
        df.to_csv('evaluation_results.csv', index=False)
        print("Đã lưu kết quả vào evaluation_results.csv")

    if not os.path.exists(DATA_DIR):
        print(f"Lỗi: Thư mục dữ liệu {DATA_DIR} không tồn tại!")
        return

    X_test, y_true, img_paths = load_test_dataset(DATA_DIR, classes)
    if X_test is None:
        return

    X_test = X_test.reshape(-1, IMG_HEIGHT, IMG_WIDTH, 1)

    all_class_accuracies = {}
    metrics = {}

    for model_name, model_path in MODEL_PATHS.items():
        print(f"\nĐánh giá mô hình: {model_name}")
        
        if not os.path.exists(model_path):
            print(f"Lỗi: Không tìm thấy file mô hình {model_path}")
            continue
        
        try:
            model = load_model(model_path, compile=False)
            model.compile(loss='categorical_crossentropy',
                         optimizer='adam',
                         metrics=['accuracy'])
            print(f"Đã tải mô hình từ {model_path}")
        except Exception as e:
            print(f"Lỗi khi tải mô hình {model_name}: {e}")
            continue
        
        y_pred, overall_accuracy, class_accuracies, precision, recall, f1 = evaluate_model(
            model, model_name, X_test, y_true, classes)
        if y_pred is None:
            continue
        
        all_class_accuracies[model_name] = class_accuracies
        metrics[model_name] = {
            'accuracy': overall_accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        print(f"\nĐộ chính xác theo từng lớp cho {model_name}:")
        for class_name, accuracy in class_accuracies.items():
            display_char = vietnamese_map.get(class_name, class_name)
            print(f"{display_char}: {accuracy:.4f}")
        
        plot_training_history(model, model_name, X_test, y_true)

    if all_class_accuracies:
        plot_class_accuracies(all_class_accuracies, classes)
        plot_metrics({k: v['precision'] for k, v in metrics.items()}, 'Precision', 'precision_comparison.png')
        plot_metrics({k: v['recall'] for k, v in metrics.items()}, 'Recall', 'recall_comparison.png')
        plot_metrics({k: v['f1'] for k, v in metrics.items()}, 'F1-score', 'f1_comparison.png')
        save_results(all_class_accuracies, metrics, classes)

if __name__ == "__main__":
    print("Bắt đầu đánh giá các mô hình trên bộ dữ liệu kiểm tra...")
    evaluate_models_on_dataset()