import os
import glob
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay
import pickle
import tensorflow as tf
from tqdm import tqdm

# Định nghĩa các lớp TensorFlow (từ code gốc)
class LinearSVM:
    def __init__(self, input_dim, num_classes, C=1.0):
        self.W = tf.Variable(tf.zeros([input_dim, num_classes], dtype=tf.float32), name='weights')
        self.b = tf.Variable(tf.zeros([num_classes], dtype=tf.float32), name='bias')
        self.C = C

    def predict(self, X):
        scores = tf.matmul(X, self.W) + self.b
        return tf.argmax(scores, axis=1)

class LogisticRegressionTF:
    def __init__(self, input_dim, num_classes, C=1.0):
        self.W = tf.Variable(tf.zeros([input_dim, num_classes], dtype=tf.float32), name='weights')
        self.b = tf.Variable(tf.zeros([num_classes], dtype=tf.float32), name='bias')
        self.C = C

    def predict(self, X):
        logits = tf.matmul(X, self.W) + self.b
        return tf.argmax(logits, axis=1)

# Hàm tiền xử lý dữ liệu
def load_images_from_folder(folder, img_size=(50, 150)):
    """Tiền xử lý và tải ảnh từ thư mục."""
    X = []
    y = []
    class_names = sorted([name for name in os.listdir(folder) if os.path.isdir(os.path.join(folder, name))])
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_names)}

    for cls_name in tqdm(class_names, desc="Đang tải ảnh theo lớp"):
        cls_path = os.path.join(folder, cls_name)
        image_paths = glob.glob(os.path.join(cls_path, '*.jpg'))

        for img_path in image_paths:
            try:
                img = Image.open(img_path).resize(img_size).convert("L")  # Chuyển sang grayscale
                img_np = np.array(img).flatten() / 255.0  # Chuẩn hóa
                X.append(img_np)
                y.append(class_to_idx[cls_name])
            except Exception as e:
                print(f"Lỗi khi xử lý {img_path}: {e}")

    return np.array(X), np.array(y), class_to_idx, class_names

def normalize_data(X):
    """Chuẩn hóa dữ liệu."""
    mean = np.mean(X, axis=0, keepdims=True)
    std = np.std(X, axis=0, keepdims=True) + 1e-6
    return (X - mean) / std

# Hàm trực quan hóa
def plot_confusion_matrix(y_true, y_pred, class_names, title, save_path=None):
    """Vẽ và lưu ma trận nhầm lẫn."""
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    plt.figure(figsize=(10, 8))
    disp.plot(cmap='Blues', xticks_rotation=45)
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def evaluate_model(model, X_test, y_test, model_name, class_names, save_dir="results"):
    """Đánh giá mô hình và lưu kết quả."""
    try:
        y_pred = model.predict(X_test)
    except AttributeError:  # Cho các mô hình TensorFlow
        y_pred = model.predict(X_test).numpy()

    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')

    print(f"\nĐánh giá mô hình {model_name}:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")

    # Lưu ma trận nhầm lẫn
    os.makedirs(save_dir, exist_ok=True)
    cm_save_path = os.path.join(save_dir, f"{model_name.replace(' ', '_')}_cm.png")
    plot_confusion_matrix(y_test, y_pred, class_names, f"Confusion Matrix - {model_name}", cm_save_path)

    return {"Model": model_name, "Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1-score": f1}

# Hàm tải mô hình
def load_models(save_model_dir="./save_model", num_classes=10, input_dim=7500):
    """Tải tất cả các mô hình từ thư mục save_model."""
    models = []
    for file_name in os.listdir(save_model_dir):
        file_path = os.path.join(save_model_dir, file_name)
        if not file_name.endswith('.pkl'):
            continue

        try:
            with open(file_path, 'rb') as f:
                model_data = pickle.load(f)

            # Xử lý mô hình sklearn
            if not isinstance(model_data, dict):
                model_name = file_name.replace('_model.pkl', '').replace('_', ' ').title()
                models.append((model_name, model_data))
                continue

            # Xử lý mô hình TensorFlow
            if 'svm_linear' in file_name:
                model = LinearSVM(input_dim, num_classes)
                model_name = file_name.replace('_weights.pkl', '').replace('_', ' ').title()
                model.W.assign(model_data['W'])
                model.b.assign(model_data['b'])
                models.append((f"SVM Linear {model_name}", model))
            elif 'logistic' in file_name:
                model = LogisticRegressionTF(input_dim, num_classes)
                model_name = file_name.replace('_weights.pkl', '').replace('_', ' ').title()
                model.W.assign(model_data['W'])
                model.b.assign(model_data['b'])
                models.append((f"Logistic Regression {model_name}", model))

        except Exception as e:
            print(f"Lỗi khi tải mô hình {file_name}: {e}")

    return models

# Pipeline chính
def main():
    # Tải dữ liệu test
    print("Tải dữ liệu test...")
    X_test, y_test, class_to_idx, class_names = load_images_from_folder('preprocessed_data/test')
    X_test = normalize_data(X_test)
    X_test_tf = tf.convert_to_tensor(X_test, dtype=tf.float32)
    y_test_tf = tf.convert_to_tensor(y_test, dtype=tf.int32)

    # Tải tất cả các mô hình
    print("Tải các mô hình từ ./save_model...")
    models = load_models(num_classes=len(class_to_idx), input_dim=X_test.shape[1])

    # Đánh giá và lưu kết quả
    results = []
    for model_name, model in models:
        print(f"\nĐánh giá {model_name} trên tập test...")
        result = evaluate_model(model, X_test_tf if 'Linear' in model_name or 'Logistic' in model_name else X_test,
                               y_test_tf if 'Linear' in model_name or 'Logistic' in model_name else y_test,
                               model_name, class_names)
        results.append(result)

    # Lưu kết quả vào CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv('results/test_evaluation_results.csv', index=False, encoding='utf-8')
    print("\nKết quả đã được lưu vào 'results/test_evaluation_results.csv'")

    # Vẽ biểu đồ so sánh
    plt.figure(figsize=(12, 6))
    plt.bar(results_df['Model'], results_df['Accuracy'], label='Accuracy')
    plt.xticks(rotation=90)
    plt.title('So sánh Accuracy của các mô hình trên tập test')
    plt.ylabel('Accuracy')
    plt.tight_layout()
    plt.savefig('results/accuracy_comparison_test.png')
    plt.close()

if __name__ == "__main__":
    main()