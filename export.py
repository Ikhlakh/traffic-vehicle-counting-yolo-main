# export.py
from ultralytics import YOLO
import os

def export_model():
    model_path = "models/best.pt"
    if not os.path.exists(model_path):
        print("Lỗi: Không tìm thấy model.")
        return

    model = YOLO(model_path)
    print("Đang biên dịch sang định dạng ONNX...")
    
    # Đổi format="onnx", half=True vẫn giữ để tối ưu card RTX 2060 SUPER
    model.export(
        format="onnx", 
        imgsz=640,       
        half=True,       
        device=0,
        dynamic=False    # Khóa cứng kích thước để chạy nhanh nhất
    )
    print("\nHoàn tất! File best.onnx đã được tạo. Hãy chuyển nó vào thư mục weights/")

if __name__ == "__main__":
    export_model()