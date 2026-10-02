from ultralytics import YOLO
import os
import shutil

def main():
    # 1. Khởi tạo mô hình YOLO (Baseline)
    model = YOLO('yolo26s.pt') 

    # 2. Đường dẫn tuyệt đối đến file cấu hình dữ liệu
    data_yaml_path = os.path.abspath("config/data.yaml")

    print("Bắt đầu tiến trình huấn luyện mô hình Baseline (Không Augmentation)...")

    # 3. Chạy huấn luyện
    results = model.train(
        data=data_yaml_path,
        epochs=300,              
        patience=50,             
        batch=16,                
        imgsz=640,               
        device=0,                
        workers=8,               
        project='runs/train',    
        name='yolo26_baseline',  # Đổi tên phiên bản huấn luyện thành baseline
        
        # ==============================================================
        # TẮT TOÀN BỘ AUGMENTATION (BASELINE MẶC ĐỊNH)
        # Bằng cách ép các thông số biến đổi về 0.0
        # ==============================================================
        hsv_h=0.0,               # Không đổi màu (Hue)
        hsv_s=0.0,               # Không đổi độ bão hòa (Saturation)
        hsv_v=0.0,               # Không đổi độ sáng (Value)
        degrees=0.0,             # Không xoay ảnh
        translate=0.0,           # Không dịch chuyển
        scale=0.0,               # Không thu phóng
        perspective=0.0,         # Không thay đổi phối cảnh
        flipud=0.0,              # Không lật dọc
        fliplr=0.0,              # Không lật ngang
        mosaic=0.0,              # Tắt tính năng ghép 4 ảnh (Mosaic)
        mixup=0.0,               # Tắt tính năng trộn ảnh (MixUp)
        copy_paste=0.0,          # Tắt copy-paste vật thể
        erasing=0.0,             # Tắt xóa ngẫu nhiên (Random Erasing)
        crop_fraction=1.0        # Giữ nguyên toàn bộ ảnh, không crop
        # ==============================================================
    )
    
    print("\nHuấn luyện baseline hoàn tất!")

    # 4. TỰ ĐỘNG COPY VÀ ĐỔI TÊN THÀNH 'best_baseline.pt'
    # Do 'name' ở trên là 'yolo26_baseline', đường dẫn sinh ra sẽ thay đổi tương ứng
    source_path = "runs/train/yolo26_baseline/weights/best.pt"
    dest_dir = "models"
    dest_path = os.path.join(dest_dir, "best_baseline.pt") # Lưu tên file theo yêu cầu

    # Kiểm tra và tiến hành copy file
    if os.path.exists(source_path):
        os.makedirs(dest_dir, exist_ok=True)
        shutil.copy(source_path, dest_path)
        print(f"[Hệ thống] Đã tự động chuyển và đổi tên trọng số thành: {dest_path}")
    else:
        print(f"[Cảnh báo] Không tìm thấy file {source_path}. Vui lòng kiểm tra lại log huấn luyện.")

if __name__ == '__main__':
    main()