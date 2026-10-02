from ultralytics import YOLO
import os
import shutil  # Bổ sung thư viện để xử lý việc copy file

def main():
    # 1. Khởi tạo mô hình YOLO26 (Sử dụng bản 'small' để tối ưu tốc độ realtime)
    # Nếu có file weights tải sẵn, bạn sửa thành đường dẫn file (VD: 'yolo26s.pt')
    model = YOLO('yolo26s.pt') 

    # 2. Đường dẫn tuyệt đối đến file cấu hình dữ liệu
    data_yaml_path = os.path.abspath("config/data.yaml")

    print("Bắt đầu tiến trình huấn luyện mô hình...")

    # 3. Chạy huấn luyện với bộ Augmentation chuyên dụng cho camera giao thông
    results = model.train(
        data=data_yaml_path,
        epochs=300,              # Train tối đa 300 vòng
        patience=50,             # EARLY STOPPING: Dừng nếu sau 50 epochs mAP không cải thiện
        batch=16,                # Tối ưu cho VRAM của RTX 2060 SUPER
        imgsz=640,               # Kích thước ảnh chuẩn 
        device=0,                # Chỉ định chạy trên GPU
        workers=8,               # Số luồng CPU hỗ trợ nạp dữ liệu
        project='runs/train',    # Thư mục lưu kết quả
        name='yolo26_traffic',   # Tên phiên bản huấn luyện
        
        # ==============================================================
        # BỘ AUGMENTATION: TỐI ƯU CHO CAMERA NGOÀI TRỜI (OUTDOOR TRAFFIC)
        # ==============================================================
        hsv_h=0.015,             # Hue (Sắc độ): Giả lập quang sai ống kính camera
        hsv_s=0.7,               # Saturation (Độ bão hòa): Giả lập sương mù hoặc nắng gắt cháy sáng
        hsv_v=0.4,               # Value (Độ sáng): Rất quan trọng để mô hình học cả ban ngày lẫn ban đêm
        
        degrees=5.0,             # Xoay ảnh nhẹ: Giả lập cột camera bị rung lắc do gió bão
        translate=0.1,           # Dịch chuyển vị trí: Đảm bảo xe xuất hiện ở rìa ảnh vẫn nhận diện được
        scale=0.5,               # Scale: Cực kỳ quan trọng để bắt được xe ở rất xa (nhỏ) và rất gần (to)
        perspective=0.0001,      # Phối cảnh: Giả lập các góc độ chúi xuống khác nhau của camera
        
        flipud=0.0,              # Lật dọc: Tắt hoàn toàn (xe giao thông không bay lộn ngược)
        fliplr=0.5,              # Lật ngang: Bật 50% (xe có thể chạy từ trái sang phải hoặc ngược lại)
        
        mosaic=1.0,              # Mosaic: Ghép 4 ảnh thành 1 - Tăng cực mạnh khả năng bắt vật thể nhỏ
        mixup=0.1,               # MixUp: Trộn ảnh nhẹ để tăng tính khái quát (generalization)
        # ==============================================================
    )
    
    print("\nHuấn luyện hoàn tất phần lõi YOLO!")

    # 4. TỰ ĐỘNG COPY TRỌNG SỐ VÀO THƯ MỤC 'models/' ĐỂ CHUẨN HÓA CẤU TRÚC
    source_path = "runs/train/yolo26_traffic/weights/best.pt"
    dest_dir = "models"
    dest_path = os.path.join(dest_dir, "best.pt")

    # Kiểm tra xem file best.pt có được sinh ra thành công không
    if os.path.exists(source_path):
        # Tạo thư mục models nếu chưa có
        os.makedirs(dest_dir, exist_ok=True)
        # Copy file
        shutil.copy(source_path, dest_path)
        print(f"[Hệ thống] Đã tự động chuyển trọng số tốt nhất vào: {dest_path}")
        print("[Hệ thống] Bạn có thể dùng file này để chạy inference hoặc export sang TensorRT.")
    else:
        print(f"[Cảnh báo] Không tìm thấy file {source_path}. Vui lòng kiểm tra lại log huấn luyện.")

if __name__ == '__main__':
    # Hỗ trợ đa tiến trình trên Windows an toàn hơn
    main()