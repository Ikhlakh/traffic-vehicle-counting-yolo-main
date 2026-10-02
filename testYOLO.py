from ultralytics import YOLO
import os

def test_model_on_video():
    # 1. Đường dẫn đến file trọng số tốt nhất bạn vừa train xong
    model_path = "models/best.pt"
    
    # Kiểm tra xem file có tồn tại không
    if not os.path.exists(model_path):
        print(f"Lỗi: Không tìm thấy {model_path}. Hãy kiểm tra lại đường dẫn!")
        return

    # Load mô hình
    model = YOLO(model_path)
    
    print("Đang khởi động tiến trình nhận diện trên Video 4K...")

    # 2. Chạy thử trên video 4K
    # Thay 'duong_dan_video_4k_cua_ban.mp4' bằng tên file video thực tế của bạn
    results = model.predict(
        source="GiaoThong.mp4", 
        
        # Cấu hình hiển thị và lưu trữ
        show=True,            # Mở cửa sổ xem trực tiếp (Có thể cửa sổ sẽ rất to vì video 4K)
        save=True,            # BẮT BUỘC: Lưu lại video kết quả để nộp minh chứng cho giảng viên
        project="runs/detect",
        name="test_4k_video",
        
        # Cấu hình tối ưu GPU và độ chính xác
        imgsz=640,            # Bắt buộc giữ 640 để không tràn VRAM
        conf=0.4,             # Ngưỡng tin cậy: Chỉ vẽ khung cho các xe có độ chắc chắn > 40%
        iou=0.45,             # Ngưỡng IOU: Tránh vẽ chồng chéo 2 khung hình lên cùng 1 chiếc xe
        
        # Cấu hình độ mượt (chỉ dùng cho hiển thị/lưu video)
        line_width=3          # Độ dày của viền BBox (Video 4K thì nên để số to để dễ nhìn)
    )
    
    print("\nHoàn tất! Video kết quả đã được lưu tại thư mục: runs/detect/test_4k_video/")

if __name__ == "__main__":
    test_model_on_video()