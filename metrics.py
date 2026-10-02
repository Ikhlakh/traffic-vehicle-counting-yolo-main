import pandas as pd
from ultralytics import YOLO

# ==========================================
# 1. CẤU HÌNH ĐƯỜNG DẪN
# ==========================================
# Đường dẫn tới trọng số tốt nhất sau khi train
model_path = r"D:\Stuffs\ThiGiacMayTinh\DoAnTGMT\runs\train\yolo26_traffic\weights\best.pt"

# ĐƯỜNG DẪN QUAN TRỌNG: Trỏ tới file data.yaml của bạn
data_path = r"D:\Stuffs\ThiGiacMayTinh\DoAnTGMT\config\data.yaml" 

def generate_class_metrics():
    print("⏳ Đang tải mô hình và quét lại tập Validation để lấy thông số chi tiết...")
    
    # Load model
    model = YOLO(model_path)

    # Chạy validation (lưu ý: split='val' là tập mặc định)
    # Tắt verbose để terminal không bị rác
    metrics = model.val(data=data_path, split='val', verbose=False)

    # ==========================================
    # 2. BÓC TÁCH DỮ LIỆU TỪ METRICS OBJECT
    # ==========================================
    # Lấy mảng ID các class và tên class (Bike, Bus, Car, Motorbike, Truck)
    ap_classes = metrics.box.ap_class_index
    names = model.names

    # Lấy các mảng thông số (Precision, Recall, mAP50)
    precision = metrics.box.p
    recall = metrics.box.r
    ap50 = metrics.box.ap50

    # Tính toán F1-Score: 2 * (P * R) / (P + R)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-16)

    # Lấy số lượng instances (Số lượng nhãn thực tế) từ Confusion Matrix
    # Trục row = True, Trục col = Predicted. Bỏ dòng cuối cùng (Background)
    cm_matrix = metrics.confusion_matrix.matrix
    instances = cm_matrix.sum(axis=1)[:-1]

    data = []
    total_instances = 0

    # ==========================================
    # 3. GOM DỮ LIỆU TỪNG LỚP
    # ==========================================
    for i, class_idx in enumerate(ap_classes):
        class_name = names[class_idx]
        p = precision[i]
        r = recall[i]
        f1 = f1_scores[i]
        ap = ap50[i]
        inst = int(instances[class_idx])
        total_instances += inst
        
        data.append([class_name, round(p, 4), round(r, 4), round(f1, 4), round(ap, 4), inst])

    # Thêm dòng "Trung bình (mAP)" ở cuối cùng giống ảnh mẫu
    mean_p = precision.mean()
    mean_r = recall.mean()
    mean_f1 = f1_scores.mean()
    mean_ap50 = ap50.mean()

    data.append([
        "Trung bình (mAP)", 
        round(mean_p, 4), 
        round(mean_r, 4), 
        round(mean_f1, 4), 
        round(mean_ap50, 4), 
        int(total_instances)
    ])

    # ==========================================
    # 4. TẠO DATAFRAME VÀ IN RA BẢNG
    # ==========================================
    columns = ["Lớp đối tượng", "Precision", "Recall", "F1-Score", "AP@50", "Số instances"]
    df = pd.DataFrame(data, columns=columns)

    print("\n" + "="*80)
    print(" BẢNG THỐNG KÊ ĐÁNH GIÁ MÔ HÌNH (PER-CLASS METRICS)")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)

    # Xuất ra file CSV để copy vào Word/Excel làm báo cáo
    output_csv = r"D:\Stuffs\ThiGiacMayTinh\DoAnTGMT\runs\train\yolo26_traffic\class_metrics_report.csv"
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"\n✅ Đã lưu file báo cáo chi tiết tại: {output_csv}")

if __name__ == "__main__":
    generate_class_metrics()