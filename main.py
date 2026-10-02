import cv2
import yaml
import tkinter as tk
from tkinter import filedialog, messagebox
from datetime import datetime
import time # Dùng để tính FPS

# Import Module
from src import FrameState, YoloDetector, ByteTrackerWrapper, PolygonCounter, UIVisualizer, SpeedEstimator
from utils import InteractiveROIDrawer, VideoStream, CSVReporter
from ui import TrafficDashboard

class TrafficApp:
    def __init__(self, root):
        self.root = root
        
        with open("config/settings.yaml", 'r', encoding='utf-8') as f:
            self.cfg = yaml.safe_load(f)
            
        # Nối thêm hàm on_redraw vào Dashboard
        self.dashboard = TrafficDashboard(self.root, self.on_start, self.on_pause, self.on_browse, self.on_redraw)
        
        self.stream = None
        self.ai_modules = {}
        self.frame_generator = None
        self.start_time = None
        self.is_playing = False
        self.video_path = None
        
        self.total_in = 0
        self.total_out = 0
        
        # Biến phục vụ tính FPS và Vẽ lại ROI
        self.last_raw_frame = None 
        self.current_fps = 0.0
        self.prev_frame_time = time.time()

    def on_browse(self):
        file_path = filedialog.askopenfilename(
            title="Chọn video giao thông",
            filetypes=[("Video files", "*.mp4 *.avi *.mov")]
        )
        if not file_path: return 
            
        self.video_path = file_path
        self.dashboard.update_video_path(file_path) 
        
        self.total_in = 0
        self.total_out = 0
        self.current_fps = 0.0
        
        if self.stream: self.stream.release() 
        self.stream = VideoStream(self.video_path, self.cfg['io']['output_video'])
        
        first_frame = self.stream.get_first_frame()
        self.last_raw_frame = first_frame # Lưu lại frame đầu để dự phòng
        
        if first_frame is None:
            messagebox.showerror("Lỗi", "Không thể đọc video được chọn!")
            return

        drawer = InteractiveROIDrawer("Ve Vung ROI (Nhan ENTER de hoan tat)")
        roi_points = drawer.draw(first_frame)
        
        if len(roi_points) < 3:
            roi_points = self.cfg.get('roi_polygon', [])
        else:
            roi_points_list = [list(point) for point in roi_points]
            self.cfg['roi_polygon'] = roi_points_list
            with open("config/settings.yaml", 'w', encoding='utf-8') as f:
                yaml.dump(self.cfg, f, default_flow_style=False)
        
        print("[Hệ thống] Đang tải AI Models... Vui lòng đợi.")
        self.ai_modules['detector'] = YoloDetector(self.cfg['model']['path'], self.cfg['model']['conf_threshold'], self.cfg['model']['imgsz'])
        self.ai_modules['tracker'] = ByteTrackerWrapper(self.cfg['classes']['names'], self.cfg['tracking']['track_thresh'], self.cfg['tracking']['match_thresh'])
        self.ai_modules['counter'] = PolygonCounter(roi_points)
        self.ai_modules['speed_estimator'] = SpeedEstimator(roi_points, fps=self.stream.fps, real_length_meters=20.0)
        self.ai_modules['visualizer'] = UIVisualizer(roi_points)
        
        if 'reporter' in self.ai_modules:
            self.ai_modules['reporter'].close()
        self.ai_modules['reporter'] = CSVReporter(self.cfg['io']['output_csv'])
        
        self.frame_generator = self.stream.read()
        print("[Hệ thống] Đã sẵn sàng! Hãy bấm nút Start Detection.")

    # --- BƯỚC 3: HÀM XỬ LÝ VẼ LẠI ROI KHI ĐANG CHẠY ---
    def on_redraw(self):
        if not self.video_path or self.last_raw_frame is None:
            messagebox.showwarning("Cảnh báo", "Bạn phải chọn video trước khi vẽ lại ROI!")
            return
            
        # Tạm dừng video nếu đang chạy
        was_playing = self.is_playing
        self.is_playing = False 
        
        # Mở công cụ vẽ với đúng khung hình hiện tại
        drawer = InteractiveROIDrawer("Ve lai vach ke (Nhan ENTER de hoan tat)")
        roi_points = drawer.draw(self.last_raw_frame)
        
        if len(roi_points) >= 3:
            # Lưu cấu hình
            roi_points_list = [list(point) for point in roi_points]
            self.cfg['roi_polygon'] = roi_points_list
            with open("config/settings.yaml", 'w', encoding='utf-8') as f:
                yaml.dump(self.cfg, f, default_flow_style=False)
                
            # Cập nhật nóng vào module AI mà không cần load lại mô hình YOLO
            self.ai_modules['counter'] = PolygonCounter(roi_points)
            self.ai_modules['visualizer'] = UIVisualizer(roi_points)
            print("[Hệ thống] Đã cập nhật nóng vạch đếm mới!")
            
        # Tiếp tục phát video nếu trước đó đang phát
        if was_playing:
            self.is_playing = True
            self._process_next_frame()

    def on_start(self):
        if not self.video_path or 'detector' not in self.ai_modules:
            messagebox.showwarning("Cảnh báo", "Vui lòng Browse Video và vẽ ROI trước khi Start!")
            return
            
        if not self.is_playing:
            self.is_playing = True
            if self.start_time is None:
                self.start_time = datetime.now()
            self.prev_frame_time = time.time() # Reset đồng hồ FPS
            self._process_next_frame()

    def on_pause(self):
        self.is_playing = False

    def _process_next_frame(self):
        if not self.is_playing: return 

        try:
            frame_id, raw_frame = next(self.frame_generator)
            self.last_raw_frame = raw_frame # Lưu lại frame để phòng trường hợp vẽ lại ROI
            
            #Frame skipping
            if frame_id % 2 != 0:  # Bỏ qua các frame lẻ để giảm tải
                self.root.after(1, self._process_next_frame)
                return

            # Tính toán FPS (Trung bình động để số không bị giật)
            new_time = time.time()
            time_diff = new_time - self.prev_frame_time
            fps = 1.0 / time_diff if time_diff > 0 else 0
            self.current_fps = (self.current_fps * 0.9) + (fps * 0.1) 
            self.prev_frame_time = new_time

            # --- AI PIPELINE ---
            state = FrameState(frame_id, raw_frame)
            state = self.ai_modules['detector'].process(state)
            state = self.ai_modules['tracker'].process(state)
            state = self.ai_modules['speed_estimator'].process(state)
            state, live_events = self.ai_modules['counter'].process(state)
            
            if live_events:
                self.dashboard.update_logs(live_events) 

                for e in live_events:
                    if e['status'] == 'IN': self.total_in += 1
                    elif e['status'] == 'OUT': self.total_out += 1

                out_events = [e for e in live_events if e['status'] == 'OUT']
                if out_events:
                    self.ai_modules['reporter'].log_events(out_events)

            # --- TÍNH TOÁN THỐNG KÊ (ANALYTICS) ---
            elapsed = (datetime.now() - self.start_time).total_seconds()
            total_veh_counted = sum(state.counts.values()) 
            
            state.analytics = {
                "FPS": int(self.current_fps), # Chèn thêm số FPS
                "Flow (Veh/Min)": round((total_veh_counted / elapsed) * 60, 1) if elapsed > 0 else 0,
                "Car/Bus/Truck %": round((sum([state.counts.get(k,0) for k in ['Car','Bus','Truck']]) / max(total_veh_counted, 1)) * 100, 1),
                "Motorbike %": round((sum([state.counts.get(k,0) for k in ['Motorbike','Bike']]) / max(total_veh_counted, 1)) * 100, 1)
            }
            
            self.dashboard.update_stats(state.counts, state.analytics, self.total_in, self.total_out)

            # --- VẼ ĐỒ HỌA ---
            final_frame = self.ai_modules['visualizer'].draw(state)
            self.dashboard.update_video(final_frame)
            self.stream.write(final_frame)

            self.root.after(1, self._process_next_frame)

        except StopIteration:
            print("[Hệ thống] Đã chạy xong toàn bộ Video!")
            self.stream.release()
            self.ai_modules['reporter'].close()
            self.is_playing = False
            messagebox.showinfo("Hoàn thành", "Video đã được phân tích xong!")

def main():
    root = tk.Tk()
    app = TrafficApp(root)
    
    def on_closing():
        print("[Hệ thống] Đang dọn dẹp và thoát...")
        if app.stream: app.stream.release()
        if 'reporter' in app.ai_modules: app.ai_modules['reporter'].close()
        root.destroy()
        
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()