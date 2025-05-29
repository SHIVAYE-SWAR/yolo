import sys
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
from ultralytics import YOLO
import torch
import time
from collections import defaultdict

class SpeedDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Speed Detection System")
        self.root.geometry("1400x900")

        # Initialize variables
        self.cap = None
        self.current_frame = None
        self.is_playing = False
        self.is_paused = False
        self.frame_count = 0
        self.current_frame_index = 0
        self.pixels_per_meter = 30

        # Car tracking variables
        self.car_tracks = defaultdict(list)
        self.car_speeds = defaultdict(list)
        self.last_positions = {}
        self.track_counter = 0

        # Load YOLO model
        self.model = YOLO('yolov8n.pt')

        # Initialize UI
        self.init_ui()

    def init_ui(self):
        self.main_container = tk.Frame(self.root)
        self.main_container.pack(fill="both", expand=True, padx=10, pady=10)

        self.video_frame = tk.Frame(self.main_container)
        self.video_frame.pack(side="left", fill="both", expand=True, padx=5, pady=5)

        self.video_label = tk.Label(self.video_frame, text="")
        self.video_label.pack(fill="both", expand=True, padx=5, pady=5)

        self.control_frame = tk.Frame(self.main_container, width=300)
        self.control_frame.pack(side="right", fill="y", padx=5, pady=5)

        title_label = tk.Label(self.control_frame, text="Speed Detection System", font=("Helvetica", 16, "bold"))
        title_label.pack(pady=10)

        calibration_frame = tk.Frame(self.control_frame)
        calibration_frame.pack(fill="x", padx=10, pady=5)

        tk.Label(calibration_frame, text="Calibration", font=("Helvetica", 10, "bold")).pack(pady=5)
        tk.Label(calibration_frame, text="Pixels per meter:").pack()
        self.pixels_entry = tk.Entry(calibration_frame)
        self.pixels_entry.insert(0, str(self.pixels_per_meter))
        self.pixels_entry.pack(pady=5)
        tk.Button(calibration_frame, text="Update Calibration", command=self.update_calibration).pack(pady=5)

        control_buttons_frame = tk.Frame(self.control_frame)
        control_buttons_frame.pack(fill="x", padx=10, pady=10)

        tk.Label(control_buttons_frame, text="Controls", font=("Helvetica", 10, "bold")).pack(pady=5)
        self.open_video_btn = tk.Button(control_buttons_frame, text="Open Video", command=self.open_video)
        self.open_video_btn.pack(fill="x", pady=5)

        self.webcam_btn = tk.Button(control_buttons_frame, text="Start Webcam", command=self.start_webcam)
        self.webcam_btn.pack(fill="x", pady=5)

        self.stop_btn = tk.Button(control_buttons_frame, text="Stop / Resume", command=self.toggle_video)
        self.stop_btn.pack(fill="x", pady=5)



        speed_frame = tk.Frame(self.control_frame)
        speed_frame.pack(fill="x", padx=10, pady=10)
        tk.Label(speed_frame, text="Detected Speeds", font=("Helvetica", 10, "bold")).pack(pady=5)
        self.speed_label = tk.Label(speed_frame, text="No cars detected", wraplength=280)
        self.speed_label.pack(fill="x", pady=5)

        self.status_label = tk.Label(self.root, text="Ready", anchor="w", height=2)
        self.status_label.pack(fill="x", side="bottom", padx=10)

    def update_calibration(self):
        try:
            new_value = float(self.pixels_entry.get())
            if new_value > 0:
                self.pixels_per_meter = new_value
                self.status_label.configure(text=f"Calibration updated: {new_value} pixels/meter")
            else:
                messagebox.showerror("Error", "Value must be greater than 0")
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number")

    def update_status(self, message):
        self.status_label.configure(text=message)

    def get_track_id(self, center_x, center_y):
        for track_id, positions in self.last_positions.items():
            last_x, last_y = positions
            distance = np.sqrt((center_x - last_x)**2 + (center_y - last_y)**2)
            if distance < 50:
                return track_id

        self.track_counter += 1
        return f"car_{self.track_counter}"

    def calculate_speed(self, track_id, current_pos):
        if track_id not in self.last_positions:
            self.last_positions[track_id] = current_pos
            return 0

        last_pos = self.last_positions[track_id]
        distance_pixels = np.sqrt((current_pos[0] - last_pos[0])**2 + (current_pos[1] - last_pos[1])**2)
        time_diff = 1/30
        distance_meters = distance_pixels / self.pixels_per_meter
        speed = distance_meters / time_diff if time_diff > 0 else 0
        speed_kmh = speed * 3.6
        self.last_positions[track_id] = current_pos
        return speed_kmh

    def update_frame(self):
        if self.cap is not None and self.is_playing:
            ret, frame = self.cap.read()
            if ret:
                self.current_frame_index = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                self.update_status(f"Frame: {self.current_frame_index}/{self.frame_count}")
                self.show_frame(frame)
                if self.is_playing:
                    self.root.after(30, self.update_frame)
            else:
                self.is_playing = False
                self.update_status("Video finished")

    def show_frame(self, frame):
        results = self.model(frame, conf=0.5)
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                cls = box.cls[0]
                if int(cls) == 2:
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    track_id = self.get_track_id(center_x, center_y)
                    speed = self.calculate_speed(track_id, (center_x, center_y))
                    self.car_speeds[track_id].append(speed)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    speed_text = f"Car {track_id}: {speed:.1f} km/h"
                    cv2.putText(frame, speed_text, (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if self.car_speeds:
            speed_text = ""
            for track_id, speeds in self.car_speeds.items():
                if speeds:
                    recent_speeds = speeds[-3:]
                    avg_speed = sum(recent_speeds) / len(recent_speeds)
                    speed_text += f"Car {track_id}: {avg_speed:.1f} km/h\n"
            self.speed_label.configure(text=speed_text)
        else:
            self.speed_label.configure(text="No cars detected")

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (800, 600))
        image = Image.fromarray(frame)
        photo = ImageTk.PhotoImage(image=image)
        self.video_label.configure(image=photo)
        self.video_label.image = photo


    def open_video(self):
        file_name = filedialog.askopenfilename(
            title="Open Video File",
            filetypes=[("Video Files", "*.mp4 *.avi *.mkv")]
        )
        if file_name:
            self.cap = cv2.VideoCapture(file_name)
            self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.current_frame_index = 0
            self.is_playing = True
            self.is_paused = False
            self.car_tracks.clear()
            self.car_speeds.clear()
            self.last_positions.clear()
            self.track_counter = 0
            self.update_status(f"Playing: {file_name}")
            self.update_frame()

    def start_webcam(self):
        self.cap = cv2.VideoCapture(0)
        self.is_playing = True
        self.is_paused = False
        self.car_tracks.clear()
        self.car_speeds.clear()
        self.last_positions.clear()
        self.track_counter = 0
        self.update_status("Webcam active")
        self.update_frame()

    def toggle_video(self):
        if self.cap is not None:
            self.is_playing = not self.is_playing
            if self.is_playing:
                self.update_status("Resumed")
                self.update_frame()
            else:
                self.update_status("Paused")

    def on_closing(self):
        if self.cap is not None:
            self.cap.release()
        self.root.destroy()

if __name__ == '__main__':
    root = tk.Tk()
    app = SpeedDetectionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
