import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # You can change to 'yolov8s.pt' for better accuracy

# Initialize GUI
root = tk.Tk()
root.title("Pedestrian Detection (YOLOv8)")
root.geometry("800x650")

# People Count Label
count_label = tk.Label(root, text="People Count: 0", font=("Arial", 16), bg="black", fg="white")
count_label.pack(fill=tk.X)

# Image/Video Display Panel
panel = tk.Label(root)
panel.pack()

# Global variable for live camera
cap = None

# Function to detect and display people
def detect_and_display(image):
    results = model(image)
    people_count = 0

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if class_id == 0 and conf > 0.8:  # class 0 = person
                people_count += 1
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, f"Person: {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    count_label.config(text=f"People Count: {people_count}")

    # Convert and show in Tkinter
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)
    image_tk = ImageTk.PhotoImage(image_pil)

    panel.config(image=image_tk)
    panel.image = image_tk

# Load image from file
def load_image():
    file_path = filedialog.askopenfilename()
    if not file_path:
        return
    image = cv2.imread(file_path)
    detect_and_display(image)

# Take a photo using webcam
def take_photo():
    temp_cap = cv2.VideoCapture(0)
    if not temp_cap.isOpened():
        print("Camera couldn't open.")
        return

    while True:
        ret, frame = temp_cap.read()
        if not ret:
            break
        cv2.imshow("Press 's' to Capture", frame)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            detect_and_display(frame)
            break

    temp_cap.release()
    cv2.destroyAllWindows()

# Live Detection using Tkinter loop
def live_detection():
    global cap
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera couldn't open.")
        return
    update_frame()

# Frame update loop
def update_frame():
    global cap
    if cap:
        ret, frame = cap.read()
        if ret:
            detect_and_display(frame)
        root.after(30, update_frame)

# Cleanup on exit
def on_closing():
    global cap
    if cap and cap.isOpened():
        cap.release()
    root.destroy()

# Buttons
btn_load = tk.Button(root, text="Select Image", command=load_image, font=("Arial", 14), bg="#2196F3", fg="white")
btn_load.pack(pady=10)

btn_camera = tk.Button(root, text="Take Photo with Camera", command=take_photo, font=("Arial", 14), bg="#4CAF50", fg="white")
btn_camera.pack(pady=5)

btn_live = tk.Button(root, text="Live Detection", command=live_detection, font=("Arial", 14), bg="#FF9800", fg="white")
btn_live.pack(pady=5)

# Bind close event
root.protocol("WM_DELETE_WINDOW", on_closing)

# Start GUI loop
root.mainloop()
