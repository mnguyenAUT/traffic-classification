import tkinter as tk
from tkinter import filedialog, Canvas, Text
import cv2
from PIL import Image, ImageTk
import threading
import time
import torch
from ultralytics import YOLO
import numpy as np

class VideoPlayer:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Player with YOLOv8 Object Detection")
        
        self.left_canvas = Canvas(root, width=800, height=600)
        self.left_canvas.pack(side=tk.LEFT)
        
        self.right_canvas = Canvas(root, width=800, height=600, bg='black')
        self.right_canvas.pack(side=tk.RIGHT)
        
        self.btn_frame = tk.Frame(root)
        self.btn_frame.pack(fill=tk.X)

        self.btn_browse = tk.Button(self.btn_frame, text="Browse Video", command=self.load_video)
        self.btn_browse.pack(side=tk.LEFT)

        self.btn_detect = tk.Button(self.btn_frame, text="YOLO v8 Object Detection", command=self.detect_objects)
        self.btn_detect.pack(side=tk.LEFT)

        # Add the text areas below the buttons
        self.text_area = Text(root, height=10, width=50)
        self.text_area.pack(fill=tk.X, padx=10, pady=5)
        
        self.traffic_area = Text(root, height=10, width=50)
        self.traffic_area.pack(fill=tk.X, padx=10, pady=5)

        self.cap = None
        self.frame = None
        self.running = False
        self.yolo = YOLO('yolov8l.pt')  # Load YOLOv8 model
        self.lock = threading.Lock()
        
        # Define the classes to be detected
        self.allowed_classes = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorbike', 5: 'bus', 7: 'truck'}
        self.black_frame = None

    def load_video(self):
        video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.*")])
        if video_path:
            self.cap = cv2.VideoCapture(video_path)
            self.running = True
            self.play_video()

    def play_video(self):
        if self.cap.isOpened():
            with self.lock:
                ret, frame = self.cap.read()
                if ret:
                    self.frame = frame
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_rgb = self.resize_frame(frame_rgb, 800, 600)
                    img = Image.fromarray(frame_rgb)
                    imgtk = ImageTk.PhotoImage(image=img)
                    self.left_canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
                    self.left_canvas.imgtk = imgtk

                    if self.black_frame is None:
                        self.black_frame = np.zeros_like(frame_rgb)

                    black_frame_with_boxes = self.draw_white_boxes(self.black_frame, [])
                    imgtk_black = ImageTk.PhotoImage(image=black_frame_with_boxes)
                    self.right_canvas.create_image(0, 0, anchor=tk.NW, image=imgtk_black)
                    self.right_canvas.imgtk = imgtk_black

            if self.running:
                self.root.after(10, self.play_video)

    def resize_frame(self, frame, width, height):
        return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

    def detect_objects(self):
        if not self.running or self.frame is None:
            return

        def detection_thread():
            while self.running:
                with self.lock:
                    ret, frame = self.cap.read()
                if ret:
                    results = self.yolo(frame)
                    frame_with_boxes = self.draw_boxes(frame, results)
                    frame_with_boxes = cv2.cvtColor(frame_with_boxes, cv2.COLOR_BGR2RGB)
                    frame_with_boxes = self.resize_frame(frame_with_boxes, 800, 600)
                    img_with_boxes = Image.fromarray(frame_with_boxes)
                    imgtk_with_boxes = ImageTk.PhotoImage(image=img_with_boxes)
                    self.left_canvas.create_image(0, 0, anchor=tk.NW, image=imgtk_with_boxes)
                    self.left_canvas.imgtk = imgtk_with_boxes

                    self.black_frame = self.draw_white_boxes(self.black_frame, results)
                    black_frame_with_boxes = cv2.cvtColor(self.black_frame, cv2.COLOR_BGR2RGB)
                    black_frame_with_boxes = self.resize_frame(black_frame_with_boxes, 800, 600)
                    img_black_with_boxes = Image.fromarray(black_frame_with_boxes)
                    imgtk_black_with_boxes = ImageTk.PhotoImage(image=img_black_with_boxes)
                    self.right_canvas.create_image(0, 0, anchor=tk.NW, image=imgtk_black_with_boxes)
                    self.right_canvas.imgtk = imgtk_black_with_boxes

                    # Update the text area with the percentage of objects
                    percentageOfObj = self.calculate_percentage_of_obj(self.black_frame)
                    self.update_text_area(percentageOfObj)
                    
                time.sleep(0.01)

        threading.Thread(target=detection_thread).start()

    def draw_boxes(self, frame, results):
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                if cls_id in self.allowed_classes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    label = self.allowed_classes[cls_id]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        return frame    
    
    def draw_white_boxes(self, black_frame, results):
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                if cls_id in self.allowed_classes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    x1, y1 = self.resize_coords(x1, y1, black_frame.shape[1], black_frame.shape[0])
                    x2, y2 = self.resize_coords(x2, y2, black_frame.shape[1], black_frame.shape[0])
                    cv2.rectangle(black_frame, (x1, int((y1+y2)/2)), (x2, y2), (255, 255, 255), -1)
        # Create a mask where the frame is greater than 0
        mask = black_frame > 0        
        
        black_frame[mask] = np.log1p(black_frame[mask]) / np.log(1.05)        
        #black_frame = cv2.applyColorMap(black_frame, cv2.COLORMAP_JET)
        #black_frame = cv2.cvtColor(black_frame, cv2.COLOR_BGR2GRAY)

        return black_frame

    def resize_coords(self, x, y, width, height):
        orig_width, orig_height = self.frame.shape[1], self.frame.shape[0]
        x = int(x * width / orig_width)
        y = int(y * height / orig_height)
        return x, y

    def calculate_percentage_of_obj(self, black_frame):
        non_zero_pixels = black_frame[black_frame > 0]
        car_pixels = black_frame[black_frame > 100]
        if non_zero_pixels.size == 0:
            return 0
        percentageOfObj = 100 * cv2.countNonZero(car_pixels) / cv2.countNonZero(non_zero_pixels)
        return percentageOfObj

    def update_text_area(self, percentageOfObj):
        self.text_area.delete('1.0', tk.END)
        self.text_area.insert(tk.END, f"Percentage of OBJ on Road: {percentageOfObj:.2f}%\n")
        self.traffic_area.delete('1.0', tk.END)
        traffic_description = self.get_traffic_description(percentageOfObj)
        self.traffic_area.insert(tk.END, f"{traffic_description}\n")

    def get_traffic_description(self, percentageOfObj):
        if percentageOfObj <= 1:
            return "Empty: \nNo vehicles are present, \nthe road is completely clear."
        elif percentageOfObj <= 5:
            return "Very Light Traffic: \nVery few vehicles, \nwith plenty of open space between them."
        elif percentageOfObj <= 10:
            return "Light Traffic: \nA few vehicles on the road, \nbut no delays or slowdowns."
        elif percentageOfObj <= 15:
            return "Moderate Traffic: \nA steady flow of vehicles, \nbut still moving smoothly."
        elif percentageOfObj <= 20:
            return "Busy: \nIncreased number of vehicles, \nwith some minor slowdowns and occasional stops."
        elif percentageOfObj <= 25:
            return "Very Busy: \nHigh volume of vehicles, \nfrequent stops, and slower overall speed."
        else:
            return "Congested: \nVehicles are moving very slowly, \nwith regular stopping and starting."

    def on_closing(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    player = VideoPlayer(root)
    root.protocol("WM_DELETE_WINDOW", player.on_closing)
    root.mainloop()
