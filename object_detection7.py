import cv2
import numpy as np
import time
import pygame
from datetime import datetime

# Initialize pygame mixer for audio
pygame.mixer.init()

# Load the pre-trained model and class labels
net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'mobilenet_iter_73000.caffemodel')
CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
    "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

# Professional design constants
FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
FONT_THICKNESS = 1
LABEL_PADDING = 6
BOX_THICKNESS = 1
OPACITY = 0.7
FOCAL_LENGTH = 1000
KNOWN_WIDTH = 200
CONFIDENCE_THRESHOLD = 0.2
FPS_WINDOW_SIZE = 30

# Professional color scheme (BGR format)
COLORS = {
    0: (155, 155, 155),  # Gray for low risk
    1: (190, 140, 50),   # Blue for medium risk
    2: (50, 140, 90)     # Green for high risk
}

# Semi-transparent overlay colors (BGR format)
OVERLAY_COLORS = {
    0: (200, 200, 200),  # Light gray
    1: (220, 180, 100),  # Light blue
    2: (100, 180, 140)   # Light green
}

DANGER_LEVELS = {
    "background": 0, "aeroplane": 2, "bicycle": 1, "bird": 0, "boat": 1,
    "bottle": 0, "bus": 2, "car": 2, "cat": 0, "chair": 0, "cow": 1,
    "diningtable": 0, "dog": 1, "horse": 1, "motorbike": 2, "person": 1,
    "pottedplant": 0, "sheep": 1, "sofa": 0, "train": 2, "tvmonitor": 0
}

class FPSCounter:
    def __init__(self, window_size=FPS_WINDOW_SIZE):
        self.frame_times = []
        self.window_size = window_size
        self.last_time = time.time()

    def update(self):
        current_time = time.time()
        self.frame_times.append(current_time - self.last_time)
        self.last_time = current_time

        if len(self.frame_times) > self.window_size:
            self.frame_times.pop(0)

    def get_fps(self):
        if not self.frame_times:
            return 0
        return len(self.frame_times) / sum(self.frame_times)

class AudioAlert:
    def __init__(self):
        self.sound = None
        self.last_alert_time = 0
        self.alert_cooldown = 2.0

    def set_music_path(self, path):
        try:
            self.sound = pygame.mixer.Sound(path)
        except pygame.error as e:
            print(f"Error loading sound file: {e}")

    def play_alert(self, danger_level):
        current_time = time.time()
        if current_time - self.last_alert_time >= self.alert_cooldown and self.sound:
            if danger_level == 2:
                self.sound.play()
                self.last_alert_time = current_time

def create_overlay(image, text, position, color, opacity=0.7):
    overlay = image.copy()
    (text_width, text_height), baseline = cv2.getTextSize(text, FONT_FACE, FONT_SCALE, FONT_THICKNESS)

    padding = LABEL_PADDING
    p1 = (position[0] - padding, position[1] - text_height - padding)
    p2 = (position[0] + text_width + padding, position[1] + padding)

    cv2.rectangle(overlay, p1, p2, color, cv2.FILLED)
    cv2.addWeighted(overlay, opacity, image, 1 - opacity, 0, image)

    cv2.putText(image, text, position, FONT_FACE, FONT_SCALE, (0, 0, 0), FONT_THICKNESS)

def draw_legend(image):
    legend_height = 80
    legend_width = 120
    padding = 10

    overlay = image.copy()
    cv2.rectangle(overlay, (padding, padding), (legend_width + padding, legend_height + padding), (240, 240, 240), cv2.FILLED)
    cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)

    y_offset = 30
    for i, level in enumerate(['Low', 'Medium', 'High']):
        cv2.rectangle(image, (padding + 5, padding + y_offset * i + 5), (padding + 20, padding + y_offset * i + 20), COLORS[i], cv2.FILLED)
        cv2.putText(image, level, (padding + 30, padding + y_offset * i + 15), FONT_FACE, FONT_SCALE, (0, 0, 0), FONT_THICKNESS)

def process_frame(frame, fps_counter, audio_alert):
    fps_counter.update()
    current_fps = fps_counter.get_fps()

    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)

    detections = net.forward()
    max_danger_level = 0

    draw_legend(frame)

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > CONFIDENCE_THRESHOLD:
            idx = int(detections[0, 0, i, 1])
            class_name = CLASSES[idx]
            danger_level = DANGER_LEVELS[class_name]
            max_danger_level = max(max_danger_level, danger_level)

            box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            (startX, startY, endX, endY) = box.astype("int")

            cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[danger_level], BOX_THICKNESS)

            y = startY - 15 if startY - 15 > 15 else startY + 15
            label = f"{class_name} ({confidence * 100:.1f}%)"
            distance = (KNOWN_WIDTH * FOCAL_LENGTH) / (endX - startX)
            distance_label = f"{distance:.1f}m"

            create_overlay(frame, label, (startX, y), OVERLAY_COLORS[danger_level])
            create_overlay(frame, distance_label, (startX, y + 25), OVERLAY_COLORS[danger_level])

    timestamp = datetime.now().strftime("%H:%M:%S")
    create_overlay(frame, timestamp, (frame.shape[1] - 100, 30), (220, 220, 220))

    create_overlay(frame, f"FPS: {current_fps:.1f}", (frame.shape[1] - 100, frame.shape[0] - 20), (220, 220, 220))

    cv2.imshow("Professional Object Detection", frame)

def detect_objects(input_type, fps_counter, audio_alert):
    if input_type == 'webcam':
        cap = cv2.VideoCapture(0)
    elif input_type == 'video':
        video_path = input("Enter video file path: ")
        cap = cv2.VideoCapture(video_path)
    else:
        print("Invalid input type")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        process_frame(frame, fps_counter, audio_alert)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    fps_counter = FPSCounter()
    audio_alert = AudioAlert()
    audio_alert.set_music_path('alert.wav')

    print("Select input source:")
    print("1. Webcam")
    print("2. Video file")
    choice = input("Enter 1 or 2: ")

    if choice == '1':
        detect_objects('webcam', fps_counter, audio_alert)
    elif choice == '2':
        detect_objects('video', fps_counter, audio_alert)
    else:
        print("Invalid option")

if __name__ == "__main__":
    main()
