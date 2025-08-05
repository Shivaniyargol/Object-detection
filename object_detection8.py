import cv2
import numpy as np
import time
from datetime import datetime
from typing import Tuple, Dict, Optional
import sys
import os

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "1"

import pygame
pygame.mixer.init()



class FPSCounter:
    def __init__(self, window_size: int = 30):
        self.frame_times = []
        self.window_size = window_size
        self.last_time = time.time()

    def update(self) -> None:
        current_time = time.time()
        self.frame_times.append(current_time - self.last_time)
        self.last_time = current_time

        
        if len(self.frame_times) > self.window_size:
            self.frame_times.pop(0)

    def get_fps(self) -> float:
        if not self.frame_times:
            return 0.0
        return len(self.frame_times) / sum(self.frame_times)


class DesignConfig:
    def __init__(self):
        self.UI_DARK = (40, 40, 40)  
        self.UI_LIGHT = (255, 255, 255)  
        self.STATUS_COLORS = {
            'normal': (0, 255, 0), 
            'warning': (255, 255, 0),  
            'error': (255, 0, 0)  
        }
        self.BG_COLORS = {
            'low_risk': (0, 255, 0), 
            'medium_risk': (255, 255, 0), 
            'high_risk': (255, 0, 0)  
        }
        self.FONT_FAMILY = cv2.FONT_HERSHEY_SIMPLEX
        self.FONT_SCALE = 0.5
        self.FONT_THICKNESS = 1
        self.PADDING = 5
        self.LABEL_OPACITY = 0.8
        self.CORNER_RADIUS = 5
        self.BOX_THICKNESS = 2


class AudioAlert:
    def __init__(self):
        self.sound = None
        self.last_alert_time = 0
        self.alert_cooldown = 2.0  

    def set_sound(self, path: str) -> None:
        try:
            self.sound = pygame.mixer.Sound(path)
        except pygame.error as e:
            print(f"Error loading sound file: {e}")

    def play_alert(self, risk_level: int) -> None:
        current_time = time.time()
        if (current_time - self.last_alert_time >= self.alert_cooldown
                and self.sound and risk_level == 2):
            self.sound.play()
            self.last_alert_time = current_time


class ObjectDetector:
    def __init__(self, model_path: str, config_path: str, classes_path: Optional[str] = None):
        self.net = cv2.dnn.readNetFromCaffe(config_path, model_path)
        self.classes = self._load_classes(classes_path) if classes_path else [
            "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
            "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
            "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
        ]
        self.risk_levels = {
            "background": 0, "aeroplane": 2, "bicycle": 1, "bird": 0, "boat": 1,
            "bottle": 0, "bus": 2, "car": 2, "cat": 0, "chair": 0, "cow": 1,
            "diningtable": 0, "dog": 1, "horse": 1, "motorbike": 2, "person": 1,
            "pottedplant": 0, "sheep": 1, "sofa": 0, "train": 2, "tvmonitor": 0
        }

    @staticmethod
    def _load_classes(path: str) -> list:
        with open(path, 'r') as f:
            return [line.strip() for line in f.readlines()]


class ProfessionalUI:
    def __init__(self):
        self.config = DesignConfig()
        self._init_display_settings()

    def _init_display_settings(self):
        self.header_height = 40
        self.footer_height = 35
        self.sidebar_width = 180

    def draw_rounded_rectangle(self, image: np.ndarray, p1: Tuple[int, int], p2: Tuple[int, int], color: Tuple[int, int, int], opacity: float = 1.0, radius: int = 3) -> None:
        overlay = image.copy()

        cv2.rectangle(overlay, (p1[0] + radius, p1[1]), (p2[0] - radius, p2[1]), color, -1)
        cv2.rectangle(overlay, (p1[0], p1[1] + radius), (p2[0], p2[1] - radius), color, -1)

        corners = [
            (p1[0] + radius, p1[1] + radius),
            (p2[0] - radius, p1[1] + radius),
            (p1[0] + radius, p2[1] - radius),
            (p2[0] - radius, p2[1] - radius)
        ]

        for corner in corners:
            cv2.circle(overlay, corner, radius, color, -1)

        cv2.addWeighted(overlay, opacity, image, 1 - opacity, 0, image)

    def draw_header(self, frame: np.ndarray) -> None:
        self.draw_rounded_rectangle(frame, (0, 0), (frame.shape[1], self.header_height), self.config.UI_DARK, opacity=0.95)

        status_color = self.config.STATUS_COLORS['normal']
        cv2.circle(frame, (20, self.header_height // 2), 6, status_color, -1)

        timestamp = datetime.now().strftime("%H:%M:%S | %Y-%m-%d")
        cv2.putText(frame, timestamp, (40, self.header_height - 15), self.config.FONT_FAMILY, self.config.FONT_SCALE, self.config.UI_LIGHT, self.config.FONT_THICKNESS)

    def draw_object_label(self, frame: np.ndarray, text: str, position: Tuple[int, int], risk_level: int) -> None:
        (text_width, text_height), _ = cv2.getTextSize(text, self.config.FONT_FAMILY, self.config.FONT_SCALE, self.config.FONT_THICKNESS)

        p1 = (position[0], position[1] - text_height - self.config.PADDING * 2)
        p2 = (position[0] + text_width + self.config.PADDING * 2, position[1])

        color_key = ['low_risk', 'medium_risk', 'high_risk'][risk_level]
        self.draw_rounded_rectangle(frame, p1, p2, self.config.BG_COLORS[color_key], opacity=self.config.LABEL_OPACITY, radius=self.config.CORNER_RADIUS)

        cv2.putText(frame, text, (position[0] + self.config.PADDING, position[1] - self.config.PADDING), self.config.FONT_FAMILY, self.config.FONT_SCALE, self.config.UI_DARK, self.config.FONT_THICKNESS)

    def draw_sidebar(self, frame: np.ndarray, stats: Dict) -> None:
        self.draw_rounded_rectangle(frame, (frame.shape[1] - self.sidebar_width, self.header_height), (frame.shape[1], frame.shape[0] - self.footer_height), self.config.UI_DARK, opacity=0.95)

        y_offset = self.header_height + 30
        for key, value in stats.items():
            text = f"{key}: {value}"
            cv2.putText(frame, text, (frame.shape[1] - self.sidebar_width + 10, y_offset), self.config.FONT_FAMILY, self.config.FONT_SCALE, self.config.UI_LIGHT, self.config.FONT_THICKNESS)
            y_offset += 25

    def draw_footer(self, frame: np.ndarray, fps: float) -> None:
        self.draw_rounded_rectangle(frame, (0, frame.shape[0] - self.footer_height), (frame.shape[1], frame.shape[0]), self.config.UI_DARK, opacity=0.95)

        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(frame, fps_text, (10, frame.shape[0] - 10), self.config.FONT_FAMILY, self.config.FONT_SCALE, self.config.UI_LIGHT, self.config.FONT_THICKNESS)


def process_frame(frame: np.ndarray, detector: ObjectDetector, ui: ProfessionalUI, fps: float, audio_alert: AudioAlert) -> None:
    ui.draw_header(frame)
    ui.draw_footer(frame, fps)

    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    detector.net.setInput(blob)
    detections = detector.net.forward()

    stats = {'Objects': 0, 'High Risk': 0, 'Medium Risk': 0, 'Low Risk': 0}

    max_risk_level = 0

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.2:  
            idx = int(detections[0, 0, i, 1])
            class_name = detector.classes[idx]
            risk_level = detector.risk_levels[class_name]
            max_risk_level = max(max_risk_level, risk_level)

            stats['Objects'] += 1
            stats[['Low Risk', 'Medium Risk', 'High Risk'][risk_level]] += 1

            box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            (startX, startY, endX, endY) = box.astype("int")

            color_key = ['low_risk', 'medium_risk', 'high_risk'][risk_level]
            cv2.rectangle(frame, (startX, startY), (endX, endY), ui.config.BG_COLORS[color_key], ui.config.BOX_THICKNESS)

            label = f"{class_name} ({confidence*100:.1f}%)"
            ui.draw_object_label(frame, label, (startX, startY - 10), risk_level)

    ui.draw_sidebar(frame, stats)

    audio_alert.play_alert(max_risk_level)

def main():
    try:
        detector = ObjectDetector('mobilenet_iter_73000.caffemodel', 'deploy.prototxt')
        ui = ProfessionalUI()
        fps_counter = FPSCounter()
        audio_alert = AudioAlert()
        audio_alert.set_sound('alert.wav')

        print("Select input source:")
        print("1. Webcam")
        print("2. Video file")
        choice = input("Enter choice (1/2): ")

        if choice == '1':
            cap = cv2.VideoCapture(0)
        elif choice == '2':
            video_path = input("Enter video file path: ")
            cap = cv2.VideoCapture(video_path)
        else:
            print("Invalid choice. Defaulting to webcam.")
            cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            raise RuntimeError("Failed to open video source")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            fps_counter.update()
            process_frame(frame, detector, ui, fps_counter.get_fps(), audio_alert)

            cv2.imshow('Enterprise Object Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
