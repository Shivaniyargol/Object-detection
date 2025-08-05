import cv2
import numpy as np
import time
import pygame
from datetime import datetime
from typing import Tuple, Dict, Optional

# Initialize pygame mixer for audio alerts
pygame.mixer.init()

class FPSCounter:
    """FPS counter with moving average."""
    def __init__(self, window_size: int = 30):
        self.frame_times = []
        self.window_size = window_size
        self.last_time = time.time()

    def update(self) -> None:
        """Update frame times."""
        current_time = time.time()
        self.frame_times.append(current_time - self.last_time)
        self.last_time = current_time

        if len(self.frame_times) > self.window_size:
            self.frame_times.pop(0)

    def get_fps(self) -> float:
        """Calculate FPS."""
        if not self.frame_times:
            return 0.0
        return len(self.frame_times) / sum(self.frame_times)

class DesignConfig:
    """Configuration for UI design elements."""
    def __init__(self):
        self.UI_DARK = (40, 40, 40)
        self.UI_LIGHT = (255, 255, 255)
        self.STATUS_COLORS = {'normal': (0, 255, 0), 'warning': (255, 255, 0), 'error': (255, 0, 0)}
        self.BG_COLORS = {'low_risk': (0, 255, 0), 'medium_risk': (255, 255, 0), 'high_risk': (255, 0, 0)}
        self.FONT_FAMILY = cv2.FONT_HERSHEY_SIMPLEX
        self.FONT_SCALE = 0.5
        self.FONT_THICKNESS = 1
        self.PADDING = 5
        self.LABEL_OPACITY = 0.8
        self.CORNER_RADIUS = 5
        self.BOX_THICKNESS = 2

class AudioAlert:
    """Audio alert system for risk notifications."""
    def __init__(self):
        self.sound = None
        self.last_alert_time = 0
        self.alert_cooldown = 2.0  # Minimum time between alerts (seconds)

    def set_sound(self, path: str) -> None:
        """Load sound for alerts."""
        try:
            self.sound = pygame.mixer.Sound(path)
        except pygame.error as e:
            print(f"Error loading sound: {e}")

    def play_alert(self, risk_level: int) -> None:
        """Play alert for high-risk detections."""
        current_time = time.time()
        if risk_level == 2 and self.sound and (current_time - self.last_alert_time >= self.alert_cooldown):
            self.sound.play()
            self.last_alert_time = current_time

class ObjectDetector:
    """Object detector class for inference."""
    def __init__(self, model_path: str, config_path: str, classes_path: Optional[str] = None):
        self.net = cv2.dnn.readNetFromCaffe(config_path, model_path)
        self.classes = self._load_classes(classes_path) if classes_path else []
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
    """UI for displaying detections and stats."""
    def __init__(self):
        self.config = DesignConfig()
        self._init_display_settings()

    def _init_display_settings(self):
        self.header_height = 40
        self.footer_height = 35
        self.sidebar_width = 180

    def draw_rounded_rectangle(self, image: np.ndarray, p1: Tuple[int, int], p2: Tuple[int, int], color: Tuple[int, int, int], opacity: float, radius: int) -> None:
        overlay = image.copy()
        cv2.rectangle(overlay, (p1[0] + radius, p1[1]), (p2[0] - radius, p2[1]), color, -1)
        cv2.rectangle(overlay, (p1[0], p1[1] + radius), (p2[0], p2[1] - radius), color, -1)
        for corner in [(p1[0] + radius, p1[1] + radius), (p2[0] - radius, p1[1] + radius), (p1[0] + radius, p2[1] - radius), (p2[0] - radius, p2[1] - radius)]:
            cv2.circle(overlay, corner, radius, color, -1)
        cv2.addWeighted(overlay, opacity, image, 1 - opacity, 0, image)

    def draw_header(self, frame: np.ndarray) -> None:
        self.draw_rounded_rectangle(frame, (0, 0), (frame.shape[1], self.header_height), self.config.UI_DARK, 0.95, 5)
        cv2.circle(frame, (20, self.header_height // 2), 6, self.config.STATUS_COLORS['normal'], -1)
        timestamp = datetime.now().strftime("%H:%M:%S | %Y-%m-%d")
        cv2.putText(frame, timestamp, (40, self.header_height - 15), self.config.FONT_FAMILY, self.config.FONT_SCALE, self.config.UI_LIGHT, self.config.FONT_THICKNESS)

    def draw_sidebar(self, frame: np.ndarray, stats: Dict) -> None:
        self.draw_rounded_rectangle(frame, (frame.shape[1] - self.sidebar_width, self.header_height), (frame.shape[1], frame.shape[0] - self.footer_height), self.config.UI_DARK, 0.95, 5)
        y_offset = self.header_height + 30
        for key, value in stats.items():
            text = f"{key}: {value}"
            cv2.putText(frame, text, (frame.shape[1] - self.sidebar_width + 10, y_offset), self.config.FONT_FAMILY, self.config.FONT_SCALE, self.config.UI_LIGHT, self.config.FONT_THICKNESS)
            y_offset += 25

def process_frame(frame: np.ndarray, detector: ObjectDetector, ui: ProfessionalUI, fps: float, audio_alert: AudioAlert) -> None:
    ui.draw_header(frame)

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
            risk_level = detector.risk_levels.get(class_name, 0)
            max_risk_level = max(max_risk_level, risk_level)

            stats['Objects'] += 1
            stats['High Risk'] += risk_level == 2
            stats['Medium Risk'] += risk_level == 1
            stats['Low Risk'] += risk_level == 0

    ui.draw_sidebar(frame, stats)
    audio_alert.play_alert(max_risk_level)

# Initialization
model_path = "C:\\Users\\Nayanaa\\Desktop\\minip\\mobilenet_iter_73000.caffemodel"
config_path = "C:\\Users\\Nayanaa\\Desktop\\minip\\deploy.prototxt"
alert_sound_path = "C:\\Users\\Nayanaa\\Desktop\\minip\\alert.wav"

detector = ObjectDetector(model_path, config_path)
ui = ProfessionalUI()
fps_counter = FPSCounter()
audio_alert = AudioAlert()
audio_alert.set_sound(alert_sound_path)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    fps_counter.update()
    fps = fps_counter.get_fps()

    process_frame(frame, detector, ui, fps, audio_alert)
    cv2.imshow("Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
