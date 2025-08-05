import cv2
import numpy as np
import time
import pygame

# Initialize pygame mixer for audio
pygame.mixer.init()

# Load the pre-trained model and class labels
net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'mobilenet_iter_73000.caffemodel')
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", 
          "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", 
          "train", "tvmonitor"]

# Add the missing OBSTACLE_THRESHOLD constant
OBSTACLE_THRESHOLD = 100

# Define danger levels for each class (0: Low, 1: Medium, 2: High)
DANGER_LEVELS = {
    "background": 0, "aeroplane": 2, "bicycle": 1, "bird": 0, "boat": 1, "bottle": 0,
    "bus": 2, "car": 2, "cat": 0, "chair": 0, "cow": 1, "diningtable": 0, "dog": 1,
    "horse": 1, "motorbike": 2, "person": 1, "pottedplant": 0, "sheep": 1, "sofa": 0,
    "train": 2, "tvmonitor": 0
}

# Color mapping for danger levels
DANGER_COLORS = {
    0: (0, 255, 0),    # Green for low danger
    1: (0, 255, 255),  # Yellow for medium danger
    2: (0, 0, 255)     # Red for high danger
}

class AudioAlert:
    def __init__(self):
        self.sound = None
        self.last_alert_time = 0
        self.alert_cooldown = 2.0  # Seconds between alerts

    def set_music_path(self, path):
        """Set the audio file path for alerts"""
        try:
            self.sound = pygame.mixer.Sound(path)
        except pygame.error as e:
            print(f"Error loading sound file: {e}")

    def play_alert(self, danger_level):
        """Play alert sound based on danger level and cooldown"""
        current_time = time.time()
        if current_time - self.last_alert_time >= self.alert_cooldown and self.sound:
            if danger_level == 2:  # Only play for high danger
                self.sound.play()
                self.last_alert_time = current_time

# Create audio alert instance
audio_alert = AudioAlert()

def detect_objects(input_type):
    if input_type == 'webcam':
        cap = cv2.VideoCapture(0)
    elif input_type == 'video':
        video_path = input("Enter video file path: ")
        cap = cv2.VideoCapture(video_path)
    elif input_type == 'image':
        image_path = input("Enter image file path: ")
        image = cv2.imread(image_path)
        if image is None:
            print("Error: Image not found.")
            return
        process_image(image)
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        process_frame(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def process_frame(frame):
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    move_direction = "Clear"
    frame_center = frame.shape[1] // 2
    max_danger_level = 0

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.2:
            idx = int(detections[0, 0, i, 1])
            class_name = CLASSES[idx]
            danger_level = DANGER_LEVELS[class_name]
            max_danger_level = max(max_danger_level, danger_level)
            
            box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            (startX, startY, endX, endY) = box.astype("int")

            object_width = endX - startX
            if object_width > OBSTACLE_THRESHOLD:
                if (startX + endX) // 2 < frame_center:
                    move_direction = "Move Right"
                else:
                    move_direction = "Move Left"

                # Use danger level colors for bounding box
                cv2.rectangle(frame, (startX, startY), (endX, endY), DANGER_COLORS[danger_level], 2)
                danger_text = ["Low", "Medium", "High"][danger_level]
                label = f"{class_name}: {danger_text} Danger"
            else:
                label = f"{class_name}: Clear"

            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, DANGER_COLORS[danger_level], 2)

    # Play audio alert based on highest danger level detected
    audio_alert.play_alert(max_danger_level)

    # Display movement suggestion and danger level
    cv2.putText(frame, f"Direction: {move_direction}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    danger_text = ["Low", "Medium", "High"][max_danger_level]
    cv2.putText(frame, f"Danger Level: {danger_text}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, DANGER_COLORS[max_danger_level], 2)
    cv2.imshow("Object Detection & Obstacle Avoidance", frame)

def process_image(image):
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    max_danger_level = 0

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.2:
            idx = int(detections[0, 0, i, 1])
            class_name = CLASSES[idx]
            danger_level = DANGER_LEVELS[class_name]
            max_danger_level = max(max_danger_level, danger_level)

            box = detections[0, 0, i, 3:7] * np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])
            (startX, startY, endX, endY) = box.astype("int")
            
            cv2.rectangle(image, (startX, startY), (endX, endY), DANGER_COLORS[danger_level], 2)
            danger_text = ["Low", "Medium", "High"][danger_level]
            label = f"{class_name}: {danger_text} Danger ({confidence * 100:.2f}%)"
            
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(image, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, DANGER_COLORS[danger_level], 2)

    # Display overall danger level
    danger_text = ["Low", "Medium", "High"][max_danger_level]
    cv2.putText(image, f"Danger Level: {danger_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, DANGER_COLORS[max_danger_level], 2)
    cv2.imshow("Object Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    print("Main function started")
    audio_alert.set_music_path('alert.wav')  # Replace with your audio file path
    
    print("Select input source:")
    print("1. Webcam")
    print("2. Video file")
    print("3. Image file")
    option = input("Enter 1, 2 or 3: ")

    if option == '1':
        detect_objects('webcam')
    elif option == '2':
        detect_objects('video')
    elif option == '3':
        detect_objects('image')
    else:
        print("Invalid option")

if __name__ == "__main__":
    main()
