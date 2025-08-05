import cv2
import numpy as np
import time

# Load the pre-trained model and class labels
net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'mobilenet_iter_73000.caffemodel')
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", 
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor", "apple", "banana", 
           "carrot", "donut", "hot dog", "orange", "pizza", "sandwich", "broccoli", "cake", "cup", "fork", "knife", "spoon", 
           "bowl", "wine glass", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "frisbee", 
           "kite", "snowboard", "skis", "snowball", "motor", "truck", "traffic light", "fire hydrant", "parking meter", 
           "bench", "dog", "cat", "horse", "person", "cow", "table", "laptop", "potted plant", "vase", "hair dryer", "toothbrush"]

COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

def detect_objects(input_type):
    if input_type == 'webcam':
        cap = cv2.VideoCapture(0)  # Open webcam
    elif input_type == 'video':
        video_path = input("Enter video file path: ")
        cap = cv2.VideoCapture(video_path)  # Open video file
    elif input_type == 'image':
        image_path = input("Enter image file path: ")
        image = cv2.imread(image_path)  # Open image file

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
    # Prepare the frame for object detection
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.2:
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            (startX, startY, endX, endY) = box.astype("int")
            label = f"{CLASSES[idx]}: {confidence * 100:.2f}%"
            cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

    cv2.imshow("Object Detection", frame)

def process_image(image):
    # Prepare the image for object detection
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.2:
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])
            (startX, startY, endX, endY) = box.astype("int")
            label = f"{CLASSES[idx]}: {confidence * 100:.2f}%"
            cv2.rectangle(image, (startX, startY), (endX, endY), COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(image, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

    cv2.imshow("Object Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
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
