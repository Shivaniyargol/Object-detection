import cv2
import numpy as np

net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'mobilenet_iter_73000.caffemodel')

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor", "apple", "banana", "carrot", "donut",
           "hot dog", "orange", "pizza", "sandwich", "broccoli", "cake", 
           "cup", "fork", "knife", "spoon", "bowl", "wine glass", "baseball bat", 
           "baseball glove", "skateboard", "surfboard", "tennis racket", "frisbee",
           "kite", "snowboard", "skis", "snowball", "motor", "truck", "traffic light",
           "fire hydrant", "parking meter", "bench", "dog", "cat", "horse", "person", 
           "cow", "table", "laptop", "potted plant", "vase", "hair dryer", "toothbrush"]

COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

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

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()