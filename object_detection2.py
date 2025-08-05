import cv2
import numpy as np

# Load the pre-trained MobileNet SSD model and configuration
net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'mobilenet_iter_73000.caffemodel')

# List of class labels the model can detect
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

# Generate random colors for each class
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# Load the image for object detection
image_path = "img2.jpg"  # Replace with the path to your image
image = cv2.imread(image_path)

if image is None:
    print("Error: Image not found. Check the image path.")
else:
    # Prepare the image for the model (blob format)
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    # Loop through detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Filter weak detections (confidence threshold: 0.2)
        if confidence > 0.2:
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])
            (startX, startY, endX, endY) = box.astype("int")

            # Create label with class name and confidence
            label = f"{CLASSES[idx]}: {confidence * 100:.2f}%"
            cv2.rectangle(image, (startX, startY), (endX, endY), COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(image, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

    # Display the image with detections
    cv2.imshow("Object Detection", image)

    # Save the output image
    output_path = "output_image.jpg"
    cv2.imwrite(output_path, image)
    print(f"Output saved to {output_path}")

    # Wait for a key press and close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()
