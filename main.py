import cv2
import numpy as np
import os
import time
from gtts import gTTS
from playsound import playsound

# -------------------------------
# Load class labels (COCO dataset)
# -------------------------------
with open("coco.names", "r") as f:
    CLASSES = [line.strip() for line in f.readlines()]

# -------------------------------
# Load YOLOv3 model
# -------------------------------
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Generate colors for classes
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# -------------------------------
# Initialize webcam
# -------------------------------
cap = cv2.VideoCapture(0)
frame_count = 0

print("[INFO] YOLOv3 model loaded successfully")

# -------------------------------
# Main loop
# -------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    height, width, _ = frame.shape

    boxes = []
    confidences = []
    class_ids = []
    centers = []

    # Run detection every 30 frames (performance optimization)
    if frame_count % 30 == 0:

        # Convert frame to blob
        blob = cv2.dnn.blobFromImage(
            frame,
            scalefactor=1 / 255.0,
            size=(416, 416),
            swapRB=True,
            crop=False
        )

        net.setInput(blob)
        detections = net.forward(output_layers)

        # -------------------------------
        # Process detections
        # -------------------------------
        for output in detections:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
                    centers.append((center_x, center_y))

        # -------------------------------
        # Apply Non-Maximum Suppression
        # -------------------------------
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        spoken_descriptions = []

        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                label = CLASSES[class_ids[i]]
                color = COLORS[class_ids[i]]

                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(
                    frame,
                    f"{label}",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

                # -------------------------------
                # Determine object position
                # -------------------------------
                cx, cy = centers[i]

                horizontal = (
                    "left" if cx < width / 3
                    else "center" if cx < 2 * width / 3
                    else "right"
                )

                vertical = (
                    "top" if cy < height / 3
                    else "middle" if cy < 2 * height / 3
                    else "bottom"
                )

                spoken_descriptions.append(f"{vertical} {horizontal} {label}")

        # -------------------------------
        # Text-to-Speech Output
        # -------------------------------
        if spoken_descriptions:
            speech_text = "Detected " + ", ".join(spoken_descriptions)
            tts = gTTS(text=speech_text, lang="en")
            tts.save("output.mp3")
            playsound("output.mp3")
            os.remove("output.mp3")

    # Display output window
    cv2.imshow("YOLOv3 Object Detection with Audio Feedback", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -------------------------------
# Cleanup
# -------------------------------
cap.release()
cv2.destroyAllWindows()
