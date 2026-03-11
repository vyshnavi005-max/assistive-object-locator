from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("yolov8n.pt")

# Open camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # Run object detection
    results = model(frame)

    # Draw detected objects
    annotated_frame = results[0].plot()

    cv2.imshow("Object Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()