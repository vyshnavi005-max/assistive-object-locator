import cv2
import mediapipe as mp
import time

# MediaPipe Tasks API imports
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Hand landmark connections for drawing (21 landmarks)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),       # Index finger
    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle finger
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring finger
    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (5, 9), (9, 13), (13, 17),             # Palm
]


def draw_landmarks(frame, hand_landmarks, width, height):
    """Draw hand landmarks and connections on the frame."""
    # Draw connections
    for start_idx, end_idx in HAND_CONNECTIONS:
        start = hand_landmarks[start_idx]
        end = hand_landmarks[end_idx]
        x1, y1 = int(start.x * width), int(start.y * height)
        x2, y2 = int(end.x * width), int(end.y * height)
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Draw landmark points
    for landmark in hand_landmarks:
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)


# Configure the HandLandmarker
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

with HandLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create MediaPipe Image from numpy array
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Get timestamp in milliseconds
        timestamp_ms = int(time.time() * 1000)

        # Detect hand landmarks
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        # Draw landmarks if hands are detected
        h, w, _ = frame.shape
        if result.hand_landmarks:
            for hand_landmarks in result.hand_landmarks:
                draw_landmarks(frame, hand_landmarks, w, h)

        cv2.imshow("Hand Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()