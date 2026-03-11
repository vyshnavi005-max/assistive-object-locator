import cv2
import time
from ultralytics import YOLO

# Local module imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from logic.distance_calculator import calculate_center, calculate_distance
from logic.vibration_mapper import map_distance_to_vibration, get_vibration_color

# --- Configuration ---
TARGET_OBJECT = "bottle"  # What you are telling the AI to "Find". Try "cell phone", "bottle", etc.
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'yolov8n.pt')

# --- Initialization ---
print(f"Loading YOLOv8 model from {MODEL_PATH}...")
yolo_model = YOLO(MODEL_PATH)

# Start Video Capture
cap = cv2.VideoCapture(0)

print(f"\n--- AI Simulator Started ---")
print(f"TARGET: Looking for a '{TARGET_OBJECT}'")
print(f"Tracking distance from SCREEN CENTER to TARGET.")
print(f"Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    h, w, _ = frame.shape
    
    # Define screen center (simulating the pointer)
    screen_center = (w // 2, h // 2)
    
    # Draw screen center crosshair
    cv2.drawMarker(frame, screen_center, (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
    cv2.putText(frame, "SCREEN CENTER", (screen_center[0] + 15, screen_center[1] + 15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # 1. RUN YOLO OBJECT DETECTION
    yolo_results = yolo_model(frame, stream=True, verbose=False)
    
    target_bbox = None
    target_center = None
    
    # Parse YOLO results to find the target object
    for r in yolo_results:
        boxes = r.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            class_name = yolo_model.names[cls_id]
            
            x1, y1, x2, y2 = box.xyxy[0]
            bbox = [int(x1), int(y1), int(x2), int(y2)]
            center = calculate_center(bbox)
            
            # If we found the target object (e.g., 'cup' or 'bottle')
            if class_name == TARGET_OBJECT and target_bbox is None:
                target_bbox = bbox
                target_center = center
                
                # Draw a blue bounding box around the target
                cv2.rectangle(frame, (target_bbox[0], target_bbox[1]), (target_bbox[2], target_bbox[3]), (255, 0, 0), 3)
                cv2.putText(frame, f"TARGET: {TARGET_OBJECT}", (target_bbox[0], target_bbox[1] - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                cv2.circle(frame, target_center, 5, (255, 0, 0), -1)
            else:
                # Draw other objects in yellow
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 255), 1)
                cv2.putText(frame, class_name, (bbox[0], bbox[1] - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            
    # 2. CALCULATE DISTANCE & SIMULATE HAPTICS
    vibe_intensity = 0
    vibe_color = (0, 0, 255) # Default Red
    distance_text = "N/A"
    
    if target_center:
        # Estimate depth distance using the bounding box width as a proxy
        # Formula: Distance = (Real Object Width * Focal Length) / Perceived Pixel Width
        
        # Approximate values for a standard webcam and a bottle
        REAL_WIDTH_MM = 70.0  # Approx 7cm width for a standard bottle
        FOCAL_LENGTH = 600.0  # Assumed focal length for a typical 720p/1080p webcam
        
        perceived_width_px = target_bbox[2] - target_bbox[0]
        
        if perceived_width_px > 0:
            # Distance in millimeters
            estimated_distance_mm = (REAL_WIDTH_MM * FOCAL_LENGTH) / perceived_width_px
            # Convert to cm for easier reading
            estimated_distance_cm = estimated_distance_mm / 10.0
            
            distance_text = f"{int(estimated_distance_cm)} cm"
            
            # Map to vibration intensity (0-255)
            # The closer it is, the stronger the vibration.
            # Let's say max vibration happens at 20cm, and it linearly decreases up to 200cm
            MIN_DIST_CM = 20.0
            MAX_DIST_CM = 200.0
            
            # Clamp the distance
            clamped_dist = max(MIN_DIST_CM, min(estimated_distance_cm, MAX_DIST_CM))
            
            # Invert the mapping: lower distance = higher intensity
            ratio = 1.0 - ((clamped_dist - MIN_DIST_CM) / (MAX_DIST_CM - MIN_DIST_CM))
            vibe_intensity = int(ratio * 255)
            
            # Get color code mapping
            vibe_color = get_vibration_color(vibe_intensity)
            
            # Draw tracking line between center and object
            cv2.line(frame, screen_center, target_center, vibe_color, 3)
            
            # Print to console (simulated hardware command)
            print(f"DEPTH: {distance_text} | BLE COMMAND -> VIBRATION_INTENSITY: {vibe_intensity:03d}")
            
    # 3. ON-SCREEN UI
    cv2.putText(frame, f"TARGET: {TARGET_OBJECT} | DEPTH: {distance_text}", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, vibe_color, 2)
    cv2.putText(frame, f"VIBE INTENSITY: {vibe_intensity}/255", (20, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, vibe_color, 2)

    # Show the frame
    cv2.imshow("AI Assistant Prototype Simulator", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
