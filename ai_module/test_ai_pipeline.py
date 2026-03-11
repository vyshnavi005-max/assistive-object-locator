import cv2
import time
import math
import mediapipe as mp
from ultralytics import YOLO

# Local module imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from logic.distance_calculator import calculate_center, calculate_distance, get_fingertip_coords
from logic.vibration_mapper import map_distance_to_vibration, get_vibration_color
from voice_command.listener import VoiceListener

# --- Configuration ---
TARGET_OBJECT = "bottle"  # Default target (will be overridden by voice commands)
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'yolov8n.pt')
HAND_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'hand_landmarker.task')
VOSK_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vosk_model')

# --- Initialization ---
print(f"Loading YOLOv8 model from {MODEL_PATH}...")
yolo_model = YOLO(MODEL_PATH)

print(f"Loading MediaPipe Hand Tasks model...")
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=HAND_MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1
)

# Start Video Capture
cap = cv2.VideoCapture(0)

# --- Start Voice Listener ---
voice_listener = VoiceListener(model_path=VOSK_MODEL_PATH)
voice_listener.start()

print(f"\n--- AI Simulator Started ---")
print(f"DEFAULT TARGET: '{TARGET_OBJECT}'")
print(f"VOICE COMMANDS: Say 'find bottle', 'find cup', 'find phone', etc.")
print(f"Press 'q' to quit")

def is_hand_inside_bbox(hand_point, bbox):
    """
    Returns True if the hand fingertip pixel is inside the object's bounding box.
    This is the most reliable way to detect a 'grab' event from a 2D camera.
    bbox = [x1, y1, x2, y2]
    """
    if not hand_point or not bbox:
        return False
    x, y = hand_point
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2


with HandLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        h, w, _ = frame.shape
        
        # --- Poll Voice Commands (non-blocking) ---
        new_target = voice_listener.get_new_target()
        if new_target:
            TARGET_OBJECT = new_target
            print(f"[VOICE] Target switched to: '{TARGET_OBJECT}'")
        
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
                
                # If we found the target object (e.g., 'cup' or 'bottle')
                if class_name == TARGET_OBJECT:
                    x1, y1, x2, y2 = box.xyxy[0]
                    target_bbox = [int(x1), int(y1), int(x2), int(y2)]
                    target_center = calculate_center(target_bbox)
                    
                    # Draw a blue bounding box around the target
                    cv2.rectangle(frame, (target_bbox[0], target_bbox[1]), (target_bbox[2], target_bbox[3]), (255, 0, 0), 2)
                    cv2.putText(frame, f"TARGET: {TARGET_OBJECT}", (target_bbox[0], target_bbox[1] - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    cv2.circle(frame, target_center, 5, (255, 0, 0), -1)
                    break # Track the first valid target
            
            # If we found one, stop checking other boxes
            if target_center:
                break
                
        # 2. RUN MEDIAPIPE HAND TRACKING
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int(time.time() * 1000)
        
        hand_result = landmarker.detect_for_video(mp_image, timestamp_ms)
        
        hand_center = None
        
        # Parse MediaPipe results
        if hand_result.hand_landmarks:
            landmarks = hand_result.hand_landmarks[0] # Grab first hand
            
            # Use Index Finger Tip (Landmark 8) as the pointer (now returning z)
            fingertip_data = get_fingertip_coords(landmarks, w, h, finger_idx=8)
            if fingertip_data:
                hand_x, hand_y, hand_z = fingertip_data
                hand_center = (hand_x, hand_y)
            
            if hand_center:
                cv2.circle(frame, hand_center, 8, (0, 0, 255), -1)
                cv2.putText(frame, "INDEX FINGER", (hand_center[0] + 15, hand_center[1]), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # 3. CALCULATE DISTANCE & SIMULATE HAPTICS
        vibe_intensity = 0
        vibe_color = (0, 0, 255) # Default Red
        status_text = "Searching..." # New variable for explicit feedback
        
        # Screen Center for Camera Guidance
        screen_center = (w // 2, h // 2)
        cv2.drawMarker(frame, screen_center, (200, 200, 200), cv2.MARKER_CROSS, 20, 1)
        
        if target_center and hand_center:
            # --- PHASE 2/3: HAND TRACKING & REACHING ---
            # Calculate 2D pixel distance between hand and object
            distance_2d = calculate_distance(target_center, hand_center)
            
            # Map 2D distance to vibration intensity using the new deadzone logic
            vibe_intensity = map_distance_to_vibration(distance_2d, max_distance=800, tolerance=60)
            vibe_color = get_vibration_color(vibe_intensity)
            
            # Phase 2 vs Phase 3/4 Logic
            if vibe_intensity == 255:
                # Hand is 2D-aligned - now check depth
                REAL_WIDTH_MM = 70.0
                FOCAL_LENGTH = 600.0
                perceived_width_px = target_bbox[2] - target_bbox[0]
                
                if perceived_width_px > 0:
                    estimated_distance_mm = (REAL_WIDTH_MM * FOCAL_LENGTH) / perceived_width_px
                    estimated_distance_cm = estimated_distance_mm / 10.0
                    
                    # --- PHASE 4: GRAB DETECTION ---
                    # Check 1: Fingertip is physically inside the bounding box
                    finger_inside = is_hand_inside_bbox(hand_center, target_bbox)
                    # Check 2: Object appears very large (very close to camera) - bbox > 35% frame width
                    obj_very_close = perceived_width_px > (w * 0.35)
                    
                    if finger_inside or obj_very_close:
                        # SUCCESS! User has grabbed the object
                        status_text = "OBJECT GRABBED!"
                        vibe_intensity = 255
                        vibe_color = (0, 255, 100)  # Bright green
                        # Draw a solid filled circle on the target to celebrate
                        cv2.circle(frame, target_center, 30, (0, 255, 100), -1)
                        cv2.putText(frame, "GRABBED!", (target_center[0] - 40, target_center[1] - 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 100), 3)
                    else:
                        # --- PHASE 3: REACH FORWARD ---
                        status_text = f"REACH FORWARD! ({int(estimated_distance_cm)}cm away)"
                        vibe_color = (0, 255, 0)
                        pulse_radius = 20 + int(10 * abs(math.sin(time.time() * 5)))
                        cv2.circle(frame, target_center, pulse_radius, (0, 255, 0), 3)
            else:
                # PHASE 2: Hand Aligning (Point hand to target)
                if vibe_intensity == 0:
                    status_text = "HAND TOO FAR FROM TARGET"
                else:
                    status_text = "MOVE HAND CLOSER..."
                cv2.line(frame, hand_center, target_center, vibe_color, 3)
            
            print(f"[{status_text}] BLE COMMAND -> VIBE: {vibe_intensity:03d} | Z: {hand_z:.3f}")
            
        elif target_center and not hand_center:
            # --- PHASE 1: CAMERA CENTERING (Hand is missing) ---
            # Guide the user to turn their body/head so the object is dead-center.
            distance_to_center = calculate_distance(target_center, screen_center)
            
            # We want them to center it within a fairly large tolerance (e.g., 100 pixels)
            vibe_intensity = map_distance_to_vibration(distance_to_center, max_distance=600, tolerance=100)
            vibe_color = get_vibration_color(vibe_intensity)
            
            if vibe_intensity == 255:
                status_text = "CAMERA CENTERED! BRING HAND UP."
                vibe_color = (0, 255, 255) # Yellow to indicate ready for Hand Tracking
            else:
                status_text = "TURN CAMERA TO CENTER OBJECT."
                
            # Draw tracking line from Screen Center to Target
            cv2.line(frame, screen_center, target_center, vibe_color, 2)
            
            print(f"[{status_text}] BLE COMMAND -> VIBE: {vibe_intensity:03d}")
            
        else:
            # --- PHASE 0: SCANNING ---
            # Nothing is in frame. The blind user won't know unless we give a "Scanning" pulse.
            status_text = "SCANNING ROOM..."
            vibe_intensity = 0
            
            # Create a heartbeat pulse when completely lost (simulates a blip every second)
            if int(time.time() * 2) % 2 == 0: 
                vibe_intensity = 50 # Tiny blip buzz to let them know it's trying
                
            print(f"[SCANNING] Target not in view. Rotate body...")
                
        # 4. ON-SCREEN UI
        cv2.putText(frame, f"STATUS: {status_text}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, vibe_color, 2)
        cv2.putText(frame, f"VIBE INTENSITY: {vibe_intensity}/255", (20, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, vibe_color, 2)
        cv2.putText(frame, f"TARGET: {TARGET_OBJECT}", (20, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, "SAY: 'find bottle' or 'search phone'", (20, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        # Show the frame
        cv2.imshow("AI Assistant Prototype Simulator", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
voice_listener.stop()
cv2.destroyAllWindows()
