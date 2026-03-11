# AI-Assisted Object Localization System

> A wearable assistive technology enabling visually impaired users to independently locate everyday objects using AI, computer vision, and tactile feedback.

---

## 🏗️ Project Architecture

The system consists of three main components:

1.  **📱 Mobile Application (AI Processing Unit)**
    *   Acts as the main brain. Uses **CameraX API** for video streaming, **OpenCV** for image processing, and **TensorFlow Lite/MediaPipe** for object and hand detection.
    *   Processes voice commands using Android `SpeechRecognizer` and outputs audio via `TextToSpeech`.
2.  **📷 Smart Goggles (Environment Scanning)**
    *   An **ESP32-CAM** module mounted on glasses. It continuously captures live video and streams it via WiFi to the mobile application.
3.  **⌚ Haptic Wristband (Directional Feedback)**
    *   An **ESP32** microcontroller equipped with an **MPU6050 IMU** (to track hand orientation) and **4 Coin Vibration Motors** (Up, Down, Left, Right).
    *   Receives `VIBRATION_INTENSITY` data from the phone via **Bluetooth Low Energy (BLE)** using PWM for directional guidance.

---

## ⚙️ Core Logic Workflow

1.  **Object Detection (YOLOv8)**: The camera scans the room. If the user asked to "Find my water bottle", the AI detects the bottle and calculates its center `(x_obj, y_obj)`.
2.  **Hand Tracking (MediaPipe)**: The system simultaneously tracks the user's hand and calculates its center/pointing direction `(x_hand, y_hand)`.
3.  **Distance Mapping**: Calculates the distance between the target object and where the hand is pointing: `distance = √((x_obj − x_hand)² + (y_obj − y_hand)²)`
4.  **Haptic Feedback**: Maps that distance to vibration intensity. If the hand points directly at the object, the vibration is maximum (PWM 255). As the hand drifts away, the vibration weakens (PWM ~30-150).

---

## 🧑‍💻 Python Environment & Dependencies

This system requires a highly specific combination of libraries to avoid Protocol Buffer (`protobuf`) runtime conflicts between TensorFlow and MediaPipe:

### Setup
Ensure you are using Python 3.10+ (tested on Python 3.10.19).

```bash
# Strongly recommended to use a virtual environment
conda create -n vision_env python=3.10
conda activate vision_env

# Install exact working dependencies
pip install -r requirements.txt
```

### Key Library Versions

*   `mediapipe==0.10.32` (Modern MediaPipe Tasks API, legacy `solutions` API was removed)
*   `tensorflow==2.21.0` (Core ML engine)
*   `protobuf==6.31.1` (Critical version! TensorFlow 2.21 requires protobuf 6.31.1 to avoid `runtime_version` import crashes when used alongside MediaPipe)
*   `ultralytics==8.4.21` (For YOLOv8 object detection)
*   `opencv-python==4.13.0.92` (Computer Vision operations)
