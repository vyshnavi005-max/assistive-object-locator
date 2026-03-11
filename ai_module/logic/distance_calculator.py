import math

def calculate_center(bbox):
    """
    Calculate the center point of a YOLO bounding box.
    bbox format: [x1, y1, x2, y2]
    """
    x_center = int((bbox[0] + bbox[2]) / 2)
    y_center = int((bbox[1] + bbox[3]) / 2)
    return (x_center, y_center)


def calculate_distance(point1, point2):
    """
    Calculate Euclidean distance between two points.
    point1, point2 format: (x, y)
    """
    return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)


def get_fingertip_coords(hand_landmarks, width, height, finger_idx=8):
    """
    Get the (x, y) pixel coordinates and z relative depth of a specific hand landmark.
    Default finger_idx 8 is the Index Finger Tip.
    """
    if not hand_landmarks:
        return None
        
    landmark = hand_landmarks[finger_idx]
    x = int(landmark.x * width)
    y = int(landmark.y * height)
    z = landmark.z  # Extract depth
    return (x, y, z)
