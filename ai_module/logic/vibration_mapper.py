def map_distance_to_vibration(distance, max_distance=600, tolerance=40):
    if distance is None:
        return 0
        
    # If the user is close enough, give them the maximum "success" vibration
    if distance <= tolerance:
        return 255
        
    # Cap the distance
    distance = min(distance, max_distance)
    
    # Calculate intensity for the remaining distance
    # We subtract tolerance so the drop-off starts smoothly outside the deadzone
    adjusted_distance = distance - tolerance
    adjusted_max = max_distance - tolerance
    
    intensity = 255 * (1 - (adjusted_distance / adjusted_max))
    
    return int(intensity)


def get_vibration_color(intensity):
    """
    Returns an OpenCV BGR color based on vibration intensity.
    Red (Low) -> Yellow (Medium) -> Green (High)
    """
    if intensity > 200:
        return (0, 255, 0)      # Green (On target)
    elif intensity > 100:
        return (0, 255, 255)    # Yellow (Getting close)
    else:
        return (0, 0, 255)      # Red (Far away)
