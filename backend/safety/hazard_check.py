import cv2
import numpy as np

def detect_low_visibility(frame, threshold=40):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)

    # Very dark environment
    if brightness < threshold:
        return True

    return False