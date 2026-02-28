import cv2
import numpy as np

def detect_low_visibility(frame, brightness_threshold=40, contrast_threshold=20):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    brightness = np.mean(gray)
    contrast = np.std(gray)

    # Too dark OR too low contrast (fog/smoke-like)
    if brightness < brightness_threshold or contrast < contrast_threshold:
        return True

    return False