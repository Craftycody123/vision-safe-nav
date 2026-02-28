from ultralytics import YOLO
import cv2

from backend.safety.obstacle_check import is_crowded
from backend.utils.distance_estimator import is_dangerous
from backend.guidance.direction_helper import get_direction
from backend.voice.speaker import speak

model = YOLO("yolov8n.pt")

# Priority order (most urgent first)
PRIORITY_ORDER = ["ahead", "left", "right"]

def get_priority(direction):
    try:
        return PRIORITY_ORDER.index(direction)
    except ValueError:
        return 99


def start_detection():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_height, frame_width, _ = frame.shape
        results = model(frame, verbose=False)

        warnings = []
        person_count = 0

        # 🔍 Process detections
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                class_name = model.names[cls_id]

                if class_name == "person":
                    person_count += 1

                if class_name not in ["person", "chair", "couch", "bed", "dining table"]:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                coords = (x1, y1, x2, y2)

                if is_dangerous(coords):
                    direction = get_direction(coords, frame_width)
                    priority = get_priority(direction)

                    warnings.append((priority, direction, class_name))

        # 🚨 CROWD HAS HIGHEST PRIORITY
        if is_crowded(person_count):
            message = "Crowded area ahead"
            print(message)
            speak(message)

        # 🚨 Handle obstacle warnings
        elif warnings:
            warnings.sort(key=lambda w: w[0])
            _, top_direction, top_object = warnings[0]

            message = f"Obstacle on {top_direction}"
            print(message)
            speak(message)

        # Optional: clear path message (comment if too chatty)
        # else:
        #     speak("Path clear")

        annotated_frame = results[0].plot()
        cv2.imshow("Vision Safe Nav", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    start_detection()