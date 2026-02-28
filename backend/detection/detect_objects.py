from ultralytics import YOLO
import cv2

from backend.utils.distance_estimator import is_dangerous
from backend.guidance.direction_helper import get_direction

model = YOLO("yolov8n.pt")

def start_detection():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_height, frame_width, _ = frame.shape

        results = model(frame, verbose=False)

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                class_name = model.names[cls_id]

                # Only important navigation objects
                if class_name not in ["person", "chair", "couch", "bed", "dining table"]:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                coords = (x1, y1, x2, y2)

                if is_dangerous(coords):
                    direction = get_direction(coords, frame_width)
                    print(f"Obstacle on {direction}: {class_name}")

        annotated_frame = results[0].plot()
        cv2.imshow("Vision Safe Nav", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_detection()