from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

def start_detection():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)

        annotated_frame = results[0].plot()
        cv2.imshow("Vision Safe Nav", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_detection()