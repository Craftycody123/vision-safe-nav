from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import cv2
import threading
import os
from ultralytics import YOLO

from backend.utils.distance_estimator import is_dangerous
from backend.guidance.direction_helper import get_direction
from backend.voice.speaker import speak

app = FastAPI(title="Vision Safe Nav API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = YOLO("backend/models/yolov8n.pt")

TRACKED_CLASSES = ["person", "chair", "couch", "bed", "dining table"]
PRIORITY_ORDER = ["ahead", "left", "right"]

# Shared state
detection_state = {
    "running": False,
    "warnings": [],
    "frame": None,
}
state_lock = threading.Lock()
detection_thread = None


def get_priority(direction):
    try:
        return PRIORITY_ORDER.index(direction)
    except ValueError:
        return 99


def detection_loop():
    cap = cv2.VideoCapture(0)

    while True:
        with state_lock:
            if not detection_state["running"]:
                break

        ret, frame = cap.read()
        if not ret:
            break

        frame_height, frame_width, _ = frame.shape
        results = model(frame, verbose=False)

        warnings = []

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                class_name = model.names[cls_id]

                if class_name not in TRACKED_CLASSES:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                coords = (x1, y1, x2, y2)

                if is_dangerous(coords):
                    direction = get_direction(coords, frame_width)
                    warnings.append({
                        "object": class_name,
                        "direction": direction,
                        "priority": get_priority(direction)
                    })

        warnings.sort(key=lambda w: w["priority"])

        if warnings:
            top = warnings[0]
            message = f"{top['object']} {top['direction']}"
            speak(message)
        else:
            speak("path clear")

        # Encode frame for streaming
        annotated = results[0].plot()
        _, buffer = cv2.imencode(".jpg", annotated)

        with state_lock:
            detection_state["warnings"] = warnings
            detection_state["frame"] = buffer.tobytes()

    cap.release()


# ── Serve Frontend ─────────────────────────────────────────────────────────────

app.mount("/static", StaticFiles(directory="frontend/public"), name="static")

@app.get("/")
def serve_frontend():
    return FileResponse("frontend/public/index.html")


# ── API Routes ─────────────────────────────────────────────────────────────────

@app.post("/start")
def start_detection():
    global detection_thread

    with state_lock:
        if detection_state["running"]:
            return JSONResponse({"status": "already running"}, status_code=200)
        detection_state["running"] = True

    detection_thread = threading.Thread(target=detection_loop, daemon=True)
    detection_thread.start()
    return {"status": "detection started"}


@app.post("/stop")
def stop_detection():
    with state_lock:
        detection_state["running"] = False
        detection_state["warnings"] = []
        detection_state["frame"] = None
    return {"status": "detection stopped"}


@app.get("/status")
def get_status():
    with state_lock:
        return {
            "running": detection_state["running"],
            "warnings": detection_state["warnings"],
        }


def generate_frames():
    while True:
        with state_lock:
            frame = detection_state["frame"]
            running = detection_state["running"]

        if not running:
            break
        if frame is None:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        )


@app.get("/video")
def video_feed():
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )