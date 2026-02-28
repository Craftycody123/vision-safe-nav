from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import cv2
import threading
import numpy as np
from ultralytics import YOLO

from backend.safety.hazard_check import detect_low_visibility
from backend.safety.obstacle_check import is_crowded
from backend.utils.distance_estimator import is_dangerous
from backend.guidance.direction_helper import get_direction
from backend.voice.speaker import speak


# ────────────────────────────────────────────────────────────────
# App Setup
# ────────────────────────────────────────────────────────────────

app = FastAPI(title="Vision Safe Nav API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="frontend/public"), name="static")

model = YOLO("backend/models/yolov8n.pt")

TRACKED_CLASSES = ["person", "chair", "couch", "bed", "dining table"]
PRIORITY_ORDER = ["ahead", "left", "right"]


# ────────────────────────────────────────────────────────────────
# Shared Detection State
# ────────────────────────────────────────────────────────────────

detection_state = {
    "running": False,
    "warnings": [],
    "frame": None,
}

state_lock = threading.Lock()
detection_thread = None


# ────────────────────────────────────────────────────────────────
# Utility Functions
# ────────────────────────────────────────────────────────────────

def get_priority(direction):
    try:
        return PRIORITY_ORDER.index(direction)
    except ValueError:
        return 99


# ────────────────────────────────────────────────────────────────
# Detection Loop
# ────────────────────────────────────────────────────────────────

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
        person_count = 0

        # ── Object Detection ───────────────────────────────
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                class_name = model.names[cls_id]

                # Count persons for crowd detection
                if class_name == "person":
                    person_count += 1

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

        # ── Crowd Detection ────────────────────────────────
        if is_crowded(person_count):
            warnings.append({
                "object": "crowded area",
                "direction": "ahead",
                "priority": get_priority("ahead")
            })

        # ── Low Visibility Detection ───────────────────────
        if detect_low_visibility(frame):
            warnings.append({
                "object": "low visibility",
                "direction": "ahead",
                "priority": -1  # Highest priority
            })

        # ── Priority Sorting ───────────────────────────────
        warnings.sort(key=lambda w: w["priority"])

        # ── Voice Output ────────────────────────────────────
        if warnings:
            top = warnings[0]

            if top["object"] == "crowded area":
                message = "Crowded area ahead"
            elif top["object"] == "low visibility":
                message = "Low visibility detected"
            else:
                message = f"{top['object']} on {top['direction']}"

            speak(message)
        else:
            speak("Path clear")

        # ── Frame Encoding for Streaming ───────────────────
        annotated = results[0].plot()
        _, buffer = cv2.imencode(".jpg", annotated)

        with state_lock:
            detection_state["warnings"] = warnings
            detection_state["frame"] = buffer.tobytes()

    cap.release()


# ────────────────────────────────────────────────────────────────
# Frontend Serving
# ────────────────────────────────────────────────────────────────

@app.get("/")
def serve_frontend():
    return FileResponse("frontend/public/index.html")


# ────────────────────────────────────────────────────────────────
# API Routes
# ────────────────────────────────────────────────────────────────

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


# ────────────────────────────────────────────────────────────────
# Video Streaming Endpoint
# ────────────────────────────────────────────────────────────────

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