print("starting multi-threat backend")

from flask import Flask, Response, jsonify, request
from flask_cors import CORS
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import os
import shutil
import mediapipe as mp

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["http://localhost:5173"]}})

# ---------- CAMERA ----------
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

if not camera.isOpened():
    print("⚠️ Warning: Could not open camera 0")

# ---------- CONFIG: HOW OFTEN TO RUN EACH DETECTOR ----------
FIRE_EVERY = 1        # run fire detection every frame (it's cheap)
GESTURE_EVERY = 2     # run Mediapipe every 2nd frame
YOLO_EVERY = 5        # run YOLO every 5th frame  (heaviest)

CONFIDENCE_THRESHOLD = 0.4

# ============================================================
#                   FIRE DETECTION SETUP
# ============================================================

FIRE_SENSITIVITY = 1.6  # you used this earlier


def detect_fire(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    min_sat = int(100 * FIRE_SENSITIVITY)
    min_val = int(120 * FIRE_SENSITIVITY)
    min_area = int(2000 * FIRE_SENSITIVITY)

    min_sat = max(0, min(min_sat, 255))
    min_val = max(0, min(min_val, 255))

    lower = np.array([0, min_sat, min_val])
    upper = np.array([60, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    fire_zones = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        h_to_w = h / float(w)
        if h_to_w < 0.5:
            continue

        fire_zones.append((x, y, w, h))

    return fire_zones


# ============================================================
#                GESTURE (THUMBS UP) SETUP
# ============================================================

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)


def is_thumbs_up(hand_landmarks):
    thumb_up = hand_landmarks.landmark[4].y < hand_landmarks.landmark[2].y

    folded_fingers = True
    finger_tips = [8, 12, 16, 20]
    finger_mcps = [5, 9, 13, 17]

    for tip, mcp in zip(finger_tips, finger_mcps):
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[mcp].y:
            folded_fingers = False
            break

    return thumb_up and folded_fingers


def run_gesture(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    thumbs_up_detected = False

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )
            if is_thumbs_up(hand_landmarks):
                thumbs_up_detected = True

    if thumbs_up_detected:
        cv2.putText(
            frame,
            "THUMBS UP!",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )


# ============================================================
#                  YOLO WEAPON DETECTION
# ============================================================

MODEL_REPO = "Subh775/Threat-Detection-YOLOv8n"
MODEL_DIR = "models"
MODEL_LOCAL_PATH = os.path.join(MODEL_DIR, "best.pt")

device = 0 if torch.cuda.is_available() else "cpu"
print(f"Using device: {'GPU' if device == 0 else 'CPU'}")

if not os.path.exists(MODEL_LOCAL_PATH):
    print("Local YOLO model not found, downloading from HuggingFace...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    downloaded_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename="weights/best.pt"
    )
    shutil.copy(downloaded_path, MODEL_LOCAL_PATH)
    print(f"Model copied to {MODEL_LOCAL_PATH}")
else:
    print(f"Using existing local model: {MODEL_LOCAL_PATH}")

model = YOLO(MODEL_LOCAL_PATH)
print("YOLO model ready")

last_yolo_result = None  # cached detections from last YOLO run


def run_yolo(frame):
    global last_yolo_result

    results = model(
        frame,
        conf=CONFIDENCE_THRESHOLD,
        device=device,
        verbose=False
    )
    last_yolo_result = results[0]


def draw_yolo(frame):
    # use cached last_yolo_result to draw boxes
    if last_yolo_result is not None:
        plotted = last_yolo_result.plot()
        # plotted is a new image with drawings,
        # but we want to keep any fire/gesture overlays too.
        # So instead, we can copy the drawings back onto "frame".
        # Easiest: just replace frame:
        frame[:] = plotted


# ============================================================
#                CAMERA STREAM / FUSION PIPELINE
# ============================================================

def generate_frames():
    frame_idx = 0

    while True:
        success, frame = camera.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        frame_idx += 1

        # ---- FIRE every FIRE_EVERY frames ----
        if frame_idx % FIRE_EVERY == 0:
            fire_boxes = detect_fire(frame)
            for (x, y, w, h) in fire_boxes:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(
                    frame,
                    "FIRE",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )
            if fire_boxes:
                cv2.putText(
                    frame,
                    "FIRE DETECTED",
                    (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    3,
                )

        # ---- GESTURE every GESTURE_EVERY frames ----
        if frame_idx % GESTURE_EVERY == 0:
            run_gesture(frame)

        # ---- YOLO every YOLO_EVERY frames ----
        if frame_idx % YOLO_EVERY == 0:
            run_yolo(frame)

        # Always draw last YOLO result (even if not recomputed this frame)
        draw_yolo(frame)

        # ---- ENCODE & STREAM ----
        ret, buffer = cv2.imencode(".jpg", frame)
        if not ret:
            continue

        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# ====== (same API endpoints you already had) ======
@app.route("/api/stats")
def get_stats():
    stats = {
        "criticalAlerts": 3,
        "warningAlerts": 9,
        "resolvedToday": 47,
        "detectionRate": 98.7,
        "activeCameras": 4,
        "alertBadge": 12,
    }
    return jsonify(stats)


@app.route("/api/cameras")
def get_cameras():
    base_url = "http://localhost:5000/video_feed"
    cams = []
    for i in range(1, 5):
        cams.append(
            {
                "id": f"cam-{i}",
                "name": f"Laptop Camera {i}",
                "location": "Local Device",
                "status": "online",
                "thumbnail": base_url,
            }
        )
    return jsonify(cams)


@app.route("/api/alerts")
def get_alerts():
    alerts = [
        {
            "id": "a1",
            "type": "critical",
            "title": "Weapon-like object detected",
            "camera": "Laptop Camera 1",
            "time": "16:42",
            "zone": "Entrance",
        },
        {
            "id": "a2",
            "type": "warning",
            "title": "Fire pattern detected",
            "camera": "Laptop Camera 2",
            "time": "16:35",
            "zone": "Parking",
        },
        {
            "id": "a3",
            "type": "warning",
            "title": "Suspicious gesture detected",
            "camera": "Laptop Camera 3",
            "time": "16:10",
            "zone": "Corridor",
        },
        {
            "id": "a4",
            "type": "info",
            "title": "New face added to watchlist",
            "camera": "Laptop Camera 4",
            "time": "15:50",
            "zone": "Lobby",
        },
    ]
    return jsonify(alerts)


@app.route("/api/threats")
def get_threats():
    time_range = request.args.get("range", "today")

    if time_range == "today":
        threats = [
            {"id": "t1", "label": "Weapons Detected", "value": 1, "trend": "+1 vs yesterday"},
            {"id": "t2", "label": "Fire / Smoke", "value": 1, "trend": "stable"},
            {"id": "t3", "label": "Suspicious Gestures", "value": 2, "trend": "+1 last hour"},
            {"id": "t4", "label": "Suspicious Activity", "value": 6, "trend": "+2 last hour"},
        ]
    else:
        threats = [
            {"id": "t1", "label": "Weapons Detected", "value": 4, "trend": "+2 vs last week"},
            {"id": "t2", "label": "Fire / Smoke", "value": 3, "trend": "+1 vs last week"},
            {"id": "t3", "label": "Suspicious Gestures", "value": 7, "trend": "+3 vs last week"},
            {"id": "t4", "label": "Suspicious Activity", "value": 27, "trend": "+5 vs last week"},
        ]
    return jsonify(threats)


@app.route("/")
def index():
    return jsonify(
        {
            "message": "Sentinal AI backend running (multi-threat mode)",
            "device": "GPU" if device == 0 else "CPU",
            "model_repo": MODEL_REPO,
            "endpoints": [
                "/video_feed",
                "/api/stats",
                "/api/cameras",
                "/api/alerts",
                "/api/threats?range=today|week|month",
            ],
        }
    )


if __name__ == "__main__":
    try:
        app.run(host="0.0.0.0", port=5000, debug=False)
    finally:
        if camera.isOpened():
            camera.release()
        hands.close()
