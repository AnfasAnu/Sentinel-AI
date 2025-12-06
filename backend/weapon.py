print("starting weapon detection backend")

from flask import Flask, Response, jsonify, request
from flask_cors import CORS
import cv2
import torch
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import os
import shutil

app = Flask(__name__)

# Allow your Vite dev server (5173) to call this backend
CORS(app, resources={r"/*": {"origins": ["http://localhost:5173"]}})

# ---------- CAMERA SETUP ----------
# On Windows, CAP_DSHOW often opens faster and more reliably
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Try to reduce resolution to make inference faster
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

if not camera.isOpened():
    print("⚠️ Warning: Could not open camera 0")

# ====== YOLO WEAPON / THREAT DETECTION SETUP ======

MODEL_REPO = "Subh775/Threat-Detection-YOLOv8n"
CONFIDENCE_THRESHOLD = 0.4

# Local path where we keep a copy of the model (so no re-download on 2nd run)
MODEL_DIR = "models"
MODEL_LOCAL_PATH = os.path.join(MODEL_DIR, "best.pt")

# Setup device
device = 0 if torch.cuda.is_available() else "cpu"
print(f"Using device: {'GPU' if device == 0 else 'CPU'}")

# Ensure model exists locally
if not os.path.exists(MODEL_LOCAL_PATH):
    print("Local model not found, downloading from HuggingFace...")
    os.makedirs(MODEL_DIR, exist_ok=True)

    # This will still use HF cache, so even this call is fast on 2nd+ run
    downloaded_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename="weights/best.pt"
    )

    # Copy from HF cache to your project folder
    shutil.copy(downloaded_path, MODEL_LOCAL_PATH)
    print(f"Model copied to {MODEL_LOCAL_PATH}")
else:
    print(f"Using existing local model: {MODEL_LOCAL_PATH}")

# Load YOLO from local file (no network)
model = YOLO(MODEL_LOCAL_PATH)
print("YOLO model ready")


# ====== CAMERA STREAM (MJPEG video for <img src="...">) ======
def generate_frames():
    """
    Read frames from global `camera`, run YOLO threat/weapon detection,
    draw bounding boxes/labels, and stream as MJPEG.
    """
    while True:
        success, frame = camera.read()
        if not success:
            break

        # Optional: flip horizontally for CCTV style or more natural webcam view
        frame = cv2.flip(frame, 1)

        # Run YOLO inference on this frame
        results = model(
            frame,
            conf=CONFIDENCE_THRESHOLD,
            device=device,
            verbose=False
        )

        result = results[0]
        annotated_frame = result.plot()  # returns a numpy array (BGR)

        # Encode frame to JPEG
        ret, buffer = cv2.imencode(".jpg", annotated_frame)
        if not ret:
            continue

        frame_bytes = buffer.tobytes()

        # MJPEG stream
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )


@app.route("/video_feed")
def video_feed():
    """
    Stream laptop camera as MJPEG with YOLO threat/weapon overlays.
    Use this URL directly in an <img> tag in React.
    """
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# ====== API: STATS ======
@app.route("/api/stats")
def get_stats():
    stats = {
        "criticalAlerts": 3,
        "warningAlerts": 9,
        "resolvedToday": 47,
        "detectionRate": 98.7,   # frontend turns this into "98.7%"
        "activeCameras": 4,      # we expose 4 mock cameras
        "alertBadge": 12
    }
    return jsonify(stats)


# ====== API: CAMERAS ======
@app.route("/api/cameras")
def get_cameras():
    # All tiles show the same YOLO-annotated stream (/video_feed)
    
    cameras = [
        {
            "id": "cam-1",
            "name": "Laptop Camera 1",
            "location": "Local Device",
            "status": "online",
            "thumbnail": "http://localhost:5000/video_feed",
        },
        {
            "id": "cam-2",
            "name": "Laptop Camera 2",
            "location": "Local Device",
            "status": "online",
            "thumbnail": "http://localhost:5000/video_feed",
        },
        {
            "id": "cam-3",
            "name": "Laptop Camera 3",
            "location": "Local Device",
            "status": "online",
            "thumbnail": "http://localhost:5000/video_feed",
        },
        {
            "id": "cam-4",
            "name": "Laptop Camera 4",
            "location": "Local Device",
            "status": "online",
            "thumbnail": "http://localhost:5000/video_feed",
        },
    ]
    return jsonify(cameras)


# ====== API: ALERTS ======
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
            "title": "Suspicious loitering detected",
            "camera": "Laptop Camera 2",
            "time": "16:35",
            "zone": "Parking",
        },
        {
            "id": "a3",
            "type": "warning",
            "title": "Unidentified object left unattended",
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


# ====== API: THREATS ======
@app.route("/api/threats")
def get_threats():
    time_range = request.args.get("range", "today")  # today | week | month

    if time_range == "today":
        threats = [
            {"id": "t1", "label": "Weapons Detected", "value": 1, "trend": "+1 vs yesterday"},
            {"id": "t2", "label": "Violence / Assault", "value": 1, "trend": "stable"},
            {"id": "t3", "label": "Fire / Smoke", "value": 0, "trend": "no events"},
            {"id": "t4", "label": "Suspicious Activity", "value": 6, "trend": "+2 last hour"},
        ]
    elif time_range == "week":
        threats = [
            {"id": "t1", "label": "Weapons Detected", "value": 4, "trend": "+2 vs last week"},
            {"id": "t2", "label": "Violence / Assault", "value": 4, "trend": "-1 vs last week"},
            {"id": "t3", "label": "Fire / Smoke", "value": 2, "trend": "stable"},
            {"id": "t4", "label": "Suspicious Activity", "value": 27, "trend": "+5 vs last week"},
        ]
    else:  # month
        threats = [
            {"id": "t1", "label": "Weapons Detected", "value": 9, "trend": "+3 vs last month"},
            {"id": "t2", "label": "Violence / Assault", "value": 17, "trend": "+2 vs last month"},
            {"id": "t3", "label": "Fire / Smoke", "value": 7, "trend": "+3 vs last month"},
            {"id": "t4", "label": "Suspicious Activity", "value": 98, "trend": "+15 vs last month"},
        ]

    return jsonify(threats)


@app.route("/")
def index():
    return jsonify(
        {
            "message": "Sentinal AI backend running (YOLO weapon detection mode)",
            "device": "GPU" if device == 0 else "CPU",
            "model_repo": MODEL_REPO,
            "model_path": MODEL_LOCAL_PATH,
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
        # Turn OFF debug so Flask doesn't run the app twice
        app.run(host="0.0.0.0", port=5000, debug=False)
    finally:
        if camera.isOpened():
            camera.release()
