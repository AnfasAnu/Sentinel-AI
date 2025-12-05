from flask import Flask, Response, jsonify, request
from flask_cors import CORS
import cv2

app = Flask(__name__)

# Allow your Vite dev server (5173) to call this backend
CORS(app, resources={r"/*": {"origins": ["http://localhost:5173"]}})

# Open laptop camera (0). On Windows, you can also try: cv2.VideoCapture(0, cv2.CAP_DSHOW)
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("⚠️ Warning: Could not open camera 0")


# ====== CAMERA STREAM (MJPEG video for <img src="...">) ======
def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break

        # Optional: resize to reduce bandwidth
        # frame = cv2.resize(frame, (640, 360))

        ret, buffer = cv2.imencode(".jpg", frame)
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
    Stream laptop camera as MJPEG.
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
        "activeCameras": 1,      # currently only laptop cam
        "alertBadge": 12
    }
    return jsonify(stats)


# ====== API: CAMERAS ======
@app.route("/api/cameras")
def get_cameras():
    cameras = [
        {
            "id": "laptop-cam",
            "name": "Laptop Camera",
            "location": "Local Device",
            "status": "online",
            # React uses `thumbnail` as <img src="...">, so point to /video_feed
            "thumbnail": "http://localhost:5000/video_feed"
        },
        {
            "id": "laptop-cam",
            "name": "Laptop Camera",
            "location": "Local Device",
            "status": "online",
            # React uses `thumbnail` as <img src="...">, so point to /video_feed
            "thumbnail": "http://localhost:5000/video_feed"
        },
        {
            "id": "laptop-cam",
            "name": "Laptop Camera",
            "location": "Local Device",
            "status": "online",
            # React uses `thumbnail` as <img src="...">, so point to /video_feed
            "thumbnail": "http://localhost:5000/video_feed"
        },
        {
            "id": "laptop-cam",
            "name": "Laptop Camera",
            "location": "Local Device",
            "status": "online",
            # React uses `thumbnail` as <img src="...">, so point to /video_feed
            "thumbnail": "http://localhost:5000/video_feed"
        }
        # Add more cameras here later if you want
    ]
    return jsonify(cameras)


# ====== API: ALERTS ======
@app.route("/api/alerts")
def get_alerts():
    alerts = [
        {
            "id": "a1",
            "type": "critical",
            "title": "Fire detected in Block A",
            "camera": "Laptop Camera",
            "time": "16:42",
            "zone": "Entrance"
        },
        {
            "id": "a2",
            "type": "warning",
            "title": "Suspicious loitering detected",
            "camera": "Laptop Camera",
            "time": "16:35",
            "zone": "Parking"
        },
        {
            "id": "a3",
            "type": "warning",
            "title": "Unidentified object left unattended",
            "camera": "Laptop Camera",
            "time": "16:10",
            "zone": "Corridor"
        },
        {
            "id": "a4",
            "type": "info",
            "title": "New face added to watchlist",
            "camera": "Laptop Camera",
            "time": "15:50",
            "zone": "Lobby"
        },
    ]
    return jsonify(alerts)


# ====== API: THREATS ======
@app.route("/api/threats")
def get_threats():
    time_range = request.args.get("range", "today")  # today | week | month

    if time_range == "today":
        threats = [
            {"id": "t1", "label": "Fire / Smoke", "value": 3, "trend": "+1 vs yesterday"},
            {"id": "t2", "label": "Violence / Assault", "value": 1, "trend": "stable"},
            {"id": "t3", "label": "Weapons Detected", "value": 0, "trend": "no events"},
            {"id": "t4", "label": "Suspicious Activity", "value": 6, "trend": "+2 last hour"},
        ]
    elif time_range == "week":
        threats = [
            {"id": "t1", "label": "Fire / Smoke", "value": 11, "trend": "+3 vs last week"},
            {"id": "t2", "label": "Violence / Assault", "value": 4, "trend": "-1 vs last week"},
            {"id": "t3", "label": "Weapons Detected", "value": 2, "trend": "critical spike"},
            {"id": "t4", "label": "Suspicious Activity", "value": 27, "trend": "+5 vs last week"},
        ]
    else:  # month
        threats = [
            {"id": "t1", "label": "Fire / Smoke", "value": 34, "trend": "+8 vs last month"},
            {"id": "t2", "label": "Violence / Assault", "value": 17, "trend": "+2 vs last month"},
            {"id": "t3", "label": "Weapons Detected", "value": 7, "trend": "+3 vs last month"},
            {"id": "t4", "label": "Suspicious Activity", "value": 98, "trend": "+15 vs last month"},
        ]

    return jsonify(threats)


@app.route("/")
def index():
    return jsonify(
        {
            "message": "SafetyVision AI backend running",
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
        app.run(host="0.0.0.0", port=5000, debug=True)
    finally:
        if camera.isOpened():
            camera.release()
