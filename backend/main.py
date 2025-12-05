from flask import Flask, Response, jsonify, request
from flask_cors import CORS
import cv2
import numpy as np

app = Flask(__name__)

# Allow your Vite dev server (5173) to call this backend
CORS(app, resources={r"/*": {"origins": ["http://localhost:5173"]}})

# Open laptop camera (0). On Windows, you can also try: cv2.VideoCapture(0, cv2.CAP_DSHOW)
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("⚠️ Warning: Could not open camera 0")


# --- MAIN THRESHOLD CONTROL ---
# Increase → more strict (less false)
# Decrease → more sensitive (detects smaller flames)
FIRE_SENSITIVITY = 1.6  # Recommended range: 0.5 to 2.0


def detect_fire(frame):
    """
    Fire detection using HSV + contour rules.
    Returns list of bounding boxes (x, y, w, h).
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Adjust thresholds based on sensitivity
    min_sat = int(100 * FIRE_SENSITIVITY)    # controls flame color strength
    min_val = int(120 * FIRE_SENSITIVITY)    # controls brightness
    min_area = int(2000 * FIRE_SENSITIVITY)  # controls flame blob size

    # Clamp to valid HSV range
    min_sat = max(0, min(min_sat, 255))
    min_val = max(0, min(min_val, 255))

    # HSV Range for fire
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

        if area < min_area:  # controlled by FIRE_SENSITIVITY
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        height_to_width = h / float(w)
        if height_to_width < 0.5:
            # fire blobs are usually taller than wide
            continue

        fire_zones.append((x, y, w, h))

    return fire_zones


# ====== CAMERA STREAM (MJPEG video for <img src="...">) ======
def generate_frames():
    """
    Read frames from global `camera`, run fire detection,
    draw boxes/text, and stream as MJPEG.
    """
    while True:
        success, frame = camera.read()
        if not success:
            break

        # Run fire detection and draw overlay
        flame_boxes = detect_fire(frame)

        for (x, y, w, h) in flame_boxes:
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

        if flame_boxes:
            cv2.putText(
                frame,
                "FIRE DETECTED",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                3,
            )

        # Encode frame to JPEG
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
    Stream laptop camera as MJPEG with FIRE overlay.
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
    # For now, all 4 tiles show the same stream (/video_feed)
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
            "title": "Fire detected in Block A",
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
            "message": "Sentinal AI backend running",
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
