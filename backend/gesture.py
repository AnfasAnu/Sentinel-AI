print("starting gesture backend")

from flask import Flask, Response, jsonify, request
from flask_cors import CORS
import cv2

# 👍 Mediapipe imports for gesture detection
import mediapipe as mp

app = Flask(__name__)

# Allow your Vite dev server (5173) to call this backend
CORS(app, resources={r"/*": {"origins": ["http://localhost:5173"]}})

# ---------- CAMERA SETUP ----------
# On Windows, CAP_DSHOW often opens faster and more reliably
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Try to reduce resolution to make Mediapipe faster
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

if not camera.isOpened():
    print("⚠️ Warning: Could not open camera 0")

# ====== HAND / GESTURE (THUMBS UP) SETUP ======

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

print("initializing mediapipe Hands...")

# Create a single global Hands instance (don’t recreate per frame)
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

print("mediapipe Hands ready")


def is_thumbs_up(hand_landmarks):
    """
    Returns True if thumb is up based on relative landmark positions
    (Right-hand model logic).
    """
    # Thumb tip (4) should be above its MCP joint (2)
    thumb_up = hand_landmarks.landmark[4].y < hand_landmarks.landmark[2].y

    # Other four fingers (tips) should be below their MCP joints (folded)
    folded_fingers = True
    finger_tips = [8, 12, 16, 20]   # index, middle, ring, little
    finger_mcps = [5, 9, 13, 17]

    for tip, mcp in zip(finger_tips, finger_mcps):
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[mcp].y:
            folded_fingers = False
            break

    return thumb_up and folded_fingers


# ====== CAMERA STREAM (MJPEG video for <img src="...">) ======
def generate_frames():
    """
    Read frames from global `camera`, run thumbs-up detection,
    draw skeleton + 'THUMBS UP!' text, and stream as MJPEG.
    """
    frame_idx = 0

    while True:
        success, frame = camera.read()
        if not success:
            break

        # Mirror view for more natural gesture detection
        frame = cv2.flip(frame, 1)

        # Optional: skip every 2nd frame to reduce CPU load
        frame_idx += 1
        process_this_frame = (frame_idx % 2 == 0)

        if process_this_frame:
            # Convert to RGB for Mediapipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    # Draw hand skeleton
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS
                    )

                    # Check thumbs-up
                    if is_thumbs_up(hand_landmarks):
                        cv2.putText(
                            frame,
                            "THUMBS UP!",
                            (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2,
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
    Stream laptop camera as MJPEG with thumbs-up overlay.
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
            "message": "Sentinal AI backend running (gesture mode)",
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
        # Properly close Mediapipe
        hands.close()
