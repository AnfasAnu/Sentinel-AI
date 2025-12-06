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
from collections import deque
from datetime import datetime
from threading import Lock

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["http://localhost:5173"]}})

# ============================================================
#                 GLOBAL CONFIG / THRESHOLDS
# ============================================================

# ---------- HOW OFTEN TO RUN EACH DETECTOR ----------
# Smaller = more frequent = heavier load
FIRE_EVERY = 2        # run fire detection every N frames
GESTURE_EVERY = 3     # run Mediapipe every N frames
YOLO_EVERY = 5        # run YOLO every N frames (heaviest)

# ---------- FIRE DETECTION THRESHOLDS ----------
FIRE_SENSITIVITY = 1.8        # recommended ~0.8–2.0
FIRE_MIN_SAT_BASE = 100       # base saturation threshold
FIRE_MIN_VAL_BASE = 120       # base brightness threshold
FIRE_MIN_AREA_BASE = 2000     # base contour area (pixels)

# ---------- GESTURE (MEDIAPIPE) THRESHOLDS ----------
GESTURE_MIN_DET_CONF = 0.1
GESTURE_MIN_TRACK_CONF = 0.1

# ---------- YOLO WEAPON / THREAT THRESHOLDS ----------
YOLO_CONF_THRESHOLD = 0.70

# ---------- ALERT RATE LIMIT (in frames) ----------
ALERT_FIRE_MIN_FRAMES = 30      # min frames between fire alerts
ALERT_GESTURE_MIN_FRAMES = 60   # min frames between gesture alerts
ALERT_YOLO_MIN_FRAMES = 30      # min frames between weapon alerts

# ============================================================
#                        CAMERA SETUP
# ============================================================

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

if not camera.isOpened():
  print("⚠️ Warning: Could not open camera 0")

# ============================================================
#                ALERT STORAGE (GLOBAL, IN-MEMORY)
# ============================================================

MAX_ALERTS = 50
ALERTS = deque(maxlen=MAX_ALERTS)
ALERTS_LOCK = Lock()
next_alert_id = 1

# Track last frame index when we sent each alert type
last_fire_alert_frame = -999999
last_gesture_alert_frame = -999999
last_yolo_alert_frame = -999999


def add_alert(alert_type, title, camera_name="Laptop Camera 1", zone="Zone A"):
  """
  alert_type: "critical" | "warning" | "info"
  """
  global next_alert_id
  with ALERTS_LOCK:
    alert = {
      "id": f"a{next_alert_id}",
      "type": alert_type,
      "title": title,
      "camera": camera_name,
      "time": datetime.now().strftime("%H:%M:%S"),
      "zone": zone,
    }
    ALERTS.appendleft(alert)
    next_alert_id += 1


# ============================================================
#                   FIRE DETECTION SETUP
# ============================================================

def detect_fire(frame):
  hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

  # scale base thresholds by sensitivity
  min_sat = int(FIRE_MIN_SAT_BASE * FIRE_SENSITIVITY)
  min_val = int(FIRE_MIN_VAL_BASE * FIRE_SENSITIVITY)
  min_area = int(FIRE_MIN_AREA_BASE * FIRE_SENSITIVITY)

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
    if h_to_w < 0.5:  # flames usually taller than wide
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
  min_detection_confidence=GESTURE_MIN_DET_CONF,
  min_tracking_confidence=GESTURE_MIN_TRACK_CONF,
)


def is_thumbs_up(hand_landmarks):
  # Thumb tip (4) above MCP joint (2)
  thumb_up = hand_landmarks.landmark[4].y < hand_landmarks.landmark[2].y

  # Other fingers folded (tip below MCP)
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

  return thumbs_up_detected


# ============================================================
#                  YOLO WEAPON DETECTION
# ============================================================

MODEL_REPO = "Subh775/Threat-Detection-YOLOv8n"
MODEL_DIR = "models"
MODEL_LOCAL_PATH = os.path.join(MODEL_DIR, "best.pt")

device = 0 if torch.cuda.is_available() else "cpu"
print(f"Using device: {'GPU' if device == 0 else 'CPU'}")

if not os.path.exists(MODEL_LOCAL_PATH):
  print("Local YOLO model not found, downloading from HuggingFace.")
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
    conf=YOLO_CONF_THRESHOLD,
    device=device,
    verbose=False
  )
  last_yolo_result = results[0]

  # Determine if a "weapon-like" object is present
  weapon_detected = False
  other_threat_detected = False

  if last_yolo_result.boxes is not None and len(last_yolo_result.boxes) > 0:
    names = last_yolo_result.names  # class id -> name
    for box in last_yolo_result.boxes:
      cls_id = int(box.cls[0])
      cls_name = names.get(cls_id, str(cls_id)).lower()

      # heuristic for weapon-like classes
      if any(k in cls_name for k in ["gun", "pistol", "rifle", "weapon"]):
        weapon_detected = True
      else:
        other_threat_detected = True

  return weapon_detected, other_threat_detected


def draw_yolo(frame):
  # Use cached last_yolo_result to draw boxes
  if last_yolo_result is not None:
    plotted = last_yolo_result.plot()
    # Replace frame contents with YOLO drawn frame
    frame[:] = plotted


# ============================================================
#                CAMERA STREAM / FUSION PIPELINE
# ============================================================

def generate_frames():
  global last_fire_alert_frame, last_gesture_alert_frame, last_yolo_alert_frame

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
      if fire_boxes:
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

        cv2.putText(
          frame,
          "FIRE DETECTED",
          (30, 40),
          cv2.FONT_HERSHEY_SIMPLEX,
          1.0,
          (0, 0, 255),
          3,
        )

        # Rate-limited alert
        if frame_idx - last_fire_alert_frame >= ALERT_FIRE_MIN_FRAMES:
          add_alert(
            alert_type="critical",
            title="Fire pattern detected",
            camera_name="Laptop Camera 1",
            zone="Entrance",
          )
          last_fire_alert_frame = frame_idx

    # ---- GESTURE every GESTURE_EVERY frames ----
    if frame_idx % GESTURE_EVERY == 0:
      thumbs_up_detected = run_gesture(frame)

      if thumbs_up_detected and frame_idx - last_gesture_alert_frame >= ALERT_GESTURE_MIN_FRAMES:
        add_alert(
          alert_type="info",
          title="SOS gesture detected",
          camera_name="Laptop Camera 1",
          zone="Control Zone",
        )
        last_gesture_alert_frame = frame_idx

    # ---- YOLO every YOLO_EVERY frames ----
    if frame_idx % YOLO_EVERY == 0:
      weapon_detected, other_threat = run_yolo(frame)

      if weapon_detected and frame_idx - last_yolo_alert_frame >= ALERT_YOLO_MIN_FRAMES:
        add_alert(
          alert_type="critical",
          title="Weapon-like object detected",
          camera_name="Laptop Camera 1",
          zone="High-risk Zone",
        )
        last_yolo_alert_frame = frame_idx
      elif (not weapon_detected) and other_threat and frame_idx - last_yolo_alert_frame >= ALERT_YOLO_MIN_FRAMES:
        add_alert(
          alert_type="warning",
          title="Violent activity detected by YOLO",
          camera_name="Laptop Camera 1",
          zone="Monitored Area",
        )
        last_yolo_alert_frame = frame_idx

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


# ============================================================
#                JSON ENDPOINTS (DYNAMIC ALERTS)
# ============================================================

@app.route("/api/stats")
def get_stats():
  stats = {
    "criticalAlerts": sum(1 for a in ALERTS if a["type"] == "critical"),
    "warningAlerts": sum(1 for a in ALERTS if a["type"] == "warning"),
    "resolvedToday": 47,  # simple static example
    "detectionRate": 98.7,
    "activeCameras": 4,
    "alertBadge": len(ALERTS),
  }
  return jsonify(stats)


@app.route("/api/cameras")
def get_cameras():
  base_url = "http://localhost:5000/video_feed"
  cams = []
  for i in range(1, 2):
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
  with ALERTS_LOCK:
    return jsonify(list(ALERTS))


@app.route("/api/threats")
def get_threats():
  """
  Live summary of threats based on current ALERTS deque.
  Uses alert titles to bucket into: weapons, fire, gestures, suspicious.
  """
  time_range = request.args.get("range", "today")

  with ALERTS_LOCK:
    alerts_list = list(ALERTS)

  def count_where(fn):
    return sum(1 for a in alerts_list if fn(a))

  # Very simple text-based bucketing
  weapons = count_where(
    lambda a: any(
      key in a["title"].lower()
      for key in ["weapon", "gun", "pistol", "rifle"]
    )
  )
  fire = count_where(lambda a: "fire" in a["title"].lower())
  gestures = count_where(
    lambda a: "thumbs-up" in a["title"].lower()
    or "gesture" in a["title"].lower()
  )
  suspicious = count_where(lambda a: "suspicious" in a["title"].lower())

  range_label = {
    "today": "live (today)",
    "week": "live (this week)",
    "month": "live (this month)",
  }.get(time_range, f"live ({time_range})")

  threats = [
    {
      "id": "t1",
      "label": "Weapons Detected",
      "value": weapons,
      "trend": range_label,
    },
    {
      "id": "t2",
      "label": "Fire / Smoke",
      "value": fire,
      "trend": range_label,
    },
    {
      "id": "t3",
      "label": "Suspicious Gestures",
      "value": gestures,
      "trend": range_label,
    },
    {
      "id": "t4",
      "label": "Suspicious Activity",
      "value": suspicious,
      "trend": range_label,
    },
  ]

  return jsonify(threats)


@app.route("/")
def index():
  return jsonify(
    {
      "message": "Sentinal AI backend running (multi-threat mode)",
      "device": "GPU" if device == 0 else "CPU",
      "model_repo": MODEL_REPO,
      "model_path": MODEL_LOCAL_PATH,
      "config": {
        "FIRE_EVERY": FIRE_EVERY,
        "GESTURE_EVERY": GESTURE_EVERY,
        "YOLO_EVERY": YOLO_EVERY,
        "FIRE_SENSITIVITY": FIRE_SENSITIVITY,
        "YOLO_CONF_THRESHOLD": YOLO_CONF_THRESHOLD,
        "GESTURE_MIN_DET_CONF": GESTURE_MIN_DET_CONF,
        "GESTURE_MIN_TRACK_CONF": GESTURE_MIN_TRACK_CONF,
        "ALERT_FIRE_MIN_FRAMES": ALERT_FIRE_MIN_FRAMES,
        "ALERT_GESTURE_MIN_FRAMES": ALERT_GESTURE_MIN_FRAMES,
        "ALERT_YOLO_MIN_FRAMES": ALERT_YOLO_MIN_FRAMES,
      },
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
