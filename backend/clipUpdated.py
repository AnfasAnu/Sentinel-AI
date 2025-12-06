import os
import time
import csv
from collections import defaultdict, deque
from datetime import datetime
from threading import Lock

import cv2
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO
from flask import Flask, Response, jsonify
from flask_cors import CORS

# ============================================================
#                 FLASK APP SETUP
# ============================================================
app = Flask(__name__)
# allow your Vite frontend (port 5173) to access this backend
CORS(app, resources={r"/*": {"origins": ["http://localhost:5173"]}})

# ============================================================
#                 DEVICE / GLOBAL CONFIG
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

COOLDOWN_SEC = 5.0
PROB_THRESHOLD = 0.60

WEAPON_KEYWORDS = ["knife", "gun", "pistol", "rifle", "weapon", "revolver", "shotgun"]

# ============================================================
#              ALERT STORAGE (SCVD-ONLY)
# ============================================================
MAX_ALERTS = 50
ALERTS = deque(maxlen=MAX_ALERTS)
ALERTS_LOCK = Lock()
next_alert_id = 1

def add_alert(alert_type, title, camera_name="SCVD Stream 1", zone="SCVD Zone"):
    """
    alert_type: "critical" | "warning" | "info"
    """
    global next_alert_id
    with ALERTS_LOCK:
        alert = {
            "id": f"scvd-{next_alert_id}",
            "type": alert_type,
            "title": title,
            "camera": camera_name,
            "time": datetime.now().strftime("%H:%M:%S"),
            "zone": zone,
        }
        ALERTS.appendleft(alert)
        next_alert_id += 1

# ============================================================
#                 C3D MODEL
# ============================================================
class C3D(nn.Module):
    def __init__(self, num_classes=2):
        super(C3D, self).__init__()
        self.features = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),

            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),

            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),

            nn.Conv3d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7 * 1, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ============================================================
#                UTILITIES
# ============================================================
def safe_crop(img, x1, y1, x2, y2, pad=8):
    h, w = img.shape[:2]
    x1 = max(0, int(x1 - pad))
    y1 = max(0, int(y1 - pad))
    x2 = min(w, int(x2 + pad))
    y2 = min(h, int(y2 + pad))
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2]

def draw_label(img, text, x, y, color, font_scale=0.6, thickness=2):
    cv2.putText(img, text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, color, thickness, cv2.LINE_AA)

# ============================================================
#              LOAD MODELS (GLOBAL)
# ============================================================
# Person tracking YOLO
yolo_person = YOLO("yolov8n.pt")

# Weapon YOLO (fake COCO fallback)
weapon_model_path = "weapon_best.pt"
if os.path.exists(weapon_model_path):
    print(f"[INFO] Loading REAL weapon model from: {weapon_model_path}")
    yolo_weapon = YOLO(weapon_model_path)
    using_fake_weapon_model = False
else:
    print("[WARNING] weapon_best.pt not found.")
    print("[INFO] Using yolov8n.pt as TEMP weapon model (COCO).")
    print("       This will mainly detect KNIFE as a weapon.")
    yolo_weapon = YOLO("yolov8n.pt")
    using_fake_weapon_model = True

# C3D
c3d = C3D(num_classes=2).to(device)
state = torch.load("checkpoints/best_c3d.pth", map_location=device)
c3d.load_state_dict(state)
c3d.eval()
softmax = nn.Softmax(dim=1)

# ============================================================
#              STREAMING PIPELINE
# ============================================================
def generate_frames():
    clip_len = 16
    stride = 1
    input_size = 112
    smooth_k = 3

    frame_buffers = defaultdict(lambda: deque(maxlen=clip_len))
    prob_buffers = defaultdict(lambda: deque(maxlen=smooth_k))
    last_track_alert_time = {}

    # logging
    os.makedirs("alerts", exist_ok=True)
    log_csv = os.path.join("alerts", "fight_events.csv")
    file_new = not os.path.exists(log_csv)
    log_f = open(log_csv, "a", newline="", encoding="utf-8")
    logger = csv.writer(log_f)
    if file_new:
        logger.writerow([
            "timestamp", "track_id", "prob_fight",
            "x1", "y1", "x2", "y2", "source", "weapons"
        ])

    source = "fight.mp4"  # change to 0 to use webcam

    try:
        frame_idx = 0

        # 🔁 LOOP THE VIDEO FOREVER
        while True:
            # recreate YOLO stream from the beginning of the file
            stream = yolo_person.track(
                source=source,
                stream=True,
                tracker="bytetrack.yaml",
                classes=[0],
                conf=0.25,
                iou=0.45,
                verbose=False
            )

            for result in stream:
                frame = result.orig_img
                if frame is None:
                    continue

                frame_idx += 1

                # ------------- WEAPON DETECTION -------------
                weapon_dets = []
                weapon_results = yolo_weapon(frame, conf=0.35, iou=0.45, verbose=False)[0]
                weapon_boxes = weapon_results.boxes

                if weapon_boxes is not None and len(weapon_boxes) > 0:
                    w_xyxy = weapon_boxes.xyxy.cpu().numpy()
                    w_cls = weapon_boxes.cls.cpu().numpy().astype(int)
                    names = weapon_results.names

                    for (wx1, wy1, wx2, wy2), cid in zip(w_xyxy, w_cls):
                        raw_name = names[int(cid)]
                        name_lower = raw_name.lower()

                        is_weapon = any(k in name_lower for k in WEAPON_KEYWORDS)
                        if not is_weapon and using_fake_weapon_model:
                            continue

                        wlabel = raw_name.upper()
                        weapon_dets.append({
                            "x1": float(wx1),
                            "y1": float(wy1),
                            "x2": float(wx2),
                            "y2": float(wy2),
                            "label": wlabel
                        })

                        cv2.rectangle(
                            frame,
                            (int(wx1), int(wy1)),
                            (int(wx2), int(wy2)),
                            (255, 0, 0),
                            2
                        )
                        draw_label(frame, wlabel, int(wx1), max(0, int(wy1) - 5), (255, 0, 0))

                # ------------- PERSON + C3D -------------
                boxes = result.boxes
                if boxes is None or boxes.id is None or len(boxes) == 0:
                    # still stream the frame
                    ret, buffer = cv2.imencode(".jpg", frame)
                    if not ret:
                        continue
                    frame_bytes = buffer.tobytes()
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                    )
                    continue

                ids = boxes.id.cpu().numpy().astype(int)
                xyxy = boxes.xyxy.cpu().numpy()

                frame_has_fight = False  # for frame-level alert

                for (x1, y1, x2, y2), tid in zip(xyxy, ids):
                    person = safe_crop(frame, x1, y1, x2, y2, pad=10)
                    if person is None:
                        continue

                    if frame_idx % stride == 0:
                        resized = cv2.resize(person, (input_size, input_size))
                        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                        frame_buffers[tid].append(rgb)

                    prob_fight = None
                    if len(frame_buffers[tid]) == clip_len:
                        clip = np.stack(list(frame_buffers[tid]), axis=0)
                        clip = np.transpose(clip, (3, 0, 1, 2)).astype(np.float32) / 255.0
                        tens = torch.tensor(clip, dtype=torch.float32).unsqueeze(0).to(device)

                        with torch.no_grad():
                            logits = c3d(tens)
                            probs = softmax(logits).cpu().numpy()[0]
                            prob_fight = float(probs[1])

                        prob_buffers[tid].append(prob_fight)
                        prob_fight = float(np.mean(prob_buffers[tid]))

                    # weapons inside this person box
                    person_weapons = []
                    px1, py1, px2, py2 = float(x1), float(y1), float(x2), float(y2)
                    for w in weapon_dets:
                        cx = (w["x1"] + w["x2"]) / 2.0
                        cy = (w["y1"] + w["y2"]) / 2.0
                        if (px1 <= cx <= px2) and (py1 <= cy <= py2):
                            person_weapons.append(w["label"])

                    person_weapons = list(set(person_weapons))
                    weapons_str = ""
                    if len(person_weapons) > 0:
                        weapons_str = " | " + ",".join(person_weapons)

                    # label & color
                    label = f"ID {tid}"
                    color = (0, 200, 0)

                    if prob_fight is not None:
                        label = f"ID {tid} | FIGHT {prob_fight:.2f}"
                        red = int(255 * min(1.0, prob_fight))
                        green = int(255 * (1.0 - min(1.0, prob_fight)))
                        color = (0, green, red)

                        # mark frame as having violence
                        if prob_fight >= PROB_THRESHOLD:
                            frame_has_fight = True

                        now_t = time.time()
                        last_t_for_track = last_track_alert_time.get(tid, 0.0)
                        if prob_fight >= PROB_THRESHOLD and (now_t - last_t_for_track) >= COOLDOWN_SEC:
                            last_track_alert_time[tid] = now_t
                            ts = time.strftime("%Y%m%d_%H%M%S")
                            snap_path = os.path.join("alerts", f"fight_tid{tid}_{ts}.jpg")
                            cv2.imwrite(snap_path, frame)
                            logger.writerow([
                                time.strftime("%Y-%m-%d %H:%M:%S"), tid,
                                f"{prob_fight:.3f}",
                                int(x1), int(y1), int(x2), int(y2),
                                str(source),
                                ";".join(person_weapons)
                            ])
                            log_f.flush()
                            cv2.putText(
                                frame,
                                "ALERT: FIGHT DETECTED!",
                                (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1.0,
                                (0, 0, 255),
                                3,
                            )

                    label = label + weapons_str

                    cv2.rectangle(
                        frame,
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        color,
                        2
                    )
                    draw_label(frame, label, int(x1), max(0, int(y1) - 8), color)

                # ---- FRAME-LEVEL ALERT TO FRONTEND (SCVD) ----
                if frame_has_fight:
                    add_alert(
                        alert_type="critical",
                        title="Violence detected (SCVD C3D)",
                        camera_name="SCVD Stream 1",
                        zone="SCVD Zone",
                    )

                # encode + stream
                ret, buffer = cv2.imencode(".jpg", frame)
                if not ret:
                    continue
                frame_bytes = buffer.tobytes()
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                )

            # when for-loop finishes, the video ended -> while True starts again

    finally:
        log_f.close()


# ============================================================
#                 FLASK ROUTES
# ============================================================
@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

@app.route("/api/alerts")
def api_alerts():
    # SCVD-specific alerts
    with ALERTS_LOCK:
        return jsonify(list(ALERTS))

@app.route("/")
def index():
    return jsonify({
        "message": "SCVD Fight + Weapon backend running",
        "device": str(device),
        "endpoints": ["/video_feed", "/api/alerts"]
    })

# ============================================================
#                  MAIN ENTRY
# ============================================================
if __name__ == "__main__":
    # IMPORTANT: port 5001 to match frontend SCVD_STREAM
    app.run(host="0.0.0.0", port=5001, debug=False)
