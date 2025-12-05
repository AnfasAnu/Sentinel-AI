import cv2
import numpy as np

# --- MAIN THRESHOLD CONTROL ---
# Increase → more strict (less false)
# Decrease → more sensitive (detects smaller flames)
FIRE_SENSITIVITY = 1.7
# Recommended range: 0.5 to 2.0


cap = cv2.VideoCapture(0)

def detect_fire(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Adjust thresholds based on sensitivity
    min_sat = int(100 * FIRE_SENSITIVITY)   # controls flame color strength
    min_val = int(120 * FIRE_SENSITIVITY)   # controls brightness
    min_area = int(2000 * FIRE_SENSITIVITY) # controls flame blob size

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

        if area < min_area:     # controlled by FIRE_SENSITIVITY
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        height_to_width = h / float(w)
        if height_to_width < 0.5: 
            continue

        fire_zones.append((x, y, w, h))

    return fire_zones


while True:
    ret, frame = cap.read()
    if not ret:
        print("Stream ended or unavailable")
        break

    flame_boxes = detect_fire(frame)

    for (x, y, w, h) in flame_boxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, "FIRE", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    if flame_boxes:
        cv2.putText(frame, "FIRE DETECTED", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

    cv2.imshow("Flame Detector", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
