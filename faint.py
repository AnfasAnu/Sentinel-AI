import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# --- VIDEO SOURCE ---
# 0 = webcam. Replace with RTSP URL for CCTV.
cap = cv2.VideoCapture(0)

# Create pose detector
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,        # 0=lite, 1=full, 2=heavy
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

while True:
    ret, frame = cap.read()
    if not ret:
        print("No frame from camera / stream ended")
        break

    # Optional: flip for mirror view (like selfie)
    frame = cv2.flip(frame, 1)

    # Convert BGR to RGB for MediaPipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame for pose
    results = pose.process(rgb)

    # If any person detected
    if results.pose_landmarks:
        # Draw skeleton on the frame
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
        )

        # Example: access a specific joint (nose)
        nose = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
        # print(nose.x, nose.y, nose.z)  # normalized coords if you need them

    cv2.imshow("Pose Skeleton", frame)

    # ESC to quit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
