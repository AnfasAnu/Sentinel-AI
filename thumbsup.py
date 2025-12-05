import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

def is_thumbs_up(hand_landmarks):
    """
    Returns True if thumb is up based on relative landmark positions
    (Right-hand model logic)
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


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)   # mirror view

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            if is_thumbs_up(hand_landmarks):
                cv2.putText(
                    frame, "THUMBS UP!", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
                )

    cv2.imshow("Thumbs Up Detector", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break   # ESC to exit

cap.release()
cv2.destroyAllWindows()

#python version 3.12
#& C:/Users/sayoo/AppData/Local/Programs/Python/Python312/python.exe c:/Users/sayoo/programming/node/github/Sentinel-AI/thumbsup.py
