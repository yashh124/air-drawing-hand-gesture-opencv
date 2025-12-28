import cv2
import mediapipe as mp
import numpy as np

# Camera
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,        # IMPORTANT: only one hand
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# Canvas
canvas = np.zeros((720, 1280, 3), dtype=np.uint8)

# Previous finger position
prev_x, prev_y = 0, 0

# Finger tips
tips = [4, 8, 12, 16, 20]

def fingers_up(lm):
    fingers = []
    # Thumb
    fingers.append(lm[tips[0]].x < lm[tips[0]-1].x)
    # Other fingers
    for i in range(1, 5):
        fingers.append(lm[tips[i]].y < lm[tips[i]-2].y)
    return fingers

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]
        lm = hand_landmarks.landmark
        h, w, _ = frame.shape

        # Index finger tip
        x = int(lm[8].x * w)
        y = int(lm[8].y * h)

        fingers = fingers_up(lm)

        # ✋ Clear screen (open palm)
        if fingers == [True, True, True, True, True]:
            canvas[:] = 0
            prev_x, prev_y = 0, 0

        # ☝️ Draw (only index finger up)
        elif fingers == [False, True, False, False, False]:
            if prev_x == 0 and prev_y == 0:
                prev_x, prev_y = x, y

            cv2.line(canvas, (prev_x, prev_y), (x, y), (255, 0, 255), 6)
            prev_x, prev_y = x, y
        else:
            prev_x, prev_y = 0, 0

        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Merge canvas & frame
    frame = cv2.addWeighted(frame, 0.5, canvas, 1, 0)

    cv2.imshow("Hand Gesture Drawing", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
