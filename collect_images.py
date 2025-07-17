import cv2
import mediapipe as mp
import numpy as np
import os
from datetime import datetime

# Set up MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Define labels
labels = ["hello", "yes", "no", "iloveyou", "thankyou"]
label_index = {label: idx for idx, label in enumerate(labels)}

# Set data path
data_path = 'MP_Data'
os.makedirs(data_path, exist_ok=True)

# Create folders if they don't exist
for label in labels:
    os.makedirs(os.path.join(data_path, label), exist_ok=True)

# Collect data
cap = cv2.VideoCapture(0)
current_label = None
counter = 0

print("Press corresponding key to collect data:")
print("h: hello | y: yes | n: no | i: iloveyou | t: thankyou | q: quit")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(image_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            # Extract landmarks
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            
            if current_label is not None:
                npy_path = os.path.join(data_path, current_label, f'{current_label}_{datetime.now().strftime("%Y%m%d%H%M%S%f")}.npy')
                np.save(npy_path, np.array(landmarks))
                counter += 1

    cv2.putText(frame, f'Label: {current_label if current_label else "None"} | Count: {counter}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow('Collect Landmarks', frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('h'):
        current_label = "hello"
        counter = 0
    elif key == ord('y'):
        current_label = "yes"
        counter = 0
    elif key == ord('n'):
        current_label = "no"
        counter = 0
    elif key == ord('i'):
        current_label = "iloveyou"
        counter = 0
    elif key == ord('t'):
        current_label = "thankyou"
        counter = 0
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
