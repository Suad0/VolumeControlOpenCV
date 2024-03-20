import cv2
import numpy as np
import os
import mediapipe as mp


def change_volume(delta):
    # Aktuelle Lautstärke
    current_volume = int(os.popen("osascript -e 'output volume of (get volume settings)'").read())

    new_volume = max(0, min(100, current_volume + delta))

    os.system(f"osascript -e 'set volume output volume {new_volume}'")

    return new_volume


cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
fontColor = (0, 255, 0)
lineType = 2

current_volume = -1  # Aktuelle Lautstärke initialisieren

while True:
    ret, frame = cap.read()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            #  Positionen der Gelenke
            landmarks_list = [(lm.x, lm.y, lm.z) for lm in landmarks.landmark]

            # vektoren  der Gelenken
            palm_vector = np.array(landmarks_list[0]) - np.array(landmarks_list[17]) # 17 kleiner finger ende
            thumb_vector = np.array(landmarks_list[4]) - np.array(landmarks_list[3]) # vom zeige finger

            #  Winkel
            angle = np.degrees(np.arccos(
                np.dot(palm_vector, thumb_vector) / (np.linalg.norm(palm_vector) * np.linalg.norm(thumb_vector))))

            # Zeigen Sie den Winkel auf dem Bild an
            cv2.putText(frame, f'Handrotation: {angle:.2f} Grad', (10, 30), font, fontScale, fontColor, lineType)

            for i, landmark in enumerate(landmarks.landmark):
                x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

            volume_change = int(angle) - 90  # 90 Grad Neutral
            new_volume = change_volume(volume_change)
            current_volume = new_volume

    cv2.putText(frame, f'Lautstaerke: {current_volume}', (10, 60), font, fontScale, fontColor, lineType)

    cv2.imshow('Volume Control', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
