import cv2
import mediapipe as mp
import numpy as np
from math import acos, degrees
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

wCam, hCam = 1280,720

cap = cv2.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,hCam)

up = False
down = False
count = 0


with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5) as hands:

    while True:
            ret, frame = cap.read()
            if ret == False:
                break
            height, width, _ = frame.shape
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            if results.multi_hand_landmarks is not None:    
    # Accediendo a los puntos de referencia, de acuerdo a su nombre
                # for hand_landmarks in results.multi_hand_landmarks:
                #     x1 = int(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].x * width)
                #     y1 = int(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].y * height)
                #     x2 = int(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].x * width)
                #     y2 = int(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y * height)
                #     x3 = int(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x * width)
                #     y3 = int(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y * height)
                #     cv2.circle(frame, (x1, y1), 3,(255,255,200),3)
                #     cv2.circle(frame, (x2, y2), 3,(255,255,200),3)
                #     cv2.circle(frame, (x3, y3), 3,(255,255,200),3)
                    
                #     p1 = np.array([x1, y1])
                #     p2 = np.array([x2, y2])
                #     p3 = np.array([x3, y3])
                #     l1 = np.linalg.norm(p2 - p3)
                #     l2 = np.linalg.norm(p1 - p3)
                #     l3 = np.linalg.norm(p1 - p2)

                #     # Calcular el ángulo
                #     angle = degrees(acos((l1**2 + l3**2 - l2**2) / (2 * l1 * l3)))
                #     #print("count: ", count)
                #     # Visualización
                #     aux_image = np.zeros(frame.shape, np.uint8)
                #     cv2.line(aux_image, (x1, y1), (x2, y2), (0, 255, 255), 20)
                #     cv2.line(aux_image, (x2, y2), (x3, y3), (0, 255, 255), 20)
                #     cv2.line(aux_image, (x1, y1), (x3, y3), (0, 255, 255), 5)
                #     contours = np.array([[x1, y1], [x2, y2], [x3, y3]])
                #     cv2.fillPoly(aux_image, pts=[contours], color=(70, 70, 70))
                #     output = cv2.addWeighted(frame, 1, aux_image, 0.8, 0)
                #     # cv2.rectangle(output, (0, 0), (640, 480), (0, 0, 255), -1)
                #     cv2.rectangle(output, (0, 0), (100, 60), (255, 255, 255), -1)
                #     cv2.putText(output, "Sentadillas", (5, 70), 1, 1.8, (0, 0, 0), 2)
                #     cv2.circle(output, (x1, y1), 6, (0, 255, 0), 4)
                #     cv2.circle(output, (x2, y2), 6, (255, 255, 255), 4)
                #     cv2.circle(output, (x3, y3), 6, (255, 255, 255), 4)
                    
                #     cv2.rectangle(output, (0, 100), (100, 200), (255, 255, 255), -1)
                #     cv2.putText(output, str(int(angle)), (10, 150), 1, 2.8, (0, 0, 0), 2)

                    
                #     cv2.putText(output, str(count), (0, 50), 1, 3.5, (0, 0, 0), 2)
                #     cv2.putText(output, "Angulo", (5, 190), 1, 1.8, (0, 0, 0), 2)
                #     cv2.imshow("output", output)

    # Accediendo al valor de los puntos por su índice
            
                index = [7, 6, 5]
                for hand_landmarks in results.multi_hand_landmarks:
                    for (i, points) in enumerate(hand_landmarks.landmark):
                        if i in index:
                            x = int(points.x * width)
                            y = int(points.y * height)
                            cv2.circle(frame, (x, y), 3,(255, 0, 255), 3)
            cv2.imshow('Frame',frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
cap.release()
cv2.destroyAllWindows()
