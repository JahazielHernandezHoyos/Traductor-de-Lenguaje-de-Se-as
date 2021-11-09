
import cv2
import mediapipe as mp
import numpy as np
from math import acos, degrees

#cap = cv2.VideoCapture("Multimedia\sentadilla.mp4")
cap = cv2.VideoCapture(0)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

up = False
down = False
count = 0
with mp_hands.Hands() as hands:
        while True:
            ret, frame = cap.read()
            if ret == False:
                break
            frame = cv2.flip(frame, 1)
            height, width, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            if results.multi_hand_landmarks is not None:
                mp_drawing.draw_landmarks(frame, results.hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(128, 250, 250), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(255,255,255), thickness=2))
                x1 = int(results.hand_landmarks.landmark[7].x * width)
                y1 = int(results.hand_landmarks.landmark[7].y * height)
                x2 = int(results.hand_landmarks.landmark[6].x * width)
                y2 = int(results.hand_landmarks.landmark[6].y * height)
                x3 = int(results.hand_landmarks.landmark[5].x * width)
                y3 = int(results.hand_landmarks.landmark[5].y * height)
                p1 = np.array([x1, y1])
                p2 = np.array([x2, y2])
                p3 = np.array([x3, y3])
                l1 = np.linalg.norm(p2 - p3)
                l2 = np.linalg.norm(p1 - p3)
                l3 = np.linalg.norm(p1 - p2)
                # Calcular el ángulo
                angle = degrees(acos((l1**2 + l3**2 - l2**2) / (2 * l1 * l3)))
                if angle >= 160:
                        up = True
                if up == True and down == False and angle <= 90:
                        down = True
                if up == True and down == True and angle <= 70:
                        count -= 1
                        up = False
                        down = False
                        cv2.rectangle(output, (0, 0), (640, 480), (0, 0, 255), -1)
                if up == True and down == True and angle >= 160:
                        count += 1
                        up = False
                        down = False
                #print("count: ", count)
                # Visualización
                aux_image = np.zeros(frame.shape, np.uint8)
                cv2.line(aux_image, (x1, y1), (x2, y2), (0, 255, 255), 20)
                cv2.line(aux_image, (x2, y2), (x3, y3), (0, 255, 255), 20)
                cv2.line(aux_image, (x1, y1), (x3, y3), (0, 255, 255), 5)
                contours = np.array([[x1, y1], [x2, y2], [x3, y3]])
                cv2.fillPoly(aux_image, pts=[contours], color=(70, 70, 70))
                output = cv2.addWeighted(frame, 1, aux_image, 0.8, 0)
                if angle <= 75: 
                     cv2.rectangle(output, (0, 0), (640, 480), (0, 0, 255), -1)
                if 76 <= angle <= 90:
                     cv2.rectangle(output, (0, 0), (640, 480), (0, 255, 0), -1) 
                # cv2.rectangle(output, (0, 0), (640, 480), (0, 0, 255), -1)
                cv2.rectangle(output, (0, 0), (100, 60), (255, 255, 255), -1)
                cv2.putText(output, "Sentadillas", (5, 70), 1, 1.8, (0, 0, 0), 2)
                cv2.circle(output, (x1, y1), 6, (0, 255, 0), 4)
                cv2.circle(output, (x2, y2), 6, (255, 255, 255), 4)
                cv2.circle(output, (x3, y3), 6, (255, 255, 255), 4)
                
                cv2.rectangle(output, (0, 100), (100, 200), (255, 255, 255), -1)
                cv2.putText(output, str(int(angle)), (10, 150), 1, 2.8, (0, 0, 0), 2)

                
                cv2.putText(output, str(count), (0, 50), 1, 3.5, (0, 0, 0), 2)
                cv2.putText(output, "Angulo", (5, 190), 1, 1.8, (0, 0, 0), 2)
                cv2.imshow("output", output)

            cv2.imshow("Frame", frame)

            # Aqui se coloca la condicion para el contador de sentadillas y el numero para que haga su cierre o llamar otra funcion
            if count == 10:
                break
            if cv2.waitKey(1) & 0xFF == 27:
                break     
             
        cap.release()
        cv2.destroyAllWindows()
        