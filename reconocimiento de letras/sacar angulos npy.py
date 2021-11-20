import cv2
import mediapipe as mp
import numpy as np
import math
from math import *
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


#cap = cv2.VideoCapture("Letras/Letra_e.mp4")
cap = cv2.VideoCapture(0)


wCam, hCam = 1920,  1080
cap.set(3,wCam)
cap.set(4,hCam)

up = False
down = False
count = 0


with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.75) as hands:

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
                for hand_landmarks in results.multi_hand_landmarks:

                    # COORDENADAS MEÑIQUE
                    x1 = int(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].x * width)
                    y1 = int(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].y * height)
                    x2 = int(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].x * width)
                    y2 = int(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y * height)
                    x3 = int(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x * width)
                    y3 = int(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y * height)
                    
                    #COORDENADAS ANULAS
                    x4 = int(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].x * width)
                    y4 = int(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].y * height)
                    x5 = int(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].x * width)
                    y5 = int(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y * height)
                    x6 = int(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].x * width)
                    y6 = int(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y * height)

                    #COORDENADAS MEDIO
                    x7 = int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].x * width)
                    y7 = int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y * height)
                    x8 = int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].x * width)
                    y8 = int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y * height)
                    x9 = int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x * width)
                    y9 = int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y * height)

                    #COORDENADAS INDICE
                    x10 = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].x * width)
                    y10 = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y * height)
                    x11 = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].x * width)
                    y11 = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y * height)
                    x12 = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x * width)
                    y12 = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y * height)

                    #COORDENADAS PULGAR PARTE EXTERNA
                    x13 = int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * width)
                    y13 = int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * height)
                    x14 = int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x * width)
                    y14= int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y * height)
                    x15 = int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x * width)
                    y15 = int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y * height)

                    #COORDENADAS PULGAR PARTE INTERNA
                    x16 = int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x * width)
                    y16 = int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y * height)
                    x17 = int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].x * width)
                    y17 = int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].y * height)
                    x18 = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * width)
                    y18 = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * height)

                    p1 = np.array([x1, y1])
                    p2 = np.array([x2, y2])
                    p3 = np.array([x3, y3])

                    p4 = np.array([x4, y4])
                    p5 = np.array([x5, y5])
                    p6 = np.array([x6, y6])

                    p7 = np.array([x7, y7])
                    p8 = np.array([x8, y8])
                    p9 = np.array([x9, y9])

                    p10 = np.array([x10, y10])
                    p11 = np.array([x11, y11])
                    p12 = np.array([x12, y12])

                    p13 = np.array([x13, y13])
                    p14 = np.array([x14, y14])
                    p15 = np.array([x15, y15])

                    p16 = np.array([x16, y16])
                    p17 = np.array([x17, y17])
                    p18 = np.array([x18, y18])

                    l1 = np.linalg.norm(p2 - p3)
                    l2 = np.linalg.norm(p1 - p3)
                    l3 = np.linalg.norm(p1 - p2)

                    l4 = np.linalg.norm(p5 - p6)
                    l5 = np.linalg.norm(p4 - p6)
                    l6 = np.linalg.norm(p4 - p5)

                    l7 = np.linalg.norm(p8 - p9)
                    l8 = np.linalg.norm(p7 - p9)
                    l9 = np.linalg.norm(p7 - p8)

                    l10 = np.linalg.norm(p11 - p12)
                    l11 = np.linalg.norm(p10 - p12)
                    l12 = np.linalg.norm(p10 - p11)

                    l13 = np.linalg.norm(p14 - p15)
                    l14 = np.linalg.norm(p13 - p15)
                    l15 = np.linalg.norm(p13 - p14)

                    l16 = np.linalg.norm(p17 - p18)
                    l17 = np.linalg.norm(p16 - p18)
                    l18 = np.linalg.norm(p16 - p17)



                    # Calcular el ángulo
                    if l1 and l3 != 0:
                        angle1 = round(degrees(abs(atan((l1**2 + l3**2 - l2**2) / (2 * l1 * l3)))))
                    else:
                        angle1 = 1

                    if l4 and l6 != 0:
                        angle2 = round(degrees(abs(atan((l4**2 + l6**2 - l5**2) / (2 * l4 * l6)))))
                    else:
                        angle2 = 1

                    if l7 and l9 != 0:
                        angle3 = round(degrees(abs(atan((l7**2 + l9**2 - l8**2) / (2 * l7 * l9)))))
                    else: 
                        angle3 = 1

                    if l10 and l12 != 0:
                        angle4 = round(degrees(abs(atan((l10**2 + l12**2 - l11**2) / (2 * l10 * l12)))))
                    else:
                        angle4 = 1

                    if l13 and l15 != 0:  
                        angle5 = round(degrees(abs(atan((l13**2 + l15**2 - l14**2) / (2 * l13 * l15)))))
                    else:
                        angle5 = 1
                    
                    if l16 and l18 != 0:
                        angle6 = round(degrees(abs(atan((l16**2 + l18**2 - l17**2) / (2 * l16 * l18)))))
                    



                    #angulo normal 
                    # angle1 = round(degrees(acos((l1**2 + l3**2 - l2**2) / (2 * l1 * l3))))
                    # angle2 = round(degrees(acos((l4**2 + l6**2 - l5**2) / (2 * l4 * l6))))
                    # angle3 = round(degrees(acos((l7**2 + l9**2 - l8**2) / (2 * l7 * l9))))
                    # angle4 = round(degrees(acos((l10**2 + l12**2 - l11**2) / (2 * l10 * l12))))
                    # angle5 = round(degrees(acos((l13**2 + l15**2 - l14**2) / (2 * l13 * l15))))
                    # angle6 = round(degrees(acos((l16**2 + l18**2 - l17**2) / (2 * l16 * l18))))

                    #binarizacion 1 2 3
                    # angle1 = round(acos((l1**2 + l3**2 - l2**2) / (2 * l1 * l3)))
                    # angle2 = round(acos((l4**2 + l6**2 - l5**2) / (2 * l4 * l6)))
                    # angle3 = round(acos((l7**2 + l9**2 - l8**2) / (2 * l7 * l9)))
                    # angle4 = round(acos((l10**2 + l12**2 - l11**2) / (2 * l10 * l12)))
                    # angle5 = round(acos((l13**2 + l15**2 - l14**2) / (2 * l13 * l15)))
                    # angle6 = round(acos((l16**2 + l18**2 - l17**2) / (2 * l16 * l18)))

                    
                    #binarizada 0 90 180
                    # angle1 = degrees(acos(round((l1**2 + l3**2 - l2**2) / (2 * l1 * l3))))
                    # angle2 = degrees(acos(round((l4**2 + l6**2 - l5**2) / (2 * l4 * l6))))
                    # angle3 = degrees(acos(round((l7**2 + l9**2 - l8**2) / (2 * l7 * l9))))
                    # angle4 = degrees(acos(round((l10**2 + l12**2 - l11**2) / (2 * l10 * l12))))
                    # angle5 = degrees(acos(round((l13**2 + l15**2 - l14**2) / (2 * l13 * l15))))
                    # angle6 = degrees(acos(round((l16**2 + l18**2 - l17**2) / (2 * l16 * l18))))
                    
                    
                    #print("count: ", count)
                    # Visualización
                    
                    print (angle1, angle2, angle3, angle4, angle5)
                    # print (angle1, angle2, angle3, angle4, angle5, angle6)
                    aux_image = np.zeros(frame.shape, np.uint8)

                    
                    cv2.circle(frame, (x1, y1), 3,(255,255,200),3)
                    cv2.circle(frame, (x2, y2), 3,(255,255,200),3)
                    cv2.circle(frame, (x3, y3), 3,(255,255,200),3)
                    
                    cv2.line(aux_image, (x1, y1), (x2, y2), (0, 255, 255), 20)
                    cv2.line(aux_image, (x2, y2), (x3, y3), (0, 255, 255), 20)
                    cv2.line(aux_image, (x1, y1), (x3, y3), (0, 255, 255), 5)
                    contours = np.array([[x1, y1], [x2, y2], [x3, y3]])
                    cv2.fillPoly(aux_image, pts=[contours], color=(70, 70, 70))
                    output = cv2.addWeighted(frame, 1, aux_image, 0.8, 0)
                    cv2.rectangle(output, (0, 0), (640, 480), (0, 0, 255), -1)
                    cv2.rectangle(output, (0, 0), (100, 60), (255, 255, 255), -1)
                    cv2.putText(output, "cerrado", (5, 70), 1, 1.8, (0, 0, 0), 2)
                    cv2.circle(output, (x1, y1), 6, (0, 255, 0), 4)
                    cv2.circle(output, (x2, y2), 6, (255, 255, 255), 4)
                    cv2.circle(output, (x3, y3), 6, (255, 255, 255), 4)
                    
                    cv2.rectangle(output, (0, 100), (100, 200), (255, 255, 255), -1)
                    cv2.putText(output, str(int(angle6)), (10, 150), 1, 2.8, (0, 0, 0), 2)

                    
                    cv2.putText(output, str(count), (0, 50), 1, 3.5, (0, 0, 0), 2)
                    cv2.putText(output, "Angulo", (5, 190), 1, 1.8, (0, 0, 0), 2)
                    
                    cv2.imshow("output", output)

                    # print(angle1)
    # Accediendo al valor de los puntos por su índice
                

                
                index = [0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19]
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

# contador=0

# while True:
#     sccess, img = cap.read()
#     img = detector.findHands(img)
#     lmList = detector.findPosition(img, draw=False)
#     contador=contador+1
#     print(contador)


# puntos=np.zeros([contador-1,21,3])

# contador=0
# while True:
    
#     sccess, img = cap.read()
#     img = detector.findHands(img)
#     lmList = detector.findPosition(img, draw=False)
    
#     puntos[contador,:,:]=lmList
#     contador=contador+1


# np.save('Letra_O.npy',puntos)  

# #Letra_A = np.load('Letra_A.npy')

# #print(Letra_A[0:5,:,:])

