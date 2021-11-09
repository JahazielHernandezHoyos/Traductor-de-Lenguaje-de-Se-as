import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

wCam, hCam = 1280,720

cap = cv2.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,hCam)


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
                for hand_landmarks in results.multi_hand_landmarks:
                    x1 = int(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].x * width)
                    y1 = int(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].y * height)
                    x2 = int(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].x * width)
                    y2 = int(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y * height)
                    x3 = int(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x * width)
                    y3 = int(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y * height)
                    cv2.circle(frame, (x1, y1), 3,(255,255,200),3)
                    cv2.circle(frame, (x2, y2), 3,(255,255,200),3)
                    cv2.circle(frame, (x3, y3), 3,(255,255,200),3)
    # Accediendo al valor de los puntos por su índice
            
                # index = [7, 6, 5]
                # for hand_landmarks in results.multi_hand_landmarks:
                #     for (i, points) in enumerate(hand_landmarks.landmark):
                #         if i in index:
                #             x = int(points.x * width)
                #             y = int(points.y * height)
                #             cv2.circle(frame, (x, y), 3,(255, 0, 255), 3)
            cv2.imshow('Frame',frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
cap.release()
cv2.destroyAllWindows()
