import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands



wCam, hCam = 1280,720


cap = cv2.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,hCam)

        


with mp_hands.Hands() as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    height, width, _ = image.shape


    if results.multi_hand_landmarks is not None:    
    # Accediendo a los puntos de referencia, de acuerdo a su nombre
        for hand_landmarks in results.multi_hand_landmarks:
            x1 = int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * width)
            y1 = int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * height)
        

    #dibuja solamente    
        # mp_drawing.draw_landmarks(
        #     image,
        #     hand_landmarks,
        #     mp_hands.HAND_CONNECTIONS,
        #     mp_drawing_styles.get_default_hand_landmarks_style(),
        #     mp_drawing_styles.get_default_hand_connections_style())
        # x1 = int(results.multi_hand_landmarks[7].x * width)
        # y1 = int(results.multi_hand_landmarks[7].y * height)
        # x2 = int(results.multi_hand_landmarks[6].x * width)
        # y2 = int(results.multi_hand_landmarks[6].y * height)
        # x3 = int(results.multi_hand_landmarks[5].x * width)
        # y3 = int(results.multi_hand_landmarks[5].y * height)
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
        break
    cap.release()
