import cv2
import time
import mediapipe as mp

#first commit using pycharm
class HandDetector():
    def __init__(self,mode=False ,maxHands=2, detectionCon=0.5,trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(False,2)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)
        if self.results.multi_hand_landmarks:
                for handLms in self.results.multi_hand_landmarks:
                    if draw :
                        self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):

        lmList = []

        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                # print(id,lm)
                h,w,c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                lmList.append([id,cx,cy])
                if draw :
                    cv2.circle(img,(cx,cy),5,(0,0,0),cv2.FILLED)

        return lmList




#
#
#         cTime = time.time()
#         fps = 1/(cTime-pTime)
#         pTime = cTime
#
#         cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_SIMPLEX,3,(0,0,0),3)
#
#         cv2.imshow("Image",img)
#         cv2.waitKey(1)

def main():
        pTime = 0
        cTime = 0

        cap = cv2.VideoCapture(0);

        detector = HandDetector()
        while True:
                success, img = cap.read()
                img = detector.findHands(img)
                lmList = detector.findPosition(img)
                if len(lmList)!=0:
                    print(lmList[0])

                cTime = time.time()
                fps = 1/(cTime-pTime)
                pTime = cTime

                cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_SIMPLEX,3,(0,0,0),3)

                cv2.imshow("Image",img)
                cv2.waitKey(1)

if __name__ == "__main__":
    main()