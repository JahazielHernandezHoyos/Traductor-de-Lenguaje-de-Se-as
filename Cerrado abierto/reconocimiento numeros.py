import cv2
import time
import os
import HandTrackingModule as htm

wCam, hCam = 1280,720


cap = cv2.VideoCapture(1)
cap.set(3,wCam)
cap.set(4,hCam)

Path= "Letras"
myList = os.listdir(Path)
#print(myList)
overlayList = []
for imPath in myList:
    image = cv2.imread(f"{Path}/{imPath}")
    print(f"{Path}/{imPath}")
    overlayList.append(image)

#print(len(overlayList))
pTime = 0

detector = htm.HandDetector(detectionCon=0.75)

while True:
    sccess, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    #print(lmList)

    if len(lmList) != 0:
        if lmList[8][2] < lmList[6][2]:
            print("el indice esta abierto")
        else: print("El indice esta cerrado") 

    h, w, c = overlayList[0].shape
    img[0:h,0:w] = overlayList[1]

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, f"fps: {int(fps)}", (200,70), cv2.FONT_HERSHEY_PLAIN,
        3,(255,0,0),3)


    cv2.imshow("Image",img)
    cv2.waitKey(1)

