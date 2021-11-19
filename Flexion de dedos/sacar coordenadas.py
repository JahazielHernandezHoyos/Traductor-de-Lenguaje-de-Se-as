import cv2
import time
import os
import HandTrackingModule as htm

wCam, hCam = 1280,720

#cap = cv2.VideoCapture("Letra_u.mp4")

cap = cv2.VideoCapture(0)

cap.set(3,wCam)
cap.set(4,hCam)

# Path= "Letras"
# myList = os.listdir(Path)
#print(myList)
# overlayList = []
# for imPath in myList:
#     image = cv2.imread(f"{Path}/{imPath}")
#     print(f"{Path}/{imPath}")
#     overlayList.append(image)

#print(len(overlayList))
pTime = 0

detector = htm.handDetector()

tipIds = [ 4, 8, 12, 16, 20]


while True:
    sccess, img = cap.read()
    img = detector.findHands(img)
    img = cv2.flip(img, 1)
    lmList = detector.findHands(img, draw=False)
    #print(lmList)

    if len(lmList) != 0:

        dedos = []

# #pulgar
#         if lmList[tipIds[0]][1] > lmList[tipIds[0]-1][1]:
#             dedos.append(1)
#         else:
#             dedos.append(0)

# #4 dedos
#         for id in range(1,5):
#             if lmList[tipIds[id]][2] < lmList[tipIds[id]-2][2]:
#                 dedos.append(1)
#             else:
#                 dedos.append(0)
        
        #print(dedos)
        TotalDedos = dedos.count(1)
        #print(TotalDedos)

        # h, w, c = overlayList[TotalDedos-1].shape
        # img[0:h,0:w] = overlayList[TotalDedos-1]

# coordenadas

#punto 0
        punto0_x=lmList[0][1]
        punto0_y=lmList[0][2]

        punto1_x=lmList[1][1]
        punto1_y=lmList[1][2]
    
        punto2_x=lmList[2][1]
        punto2_y=lmList[2][2]

        punto3_x=lmList[3][1]
        punto3_y=lmList[3][2]

        punto4_x=lmList[4][1]
        punto4_y=lmList[4][2]

        punto5_x=lmList[5][1]
        punto5_y=lmList[5][2]

        punto6_x=lmList[6][1]
        punto6_y=lmList[6][2]

        punto7_x=lmList[7][1]
        punto7_y=lmList[7][2]

        punto8_x=lmList[8][1]
        punto8_y=lmList[8][2]

        punto9_x=lmList[9][1]
        punto9_y=lmList[9][2]

        punto10_x=lmList[10][1]
        punto10_y=lmList[10][2]

        punto12_x=lmList[12][1]
        punto12_y=lmList[12][2]

        punto13_x=lmList[13][1]
        punto13_y=lmList[13][2]

        punto14_x=lmList[14][1]
        punto14_y=lmList[14][2]

        punto15_x=lmList[15][1]
        punto15_y=lmList[15][2]

        punto16_x=lmList[16][1]
        punto16_y=lmList[16][2]

        punto17_x=lmList[17][1]
        punto17_y=lmList[17][2]

        punto18_x=lmList[18][1]
        punto18_y=lmList[18][2]

        punto19_x=lmList[19][1]
        punto19_y=lmList[19][2]

        punto20_x=lmList[20][1]
        punto20_y=lmList[20][2]



        
        print(lmList)



    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, f"fps: {int(fps)}", (200,70), cv2.FONT_HERSHEY_PLAIN,
        3,(255,0,0),3)


    cv2.imshow("Image",img)
    cv2.waitKey(1)
    if cv2.waitKey(1) == "q":
        break
cap.release()
cv2.destroyAllWindows()
