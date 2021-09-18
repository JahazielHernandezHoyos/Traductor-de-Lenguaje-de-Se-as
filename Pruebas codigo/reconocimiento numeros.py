import cv2
import time
import os
import mediapipe as mp
import HandTrackingModule as htm

#creacion de la carpeta usando libreria OS o ubicandonos en ella
nombre = "Mano_Izquierda"
direccion = "C:/Users/jahaz/Desktop/lsd/Manos/entrenamiento izquierda"
carpeta = direccion + '/' + nombre
if not os.path.exists(carpeta):
    print("carpeta creada: ", carpeta)
    os.makedirs(carpeta)
    os.makedirs(carpeta)

#asignamos un contador para el nombre de la foto
cont = 0



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
#print(overlayList)
pTime = 0

detector = htm.HandDetector(detectionCon=0.75)

tipIds = [ 4, 8, 12, 16, 20]


while True:
    sccess, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    #print(lmList)

    if len(lmList) != 0:

        dedos = []

#pulgar
        if lmList[tipIds[0]][1] > lmList[tipIds[0]-1][1]:
            dedos.append(1)
        else:
            dedos.append(0)

#4 dedos
        for id in range(1,5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id]-2][2]:
                dedos.append(1)
            else:
                dedos.append(0)
        
        #print(dedos)
        TotalDedos = dedos.count(1)
        #print(TotalDedos)
        
        #pulgar = list(lmList)
        #print(pulgar)

        punto1=lmList[0[0][1]]
        print(punto1)

        #print(dedos[0])

        h, w, c = overlayList[TotalDedos-1].shape
        img[0:h,0:w] = overlayList[TotalDedos-1]
        
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, f"fps: {int(fps)}", (200,70), cv2.FONT_HERSHEY_PLAIN,
        3,(255,0,0),3)


    cv2.imshow("Image",img)
    k = cv2.waitKey(1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    cont = 0
    if k == 27 or cont >= 200:
        break
cap.release()
cv2.destroyAllWindows()

