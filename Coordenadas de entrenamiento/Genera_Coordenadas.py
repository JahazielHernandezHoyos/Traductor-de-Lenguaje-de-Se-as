import cv2
import time
import os
import HandTrackingModule as htm
import numpy as np
wCam, hCam = 1280,720

cap = cv2.VideoCapture("C:/Users/Tecnoacademia/Desktop/Proyecto Hahaziel/Cerrado abierto/Letras/Letra_a.mp4")
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

tipIds = [ 4, 8, 12, 16, 20]

contador=0

while True:
    sccess, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    contador=contador+1
    print(contador)


puntos=np.zeros([contador-1,21,3])

contador=0
while True:
    
    sccess, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    
    puntos[contador,:,:]=lmList
    contador=contador+1


np.save('Letra_O.npy',puntos)  

#Letra_A = np.load('Letra_A.npy')

#print(Letra_A[0:5,:,:])




       
