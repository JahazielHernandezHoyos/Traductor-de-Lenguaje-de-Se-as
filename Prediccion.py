import cv2
import mediapipe as mp
import os
import numpy as np
from keras_preprocessing.image import load_img, img_to_array
from keras.models import load_model

modelo = ""
peso = ""
cnn = load_model(modelo) 
cnn.load_weights(peso)

direccion: "" #validacion direccion
dire_img = os.listdir() #guardamos lista mano izquierda mano derecha
print("Nombres: ", dire_img)

#leemos la camara (lector de camapra con cv2)
cap = cv2.VideoCapture(0)

   

#creamos un objeto que va almacenar la deteccion y seguimiento
clase_manos = mp.solutions.hands

manos = clase_manos.Hands(False, 4, 0.7, 0.1)       
    
    #cuando esta en true es para imagenes y cuando esta en false es para videos
#    static_image_mode=False,

    #maxnumhands es para indicar el numero maximo de manos detectadas en la imagen
 #   max_num_hands=4,

    #porcentaje de deteccion requerido para rasterizar
  #  min_detection_confidence=0.7)

dibujo = mp.solutions.drawing_utils




#captura de video
while (1):
    ret,frame =cap.read()
    color = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    copia = frame.copy()
    resultado = manos.process(color)
    posiciones = [] #almacena coordenadas de puntos
    
    #print(resultado.multi_hand_landmarks)

    if resultado.multi_hand_landmarks: #si hay algo en los resultados entramos al if
        for mano in resultado.multi_hand_landmarks: #buscamos la mano dentro de la lista de manos que nos da el descriptor
            for id, lm in enumerate(mano.landmark): #vamos a obtener la informacion de cada mano encontrada por el ID
                #print(id,lm) #proporcion de la imagen a pixeles 
                alto, ancho, c = frame.shape #sacar ancho y alto de los frametime o fotogramas del video 
                corx, cory = int(lm.x*ancho), int(lm.y*alto) #extraccion de la ubicacion de cada punto que perteneca a la mano en coordenadas
                posiciones.append([id,corx,cory])
                dibujo.draw_landmarks(frame, mano, clase_manos.HAND_CONNECTIONS) #realiza la conexion de los puntos
            if len(posiciones) != 0:
                pto_i1 = posiciones[4] 
                pto_i2 = posiciones[20]
                pto_i3 = posiciones[12]
                pto_i4 = posiciones[8]
                pto_i5 = posiciones[9] #punto central
                
                x1,y1 = (pto_i5[1]-80),(pto_i5[2]-80) #para obtener el punto inicial del cuadro de pixeles donde estara la mano
                ancho, alto = (x1+80),(y1+80)
                x2,y2 = x1 + ancho, y1 + alto
                dedos_reg = copia[y1:y2, x1:x2]
                dedos_reg = cv2.resize(dedos_reg, (200, 200), interpolation=cv2.INTER_CUBIC) #redimensionar fotos
                x = img_to_array(dedos_reg) #convertimos la imagen a una matriz
                x = np.expand_dims(x, axis=0) #agregamos nuevo eje
                vector = cnn.predict(x) #va a ser un arreglo de 2 dimensiones, donde va a poner 1 en la clase que crea correcta
                resultado = vector[0] # clase 1 [1,0] , clase 2[0,1] entre otros almacenar en vector 0 
                respuesta = np.armax(resultado) #nos entrega el indice del valor mas alto 0 - 1 lo almacena mayor maximo con armax

                #hacemos informacion

                if respuesta == 1:
                    print(vector, resultado)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,0), 3)
                    cv2.rectangle(frame, "{}".format(dire_img[1]), (x1, y1 - 5), 1, 1.3, (0, 0, 255), 1, cv2.LINE_AA)
                
            
            #----------------redimension de la imagen para que queden del mismo tamaño de las fotos y obligatorio que midan lo mismo para que detecte
            #dedos_reg = cv2.resize(dedos_reg,(200,200), interpolation = cv2.INTER_CUBIC)
            #cv2.imwrite(carpeta + "/mano_{}.jpg".format(cont),dedos_reg)
            #cont = cont + 1
            
            
    cv2.imshow("video",frame)
    k = cv2.waitKey(1)
    
    if k == 27:
        break
    
    #if k == 27 or cont >= 200:
     #   break
cap.release()
cv2.destroyAllWindows()

                