import cv2
import mediapipe as mp
import os


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
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,0), 3)
            
            #----------------redimension de la imagen para que queden del mismo tamaño de las fotos y obligatorio que midan lo mismo para que detecte
            #dedos_reg = cv2.resize(dedos_reg,(200,200), interpolation = cv2.INTER_CUBIC)
            #cv2.imwrite(carpeta + "/mano_{}.jpg".format(cont),dedos_reg)
            #cont = cont + 1
            
            
    cv2.imshow("video",frame)
    k = cv2.waitKey(1)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    #if k == 27 or cont >= 200:
     #   break
cap.release()
cv2.destroyAllWindows()

                