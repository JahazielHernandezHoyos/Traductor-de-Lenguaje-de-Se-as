from tensorflow.python.keras.preprocessing.image import ImageDataGenerator #preproceso de imagenes para entregar
from tensorflow.python.keras import optimizers #optimizar modelo a entrenar
from tensorflow.python.keras.models import Sequential #redes neuronales secuenciales
from tensorflow.python.keras.layers import Dropout, Flatten, Dense #activacion de una capa importante 
from tensorflow.python.keras.layers import Convolution20, MaxPooling20 #nos permite añadir capas
from tensorflow.python.keras import backend as K #limpieza de sesiones keras

K.clear_session() #limpiar

datos_entrenamiento=""
datos_validacion=""

#parametros
iteraciones = 20 #while para repetir y ajustarse
altura, longitud = 200,200 #tamaño de las fotos
batch_size = 1 #numero de imagenes a enviar secuencialmente 
pasos = 300/1 #numero de veces que se va a procesar la informacion en cada iteracion
pasos_validacion = 300/1 #despues de cada iteracion, validamos lo anterior imagenes de testeo
filtrosconv1 = 32
filtrosconv2 = 64 #numero de filtros que vamos a aplicar en cada convolucion
tam_filtro1 = (3,3)
tam_filtro2 = (2,2) #tamaño de filtros
tam_pool = (2,2) #tamaño del filtro max pooling
clases = 2 #mano abierta y cerrada (5 dedos y 8 dedos)
lr = 0.0005 #ajustes de la red neuronal para acercarse a una solucion optima  learning rate

#pre procesado
preprocesamiento_entre = ImageDataGenerator(
    rescale = 1/255,
    shear_tange = 0.3, #generar muestras imagenes inclinadas para un mejor entrenamiento
    zoom_range = 0.3, #genera imagenes con zoom para un mejor datos_entrenamiento
    horizontal_flip= True  #invertir las fotos
)

preprocesamiento_vali = ImageDataGenerator(
    rescale = 1/255
)

imagen_entreno = preprocesamiento_entre.flow_from_directory(
    datos_entrenamiento, #va a tomar fotos que ya almacenamos
    target_size = (altura, longitud),
    batch_size = batch_size,
    class_mode = "categorical", #clasificacion categorica = por clases
)

#creamos la red neuronal convolucional 
cnn = Sequential() #seleccionamos la red neuronal secuencial 
#vienen los filtros para darle profundidad en datos_entrenamiento
cnn.add(Convolution2D(filtrosconv1, tan_filtro1, padding = "same", input_shape=(altura,longitud,3), activation = "relu")) #agregamos filtro 2D primera capa

cnn.add(MaxPooling2D(pool_size=tam_pool))

cnn.add(Convolution2D(filtrosconv2, tam_filtro2, padding = "same", activation="relu")) #segunda capa

cnn.add(MaxPooling2D(pool_size=tam_pool))

#las proximas lineas son para convertir la imagen profunda en una plana para tener 1dimension con toda la informacion
cnn.add(Flatten()) #flatten es para aplanar la imagen
cnn.add(Dense(256, activation= "relu")) # metemos 256 neuronas con la regla relu y el comando Dense
cnn.add(Dropout(0.5)) #con dropout apagamos neuronas en este caso 0.5 = 50% de las neuronas
cnn.add(Dense(clases, activation= "softmax")) #ultima capa, nos dice las porbabilidad de que sea 

#agregamos parametros para optimizar el modelo
#durante el entrenamiento tiene autoevaluacion, que se optimize con adam, y la metrica sera accuracy
optimizar = tensorflow.keras.optimizers.Adan(learning_rate=lr)
cnn.compile(loss = "categorical_crossentropy", optimizer= optimizar, metrics=["accuracy"])

#siguiente linea es para entrenar la red o empezarla a entrenar
cnn.flit(imagen_entreno, steps_per_epech=pasos, epochs= iteraciones, validation_data= imagen_validacion, validacion_steps=pasos_validacion)


#guardamos modelo y pesos de nuestras 
cnn.save ("Modelo.h5")
cnn.save_weights("pesos.h5")


