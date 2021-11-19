
import numpy as np
from matplotlib import pyplot as plt
from keras import backend as K
from keras.layers import Dense, Dropout, Activation
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline




Letra_A = np.load('C:/Users/Tecnoacademia/Desktop/Proyecto Hahaziel/ANN/Letra_A.npy')
Letra_E = np.load('C:/Users/Tecnoacademia/Desktop/Proyecto Hahaziel/ANN/Letra_E.npy')
Letra_I = np.load('C:/Users/Tecnoacademia/Desktop/Proyecto Hahaziel/ANN/Letra_I.npy')
Letra_O = np.load('C:/Users/Tecnoacademia/Desktop/Proyecto Hahaziel/ANN/Letra_O.npy')
Letra_U = np.load('C:/Users/Tecnoacademia/Desktop/Proyecto Hahaziel/ANN/Letra_O.npy')


Letra_A=Letra_A[0:800,:,:]
Letra_E=Letra_E[0:800,:,:]

Letra_I=Letra_I[0:800,:,:]
Letra_O=Letra_O[0:800,:,:]
Letra_U=Letra_U[0:800,:,:]


n_img,n_puntos,coord=Letra_A.shape

y_A=np.array([1,0])
y_A=np.tile(y_A,(n_img,1))

y_E=np.array([0,1])
y_E=np.tile(y_E,(n_img,1))

y_I=np.array([0,1])
y_I=np.tile(y_I,(n_img,1))

y_O=np.array([0,1])
y_O=np.tile(y_O,(n_img,1))

y_U=np.array([0,1])
y_U=np.tile(y_U,(n_img,1))

x=np.concatenate((Letra_A, Letra_I),0)
y=np.concatenate((y_A,y_I),0)

print(x.shape)
print(y.shape)


x_1=x[:,:,1]
x_2=x[:,:,2]
x_3=np.concatenate((x_1, x_2),1)
print(x_3.shape)



plt.figure()
plt.imshow(Letra_A[0,:,:])



model = Sequential() 
#capa 1 
model.add(Dense(42, input_dim=42, activation='relu'))
#capa 2
model.add(Dense(20, activation='relu'))
#capa 3
model.add(Dense(2, activation='softmax'))
#compila el modelo
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#modelo creado
model.summary()

    # Fit the model
history=model.fit(x_3, y,
          batch_size=10,
          epochs=200,
          validation_data=(x_3, y),
          shuffle=True,
          # callbacks=callbacks_list,
          verbose=1)

#historia = modelo.fit(X,Y,epochs=n_its,batch_size=batch_size,verbose=2)

#estimator = KerasClassifier(build_fn=baseline_model, epochs=3, batch_size=5, verbose=0)
#kfold = KFold(n_splits=10, shuffle=True)
#results = cross_val_score(estimator, x_1, y, cv=kfold)
#print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))


    


"""
l_train,a,b=izquierda_train.shape
l_test,a,b=izquierda_val.shape

x_train=np.concatenate((izquierda_train,derecha_train),axis=0)
x_train = np.expand_dims(x_train, axis=3)
print("datos de entrenamiento",x_train.shape)


x_test=np.concatenate((izquierda_val,derecha_val),axis=0)
x_test = np.expand_dims(x_test, axis=3)
print("datos de validacion",x_test.shape)


y0_train=np.zeros(l_train)
y1_train=np.ones (l_train)
y0_test =np.zeros(l_test )
y1_test =np.ones (l_test )
y_train=np.concatenate((y0_train,y1_train),axis=0)
y_test =np.concatenate((y0_test, y1_test ),axis=0)

print(y_train.shape)
print(y_test.shape)


a,b,c,d=x_test.shape

validation=[]
img_width, img_height =b,c
channels=d
epochs = 100#2*i+2
batch_size = 32

 
if K.image_data_format() == 'channels_first':
    input_shape = (channels, img_width, img_height)
else:
    input_shape = (img_width, img_height, channels)

for i in range(1,2):
    
    
    model = Sequential()  
    #capa 1
    model.add(Conv2D(256, (7, 7), strides=(2, 2), padding='valid',input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D( pool_size=(2, 2)))    
    #capa 2
    model.add(Conv2D(256, (5, 5), strides=(2, 2),  padding='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #capa 3
    model.add(Conv2D(32, (3, 3), strides=(2, 2),  padding='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(1, 1)))
    #capa 4
    model.add(Conv2D(32, (1, 1), strides=(2, 2),  padding='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(1, 1)))
    
    #capa 5
    model.add(Flatten())
    model.add(Dense(1024,activity_regularizer=l2(0.001)))
    model.add(Activation('sigmoid'))
    model.add(Dropout(0.4))
    #capa 5
    model.add(Dense(1, activation='sigmoid', name='preds'))
    model.summary()
    # initiate RMSprop optimizer
    opt=keras.optimizers.Adam(lr=0.0001)
    # Let's train the model using RMSprop
    model.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
    
    # checkpoint
    # filepath="model_al.hdf5"
    # checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    # callbacks_list = [checkpoint]
    
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    
    
    # Fit the model
    history=model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True,
              # callbacks=callbacks_list,
              verbose=1)
    
    #  "Accuracy"
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    
    validation.append(np.amax(history.history['val_accuracy']))
    K.clear_session()"""