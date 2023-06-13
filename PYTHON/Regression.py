import keras
from keras import Sequential, regularizers
from keras.layers import Dense, LSTM, Conv1D

import Feature_Extractor as fe
import Funzioni as f
import numpy as np
import cv2
'''videos=f.getAllVideoGait()
path='C:/Users/drugo/PycharmProjects/ProgettoFVAB/Video_Cut_Landmarks/'
list_features=[]
for video in videos:
    vid=cv2.VideoCapture(path+video)
    features=fe.extract_features_from_video(vid)
    list_features.append(features)
'''


#X_train=np.array(list_features)
#np.save('X_train.npy',X_train)
X_train=np.load('X_train.npy')
y_train=f.getListEtichettaFromVideos('happy')
y_train=np.array(y_train)
print(X_train.shape)
print(y_train.shape)
model = Sequential()
model.add(Dense(64, activation="relu",input_shape=(24,25088)))

model.add(Dense(32, activation="relu",kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
    bias_regularizer=regularizers.L2(1e-4),
    activity_regularizer=regularizers.L2(1e-5)))

model.add(Dense(1,activation="relu",kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
    bias_regularizer=regularizers.L2(1e-4),
    activity_regularizer=regularizers.L2(1e-5)))

model.compile(optimizer='adadelta', metrics=['mse','mae'])
model.fit(X_train,y_train,epochs=100,batch_size=128,validation_data=(X_train,y_train))
model.save('Modello.keras')