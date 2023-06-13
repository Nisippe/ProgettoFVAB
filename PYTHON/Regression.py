import keras
from keras import Sequential
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
model.add(Conv1D(128,3, activation="relu",input_shape=(24,25088)))
model.add(Dense(64, activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dense(1,activation="relu"))
model.compile(optimizer='adam', loss="mae")
model.fit(X_train,y_train,epochs=10,validation_data=(X_train,y_train))
model.save('Modello.keras')