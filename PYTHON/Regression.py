import numpy as np
import tensorflow as tf
import cv2
from keras import Sequential
from keras.layers import LSTM, Dense
import Funzioni as f
import Feature_Extractor as fe
num_video=5
list=[]
list_2=[]
for i in range(0,61):
    vid=cv2.VideoCapture('C:/Users/anton/Desktop/ProgettoFVAB/Video_Cut_Landmarks/VID_RGB_CUT_'+str(i)+'.mp4')
    frame_1=f.getAllFramesFromVideo(vid)
    list.append(frame_1)
    etichetta_1=f.getEtichettaFromVideo(i,'angry')
    print(etichetta_1)
    list_2.append(etichetta_1)
print('mammt')
X_train=np.array(list)
X_train=X_train[:,0,:,:,0]
print(X_train.shape)
Y_train=np.array(list_2)
#model


model = Sequential()
model.add(LSTM(64,input_shape=(1920,1080)))
model.add(Dense(128, activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.compile(optimizer='Adam',loss='mean_squared_error')

model.fit(X_train,Y_train,epochs=500,validation_split=0.3)


vid2 = cv2.VideoCapture('C:/Users/anton/Desktop/ProgettoFVAB/test/VID_RGB_061.mp4')
x_test=f.getAllFramesFromVideo(vid2)
x_test=x_test[:22]
list_cazzi=[]
list_cazzi.append(x_test)
list_cazzi=np.array(list_cazzi)
print(list_cazzi.shape)
list_cazzi=list_cazzi[:,0,:,:,0]
print(list_cazzi.shape)
print(model.predict((list_cazzi)))