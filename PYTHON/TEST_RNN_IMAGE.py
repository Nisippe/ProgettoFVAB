import numpy as np
import cv2
from keras import Sequential
from keras.layers import LSTM, Dense
import Funzioni as f

#num_video=5
list=[]
list_2=[]
for i in range(0,5):
    vid=cv2.VideoCapture('C:/Users/anton/Desktop/ProgettoFVAB/Video_Train/VID_RGB_CUT_'+str(i)+'.avi')
    frame_1=f.getAllFramesFromVideo(vid)
    frame_x=frame_1[0]
    list.append(frame_x)
    etichetta_1=f.getEtichettaFromVideo(i,'happy')
    list_2.append(etichetta_1)

#happy_range=(0,100)

X_train=np.array(list)
X_train=X_train[:,:,:,0]
Y_train=np.array(list_2)
print(X_train.shape)
print(Y_train.shape)

#modello temporaneo
model = Sequential()
model.add(LSTM(64,input_shape=(1920,1080)))
model.add(Dense(1,activation='sigmoid'))
model.compile(optimizer='Adam',loss='mean_squared_error')
model.fit(X_train,Y_train)


vid2 = cv2.VideoCapture('C:/Users/anton/Desktop/ProgettoFVAB/test/VID_RGB_061.mp4')
x_test=f.getAllFramesFromVideo(vid2)
frame_0=x_test[0]
frame_0=frame_0[:,:,0]
list=[]
list.append(frame_0)
print(model.predict(np.array(list)))