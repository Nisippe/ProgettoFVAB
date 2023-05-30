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
    frame_1=frame_1[:60]
    list.append(frame_1)
    etichetta_1=f.getEtichettaFromVideo(i,'happy')
    list_2.append(etichetta_1)


X_train=np.array(list)
X_train=X_train[:,0,:,:,0]

#happy_range=(0,100)
Y_train=np.array(list_2)



model = Sequential()
model.add(LSTM(64,input_shape=(1920,1080)))
model.add(Dense(1,activation='sigmoid'))
model.compile(optimizer='Adam',loss='mean_squared_error')

model.fit(X_train,Y_train)


vid2 = cv2.VideoCapture('C:/Users/anton/Desktop/ProgettoFVAB/test/VID_RGB_061.mp4')
x_test=f.getAllFramesFromVideo(vid2)
x_test=x_test[:60]
list_videos_Test=[]
list_videos_Test.append(x_test)
list_videos_Test=np.array(list_videos_Test)
list_videos_Test=list_videos_Test[:,0,:,:,0]


print(model.predict((list_videos_Test)))
