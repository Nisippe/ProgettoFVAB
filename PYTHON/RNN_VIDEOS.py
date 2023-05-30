import numpy as np
import cv2
from keras import Sequential
from keras.layers import LSTM, Dense
import Funzioni as f

model = Sequential()
model.add(LSTM(64, input_shape=(1920, 1080)))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='Adam', loss='mean_squared_error')

def train_model(emozione):
    #Da eseguire una sola volta e poi salvare il modello
    list = []
    list_2 = []
    for i in range(0, 5):
        vid = cv2.VideoCapture('C:/Users/anton/Desktop/ProgettoFVAB/Video_Train/VID_RGB_CUT_' + str(i) + '.avi')
        frame_1 = f.getAllFramesFromVideo(vid)
        frame_1 = frame_1[:60]
        list.append(frame_1)
        etichetta_1 = f.getEtichettaFromVideo(i,emozione)
        list_2.append(etichetta_1)
    X_train = np.array(list)
    X_train = X_train[:, 0, :, :, 0]
    Y_train = np.array(list_2)
    model.fit(X_train, Y_train)

def get_percentuale_emozione(vid):
    x_test=f.getAllFramesFromVideo(vid)
    x_test=x_test[:60]
    list_videos_Test=[]
    list_videos_Test.append(x_test)
    list_videos_Test=np.array(list_videos_Test)
    list_videos_Test=list_videos_Test[:,0,:,:,0]
    print(model.predict((list_videos_Test)))
