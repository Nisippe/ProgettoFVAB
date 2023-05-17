'''
1.Video,happy
2.
'''


import numpy as np
import tensorflow
from keras import Sequential
from keras.layers import LSTM, Dense
import cv2
import Funzioni as f
from sklearn.linear_model import LinearRegression

labels={'Happy': 0, 'Angry': 1, 'Sad': 2, 'Neutral': 3}
Lista_5_video=[]
Lista_5_video.append('VID_RGB_CUT_0.avi')
Lista_5_video.append('VID_RGB_CUT_1.avi')
Lista_5_video.append('VID_RGB_CUT_2.avi')
Lista_5_video.append('VID_RGB_CUT_3.avi')
Lista_5_video.append('VID_RGB_CUT_4.avi')
Lista_video_frames=[]
Lista_video_happy=[]
window=[]

for video in Lista_5_video:
    vid=cv2.VideoCapture('../Video_Train/'+video)
    frames=f.getAllFramesFromVideo(vid)
    frames=np.stack(frames,axis=0)
    window.append(frames)

window=np.array(window)
X=window

for i in range(0,5):
    Lista_video_happy.append(f.getEtichettaFromVideo(i,'happy'))
#etichette % happy
Y=np.array(Lista_video_happy)
print(X.shape)
print(Y.shape)

model = LinearRegression()


model.fit(X, Y)

