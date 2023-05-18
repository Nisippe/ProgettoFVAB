'''
1.Video,happy
2.
'''


import numpy as np
import tensorflow
from keras import Sequential
import cv2
import Funzioni as f
from sklearn.linear_model import LinearRegression

Lista_5_video=[]
Lista_5_video.append('VID_RGB_CUT_0.avi')
Lista_5_video.append('VID_RGB_CUT_1.avi')
Lista_5_video.append('VID_RGB_CUT_2.avi')
Lista_5_video.append('VID_RGB_CUT_3.avi')
Lista_5_video.append('VID_RGB_CUT_4.avi')
Lista_video_frames=[]
Lista_video_happy=[]

for video in Lista_5_video:
    vid=cv2.VideoCapture('../Video_Train/'+video)
    frames=f.getAllFramesFromVideo(vid)
    list_key=f.extract_Keypoints(vid)
    Lista_video_frames.append(np.array(list_key))

X=np.array(Lista_video_frames)

'''
x=[[],[],[]]


y=[42.0,69.0,700.0]
'''

for i in range(0,5):
    Lista_video_happy.append(f.getEtichettaFromVideo(i,'happy'))
#etichette % happy
Y=np.array(Lista_video_happy)
print(X.shape)
print(Y.shape)

model = LinearRegression()
model.fit(X, Y)

