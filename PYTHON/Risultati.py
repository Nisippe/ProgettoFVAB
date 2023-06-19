import keras
import cv2
import Feature_Extractor as fe
import numpy as np
model_happy=keras.models.load_model('ModelloHappy.keras')
model_angry=keras.models.load_model('ModelloAngry.keras')
model_sad=keras.models.load_model('ModelloSad.keras')
model_neutral=keras.models.load_model('ModelloNeutral.keras')

path='C:/Users/drugo/PycharmProjects/ProgettoFVAB/Video_Test_Landmarks/'

for i in range(61,76):
    vid=cv2.VideoCapture(path+'VID_RGB_CUT_'+str(i)+'.mp4')
    list=[]
    features=fe.extract_features_from_video(vid)
    list.append(features)
    list=np.array(list)
    print('NUM_VIDEO: '+str(i))
    print('% Happy: '+str(model_happy.predict(list)))
    print('% Angry: '+str(model_angry.predict(list)))
    print('% Sad: '+str(model_sad.predict(list)))
    print('% Neutral: '+str(model_neutral.predict(list)))