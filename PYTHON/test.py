import keras
import cv2
import Feature_Extractor as fe
import numpy as np
modello=keras.models.load_model('Modello.keras')
path='C:/Users/drugo/PycharmProjects/ProgettoFVAB/Video_Test_Landmarks/'

for i in range(61,76):
    vid=cv2.VideoCapture(path+'VID_RGB_CUT_'+str(i)+'.mp4')
    list=[]
    features=fe.extract_features_from_video(vid)
    list.append(features)
    list=np.array(list)
    print(list.shape)
    print(modello.predict(list))