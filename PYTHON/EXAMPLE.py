from statistics import mean
import matplotlib.pyplot as plt
import keras
import cv2
import Feature_Extractor as fe
import numpy as np
import Funzioni as f
modello=keras.models.load_model('Modello.keras')
path='C:/Users/drugo/PycharmProjects/ProgettoFVAB/Video_Test_Landmarks/'

vid = cv2.VideoCapture(path + 'VID_RGB_CUT_61.mp4')
list = []
features = fe.extract_features_from_video(vid)
list.append(features)
list = np.array(list)
print(list.shape)
predicted_values=modello.predict(list)
print(predicted_values)
