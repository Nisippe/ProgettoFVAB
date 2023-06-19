import cv2
import keras
import numpy as np
import Feature_Extractor as fe
import Funzioni as f
path='?'
path2='??'
path3='???'
videos='test videos da path'
for video in videos:
    vid=cv2.VideoCapture(path+video)
    fps = vid.get(cv2.CAP_PROP_FPS)
    width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
    codec = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter(path2 + str(video), codec, fps, (int(width), int(height)))
    f.video_cut(f.getAllFramesFromVideo(vid),out)

for video in videos:
    vid=cv2.VideoCapture(path2+video)
    fps = vid.get(cv2.CAP_PROP_FPS)
    width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
    codec = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter(path3+video, codec, fps, (int(width), int(height)))
    f.video_Draw_Landmarks(vid,out)

model_happy=keras.models.load_model('ModelloHappy.keras')
model_angry=keras.models.load_model('ModelloAngry.keras')
model_sad=keras.models.load_model('ModelloSad.keras')
model_neutral=keras.models.load_model('ModelloNeutral.keras')


for video in videos:
    vid=cv2.VideoCapture(path3+video)
    list=[]
    features=fe.extract_features_from_video(vid)
    list.append(features)
    list=np.array(list)
    print('VIDEO: '+str(video))
    print('% Happy: '+str(model_happy.predict(list)))
    print('% Angry: '+str(model_angry.predict(list)))
    print('% Sad: '+str(model_sad.predict(list)))
    print('% Neutral: '+str(model_neutral.predict(list)))
