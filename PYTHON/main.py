import cv2
import keras
import numpy as np
import pandas as pd

import Feature_Extractor as fe
import Funzioni as f
import sys
import os




# Controlla se sono stati forniti gli argomenti corretti
if len(sys.argv) < 4:
    print("Usage: python main.py arg1 arg2 arg3")
    sys.exit(1)

# Leggi gli argomenti dalla linea di comando
arg1 = sys.argv[1] #path videoS
arg2 = sys.argv[2] #path csv
arg3 = sys.argv[3] #path progetto

def getEtichettaFromVideo(n,emozione):
    """
    Ottiene la percentuale dell'n-esimo video data quell'emozione
    :param n: numero del video
    :param emozione: quale emozione scegliere (happy,angry,sad,neutral)
    :return: percentuale dell'n-esimo video data quell'emozione
    """
    p = pd.read_csv(arg3+'/TXT/Etichette_Percentuali.txt')
    etichetta=p.loc[:,emozione]
    return etichetta[n]


def create_etichette_perc(medieColonne,etichette_perc,n):
    """
    Crea un file txt dove associa per ogni video le etichette (in percentuali)
    :param medieColonne: colonna medie colonne
    :param etichette_perc: file csv etichette_percentuali
    """
    val = 0
    for i in range(n):
        list_perc = []
        list=[]
        for j in range(0,4):
            list_perc.append(round(medieColonne[val+j],2))
        list.append('Vid'+str(i))
        for j in range(0,4):
            list.append(round(float((list_perc[j]/5)*100),1))
        print(list)
        etichette_perc.loc[i]=list
        etichette_perc.to_csv(arg3+'/TXT/Etichette_Percentuali_Test.txt',sep=',')
        val+=4

'''STATISTICHE'''
df=pd.read_csv(arg2)
df=df.apply(pd.to_numeric,errors='coerce')
df=df.drop('Gait ID', axis=1)

#Soluzione Medie Colonne
df2 = df.mean(axis = 0, skipna = True)
df2=pd.DataFrame(df2)
df2.rename(columns = {0:'mediaColonne'}, inplace = True)
print(df2)
df2.to_csv(arg3+'/TXT/MediaColonneTest.txt',sep=',')

cols=pd.read_csv(arg3+'/TXT/MediaColonneTest.txt')
medieColonne=cols['mediaColonne']
etichette_perc=pd.read_csv(arg3+'/TXT/Etichette_Percentuali_Test.txt')
num_video=len(medieColonne)/4
create_etichette_perc(medieColonne,etichette_perc,num_video)


'''VIDEOS'''
videos=os.listdir(arg1)
path2='cut'
path3='landmarks'
for video in videos:
    vid=cv2.VideoCapture(arg1+video)
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




'''MAIN'''
model_happy=keras.models.load_model('ModelloHappy.keras')
model_angry=keras.models.load_model('ModelloAngry.keras')
model_sad=keras.models.load_model('ModelloSad.keras')
model_neutral=keras.models.load_model('ModelloNeutral.keras')
happy_list=[]
angry_list=[]
sad_list=[]
neutral_list=[]
j=0
for video in videos:
    vid=cv2.VideoCapture(path3+video)
    list=[]
    features=fe.extract_features_from_video(vid)
    list.append(features)
    list=np.array(list)
    happy_t=f.getEtichettaFromVideo(j,'happy')
    angry_t=f.getEtichettaFromVideo(j,'angry')
    sad_t=f.getEtichettaFromVideo(j,'sad')
    neutral_t=f.getEtichettaFromVideo(j,'neutral')
    happy=model_happy.predict(list)
    angry=model_angry.predict(list)
    sad=model_sad.predict(list)
    neutral=model_neutral.predict(list)
    j=j+1
    print('VIDEO: '+str(video))
    print('% Happy: '+str(happy) + ' Differenza predizione/label: '+ round(abs(happy[0][0]-happy_t),2))
    print('% Angry: ' + str(angry) + ' Differenza predizione/label: ' + round(abs(angry[0][0] - angry_t), 2))
    print('% Sad: ' + str(sad) + ' Differenza predizione/label: ' + round(abs(sad[0][0] - sad_t), 2))
    print('% Neutral: ' + str(neutral) + ' Differenza predizione/label: ' + round(abs(neutral[0][0] - neutral_t), 2))



