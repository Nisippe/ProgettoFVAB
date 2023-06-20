import keras
import cv2
import Feature_Extractor as fe
import numpy as np
import pandas as pd
model_happy=keras.models.load_model('ModelloHappy.keras')
model_angry=keras.models.load_model('ModelloAngry.keras')
model_sad=keras.models.load_model('ModelloSad.keras')
model_neutral=keras.models.load_model('ModelloNeutral.keras')

def getEtichettaFromVideo(n,emozione):
    """
    Ottiene la percentuale dell'n-esimo video data quell'emozione
    :param n: numero del video
    :param emozione: quale emozione scegliere (happy,angry,sad,neutral)
    :return: percentuale dell'n-esimo video data quell'emozione
    """
    p = pd.read_csv('C:/Users/drugo/PycharmProjects/ProgettoFVAB/TXT/Etichette_Percentuali_Test.txt')
    etichetta=p.loc[:,emozione]
    return etichetta[n]
'''
path='C:/Users/drugo/PycharmProjects/ProgettoFVAB/Video_Test_Landmarks/'
happy_list=[]
angry_list=[]
sad_list=[]
neutral_list=[]
j=0

etichette_perc=pd.read_csv('C:/Users/drugo/PycharmProjects/ProgettoFVAB/TXT/DifferenzeTest.txt')
df=pd.DataFrame(etichette_perc)

for i in range(61,76):
    vid=cv2.VideoCapture(path+'VID_RGB_CUT_'+str(i)+'.mp4')
    list=[]
    features=fe.extract_features_from_video(vid)
    list.append(features)
    list=np.array(list)
    happy=model_happy.predict(list)
    angry=model_angry.predict(list)
    sad=model_sad.predict(list)
    neutral=model_neutral.predict(list)

    happy_t=getEtichettaFromVideo(j,'happy')
    angry_t=getEtichettaFromVideo(j,'angry')
    sad_t=getEtichettaFromVideo(j,'sad')
    neutral_t=getEtichettaFromVideo(j,'neutral')

    happy_list.append(round(abs(happy[0][0]-happy_t),2))
    angry_list.append(round(abs(angry[0][0] - angry_t),2))
    sad_list.append(round(abs(sad[0][0] - sad_t),2))
    neutral_list.append(round(abs(neutral[0][0] - neutral_t),2))

    j=j+1


    print('NUM_VIDEO: '+str(i))
    print('% Happy: '+str(happy))
    print('% Angry: '+str(angry))
    print('% Sad: '+str(sad))
    print('% Neutral: '+str(neutral))

df['happy']=happy_list
df['angry']=angry_list
df['sad']=sad_list
df['neutral']=neutral_list
print(df)
df.to_csv('C:/Users/drugo/PycharmProjects/ProgettoFVAB/TXT/DifferenzeTest.txt',sep=',')
'''

etichette_perc=pd.read_csv('C:/Users/drugo/PycharmProjects/ProgettoFVAB/TXT/DifferenzeTest.txt')
df=pd.DataFrame(etichette_perc)
mean_happy=np.mean(df['happy'])
mean_angry=np.mean(df['angry'])
mean_sad=np.mean(df['sad'])
mean_neutral=np.mean(df['neutral'])
print(df)
list=[]
list.append(mean_happy)
list.append(mean_angry)
list.append(mean_sad)
list.append(mean_neutral)
df.loc[len(df)]=list



std_happy=np.std(df['happy'])
std_angry=np.std(df['angry'])
std_sad=np.std(df['sad'])
std_neutral=np.std(df['neutral'])

list=[]
list.append(std_happy)
list.append(std_angry)
list.append(std_sad)
list.append(std_neutral)
df.loc[len(df)]=list

df.to_csv('C:/Users/drugo/PycharmProjects/ProgettoFVAB/TXT/DifferenzeTest.txt',sep=',')