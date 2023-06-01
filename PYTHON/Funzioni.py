import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False,model_complexity=1, min_detection_confidence=0.5,min_tracking_confidence=0.5)
df=pd.read_csv('../TXT/N_Frame_Tagliati.txt')

def getAllFramesFromVideo(vid):
        """
        Ottiene tutti i frame di un video
        :param vid: video (VideoCapture)
        :return: lista di tutti i frames del video
        """
        frames=[]
        while vid.isOpened():
            ret, image = vid.read()
            if ret is True:
                frames.append(image)
            else:
                break
        return frames

def getEtichettaFromVideo(n,emozione):
    """
    Ottiene la percentuale dell'n-esimo video data quell'emozione
    :param n: numero del video
    :param emozione: quale emozione scegliere (happy,angry,sad,neutral)
    :return: percentuale dell'n-esimo video data quell'emozione
    """
    p = pd.read_csv('../TXT/Etichette_Percentuali.txt')
    etichetta=p.loc[:,emozione]
    return etichetta[n]

def getAllVideo():
    """
    Ottiene una lista con i nomi di tutti i video
    :return: lista con i nomi di tutti i video
    """
    list=[]
    for i in range(0,10):
        list.append('VID_RGB_00'+str(i)+'.mp4')
    for i in range(10,61):
        list.append('VID_RGB_0'+str(i)+'.mp4')
    return list

def getAllVideoGait():
    list=[]
    for i in range(0,61):
        list.append('VID_RGB_CUT_'+str(i))
    return list


def video_Draw_Landmarks(vid,out):
    while vid.isOpened():
        ret, image = vid.read()
        if ret is True:
            results = pose.process(image)
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                out.write(image)

            cv2.waitKey(1)
        else:
            break
        out.release
def video_DrawCut_Landmarks(vid,out):
    """
    Disegna i landmarks sul video e lo salva
    :param vid: video (VideoCapture)
    :param out: VideoWriter
    """
    last_elbow_landmark = 0
    last_elbow2_landmark = 0
    frame=None

    frame_saltati=0
    while vid.isOpened():
        ret, image = vid.read()
        if ret is True:
            results = pose.process(image)
            if results.pose_landmarks:
                shoulder_sx = results.pose_landmarks.landmark[11]
                shoulder_dx = results.pose_landmarks.landmark[12]
                elbow_sx = results.pose_landmarks.landmark[13]
                elbow_dx = results.pose_landmarks.landmark[14]
                hip_sx = results.pose_landmarks.landmark[23]
                hip_dx = results.pose_landmarks.landmark[24]
                knee_sx = results.pose_landmarks.landmark[25]
                knee_dx = results.pose_landmarks.landmark[26]
                if last_elbow_landmark == 0:
                    last_elbow_landmark = elbow_sx
                    last_elbow2_landmark = elbow_dx
                else:
                    if conf_shoulder_distance(shoulder_sx,shoulder_dx) or conf_elbow(last_elbow_landmark,elbow_sx) or conf_elbow(last_elbow2_landmark,elbow_dx)\
                            or conf_hip_distance(hip_sx,hip_dx) or conf_knees_distance(knee_sx,knee_dx):
                        frame_saltati+=1
                    else:
                        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                        out.write(image)
                    last_elbow_landmark = elbow_sx
                    last_elbow2_landmark = elbow_dx



            cv2.waitKey(1)
        else:
            break
    df.loc[len(df)]=frame_saltati
    df.to_csv('C:/Users/anton/Desktop/ProgettoFVAB/TXT/N_Frame_Tagliati.txt', sep=',')
    out.release()


def conf_shoulder_distance(shoulder_sx,shoulder_dx):
    """
    Confronta la distanza tra la spalla sx e la spalla dx
    :param shoulder_sx: coordinate spalla sx
    :param shoulder_dx: coordinate spalla dx
    :return: True se la distanza tra le due spalle è minore di 0.12 altrimenti False
    """
    if(abs(shoulder_sx.x - shoulder_dx.x) <= 0.12):
        return True

def conf_hip_distance(hip_sx,hip_dx):
    """
    Confronta la distanza tra il fianco sx e il fianco dx
    :param hip_sx: coordinate fianco sx
    :param hip_dx: coordinate fianco dx
    :return: True se la distanza tra i due fianchi è minore di 0.04 altrimenti False
    """
    if (abs(hip_sx.x - hip_dx.x) <= 0.04):
        return True

def conf_knees_distance(knee_sx,knee_dx):
    """
    Confronta la distanza tra il ginocchio sx e il ginocchio dx
    :param knee_sx: coordinate ginocchio sx
    :param knee_dx: coordinate ginocchio dx
    :return: True se la distanza tra le due ginocchia è minore di 0.03 altrimenti False
    """
    if (abs(knee_sx.x - knee_dx.x) <= 0.03):
        return True

def conf_elbow(last_elbow_landmark, elbow):
    """
    Confronta la distanza tra il gomito del frame attuale e il gomito del frame precedente
    :param last_elbow_landmark: coordinate del gomito del frame precedente
    :param elbow: coordinate del gomito del frame attuale
    :return: True se la distanza tra le due è maggiore o uguale di 0.025 altrimenti False
    """
    if (abs(elbow.x - last_elbow_landmark.x) >= 0.025):
        return True

from skimage.metrics import structural_similarity as ssim

def compare(img1,img2):
    gray_image1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray_image2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Calcola l'indice SSIM
    similarity = ssim(gray_image1, gray_image2)

    # Restituisce l'indice SSIM
    return similarity


def video_cut(vid,out):
    frame=None
    i=0
    while vid.isOpened():
        ret, image = vid.read()
        if ret is True:
            if frame is None:
                frame=image
                out.write(image)
            else:
                i = i + 1
                if i > 10:
                    if compare(frame, image) > 0.92:
                        print('ma seriamente so uguali?')
                        return
                out.write(image)
                print(compare(frame, image))

        else:
            return



vid=cv2.VideoCapture('C:/Users/anton/Desktop/ProgettoFVAB/train+validation/VID_RGB_002.mp4')
fps = vid.get(cv2.CAP_PROP_FPS)
width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
codec = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
out = cv2.VideoWriter('C:/Users/anton/Desktop/ProgettoFVAB/Video_Train/VID_RGB_CUT_2GAIT'+'.avi', codec, fps, (int(width), int(height)))
video_cut(vid,out)