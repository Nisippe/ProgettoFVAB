import cv2
import mediapipe as mp
import pandas as pd
from skimage.metrics import structural_similarity as ssim



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

def getListEtichettaFromVideos(emozione):
    """
    Ottiene la lista delle percentuali dei video data quell'emozione
    :param emozione: quale emozione scegliere (happy,angry,sad,neutral)
    :return: la lista delle percentuali
    """
    list=[]
    for i in range(0,61):
        list.append(getEtichettaFromVideo(i,emozione))
    return list

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
    """
    Ottiene una lista con i nomi di tutti i video gait/landmarks
    :return: lista con i nomi di tutti i video gait/landmarks
    """
    list=[]
    for i in range(0,61):
        list.append('VID_RGB_CUT_'+str(i)+'.mp4')
    return list

def getAllTestVideos():
    """
    Ottiene una lista con i nomi di tutti i video test
    :return: lista con i nomi di tutti i video test
    """
    list = []
    for i in range(61, 76):
        list.append('VID_RGB_0' + str(i) + '.mp4')
    return list

def video_cut(frames,out):
    """
    Confronta il primo frame con l'inizio di un nuovo ciclo di gait e lo taglia
    :param frames: frames di un video
    :param out: VideoWriter
    """
    frame_iniziale=frames[0]
    i=0
    for frame in frames:
        i=i+1
        if compare(frame_iniziale, frame) > 0.90 and i>25:
            return i
        else:
            out.write(frame)
    cv2.waitKey(1)
    out.release()

def video_cut_val(frames,out):
    """
    Confronta il primo frame con l'inizio di un nuovo ciclo di gait e lo taglia
    :param frames: frames di un video
    :param out: VideoWriter
    """
    i=0
    for frame in frames:
        i=i+1
        if i==25:
            return i
        else:
            out.write(frame)
    cv2.waitKey(1)
    out.release()

def video_Draw_Landmarks(vid,out):
    """
    Disegna i landmarks sul video e lo salva
    :param vid: video (VideoCapture)
    :param out: VideoWriter
    """
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5,min_tracking_confidence=0.5)
    while vid.isOpened():
        ret, image = vid.read()
        if ret is True:
            results = pose.process(image)
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            out.write(image)
        else:
            break
        cv2.waitKey(1)

def compare(img1,img2):
    """
    Compara due immagini
    :param img1: immagine 1
    :param img2: immagine 2
    :return: ritorna l'indice di similarità tra le due immagini
    """
    gray_image1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray_image2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Calcola l'indice SSIM
    similarity = ssim(gray_image1, gray_image2)
    print(similarity)
    # Restituisce l'indice SSIM
    return similarity







