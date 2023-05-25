import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from PIL import Image

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False,model_complexity=1, min_detection_confidence=0.5,min_tracking_confidence=0.5)
df=pd.read_csv('C:/Users/anton/Desktop/ProgettoFVAB/TXT/N_Frame_Tagliati.txt')



def extract_Keypoints(vid):
    width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
    list=[]
    list_x_y=[]
    while vid.isOpened():
        ret, image = vid.read()
        if ret is True:
            results = pose.process(image)
            x_coordinate = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * width
            y_coordinate = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * height
            list_x_y.append(x_coordinate)
            list_x_y.append(y_coordinate)
            list.append(list_x_y)
        else:
            break
    return list
def getAllFramesFromVideo(vid):
        frames=[]
        n=0
        while vid.isOpened():
            ret, image = vid.read()
            if ret is True:
                frames.append(image)
                n+=1
            else:
                break
        return frames

def getEtichettaFromVideo(n,emozione):
    p = pd.read_csv('C:/Users/anton/Desktop/ProgettoFVAB/TXT/Etichette_Percentuali.txt')
    etichetta=p.loc[:,emozione]
    return etichetta[n]

def getAllVideo():
    '''
    0-9 VID_RGB_00 + N = VID_RGB_001
    10-60 VID_RGB_0 + N = VID_RGB_010
    '''
    list=[]
    for i in range(0,10):
        list.append('VID_RGB_00'+str(i)+'.mp4')
    for i in range(10,61):
        list.append('VID_RGB_0'+str(i)+'.mp4')
    return list

def video_Draw_Landmarks(vid,out):
    last_elbow_landmark = 0
    last_elbow2_landmark = 0
    frame_saltati=0
    image_t=np.zeros((1920,1080,3),np.uint8)
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
                        #pass
                    else:
                        mp_drawing.draw_landmarks(image_t, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                        out.write(image_t)
                        image_t = np.zeros((1920, 1080, 3), np.uint8)
                        #cv2.imshow('cazzi',image_t)
                    last_elbow_landmark = elbow_sx
                    last_elbow2_landmark = elbow_dx


            cv2.waitKey(1)

        else:
            break
    print(frame_saltati)
    df.loc[len(df)]=frame_saltati
    df.to_csv('C:/Users/anton/Desktop/ProgettoFVAB/TXT/N_Frame_Tagliati.txt', sep=',')
    out.release()


def conf_shoulder_distance(shoulder_sx,shoulder_dx):
    if(abs(shoulder_sx.x - shoulder_dx.x) <= 0.12):
        return True

def conf_hip_distance(hip_sx,hip_dx):
    if (abs(hip_sx.x - hip_dx.x) <= 0.04):
        return True

def conf_knees_distance(knee_sx,knee_dx):
    if (abs(knee_sx.x - knee_dx.x) <= 0.03):
        return True

def conf_elbow(last_elbow_landmark, elbow):
    if (abs(elbow.x - last_elbow_landmark.x) >= 0.025):
        return True


vid=cv2.VideoCapture('C:/Users/anton/Desktop/ProgettoFVAB/train+validation/VID_RGB_000.mp4')
fps = vid.get(cv2.CAP_PROP_FPS)
width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
codec = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
out = cv2.VideoWriter('C:/Users/anton/Desktop/ProgettoFVAB/Video_Train/test.avi', codec, fps, (int(width), int(height)))
video_Draw_Landmarks(vid,out)