import cv2
import mediapipe as mp
import pandas as pd

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False,model_complexity=0, min_detection_confidence=0.5,min_tracking_confidence=0.5)


def getAllFramesFromVideo(vid):
        frames=[]
        n=0
        while vid.isOpened():
            # if frame is read correctly ret is True
            ret, image = vid.read()
            if ret is True:
                frames.append(image)
                n+=1
                #print(str(image) + "mammt")
            else:
                break
        print(n)
        return frames

def getEtichettaFromVideo(n,emozione):
    p = pd.read_csv('C:/Users/drugo/PycharmProjects/ProgettoFVAB/TXT/Etichette_Percentuali.txt')
    palle=p.loc[:,emozione]
    return palle[n]

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
    while vid.isOpened():
        # if frame is read correctly ret is True
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
                        pass
                    else:
                        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                        out.write(image)
                    last_elbow_landmark = elbow_sx
                    last_elbow2_landmark = elbow_dx


            cv2.waitKey(1)

        else:
            break

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

'''def conf_wrist(last_wrist_landmark, wrist):
    if (abs(wrist.x - last_wrist_landmark.x) >= 0.10):
        return True'''

