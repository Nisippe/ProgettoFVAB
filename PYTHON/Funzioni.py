import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False,model_complexity=0, min_detection_confidence=0.5,min_tracking_confidence=0.5)

vid=cv2.VideoCapture('train+validation/VID_RGB_024.mp4')
fps = vid.get(cv2.CAP_PROP_FPS)
width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
codec = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
out = cv2.VideoWriter('Video_Train/VID_RGB_CUT_24.avi', codec, fps, (int(width), int(height)))
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
    last_shoulder_landmark = 0
    last_elbow_landmark = 0
    last_hip_landmark = 0
    last_shoulder2_landmark = 0
    last_elbow2_landmark = 0
    last_hip2_landmark = 0
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
                if last_shoulder_landmark == 0 and last_shoulder2_landmark == 0 and last_elbow_landmark == 0 and last_elbow2_landmark == 0:
                    #Controllo su tutti e 4 ridondante
                    last_shoulder_landmark = shoulder_sx
                    last_shoulder2_landmark = shoulder_dx
                    last_elbow_landmark = elbow_sx
                    last_elbow2_landmark = elbow_dx
                    last_hip_landmark = hip_sx
                    last_hip2_landmark = hip_dx
                else:

                    # confronto shoulder_sx
                    '''if (abs(shoulder_sx.x - last_shoulder_landmark.x) >= 0.02) or (
                            abs(shoulder_sx.y - last_shoulder_landmark.y) >= 0.02) or (
                            abs(shoulder_sx.z - last_shoulder_landmark.z) >= 0.02):
                        pass
                    # confronto shoulder_dx
                    if (abs(shoulder_dx.x - last_shoulder2_landmark.x) >= 0.02) or (
                            abs(shoulder_dx.y - last_shoulder2_landmark.y) >= 0.02) or (
                            abs(shoulder_dx.z - last_shoulder2_landmark.z) >= 0.02):
                        pass
                        
                    if (abs(elbow_sx.x - last_elbow_landmark.x) >= 0.01) or (
                            abs(elbow_sx.y - last_elbow_landmark.y) >= 0.01) or (
                            abs(elbow_sx.z - last_elbow_landmark.z) >= 0.01):
                        print(i)
                        pass
                    # confronto elbow_dx
                    if (abs(elbow_dx.x - last_elbow2_landmark.x) >= 0.01) or (
                            abs(elbow_dx.y - last_elbow2_landmark.y) >= 0.01) or (
                            abs(elbow_dx.z - last_elbow2_landmark.z) >= 0.01):
                        print(i)
                        pass
                        '''
                    if conf_shoulder_distance(shoulder_sx,shoulder_dx) or conf_elbow(last_elbow_landmark,elbow_sx) or conf_elbow(last_elbow2_landmark,elbow_dx)\
                            or conf_hip(last_hip_landmark,hip_sx) or conf_hip(last_hip2_landmark,hip_dx):
                        pass
                    else:
                        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                        out.write(image)
                    last_shoulder_landmark = shoulder_sx
                    last_shoulder2_landmark = shoulder_dx
                    last_elbow_landmark = elbow_sx
                    last_elbow2_landmark = elbow_dx
                    last_hip_landmark = hip_sx
                    last_hip2_landmark = hip_dx

            cv2.waitKey(1)

        else:
            break

    out.release()

def conf_shoulder(last_shoulder_landmark, shoulder):
    if (abs(shoulder.x - last_shoulder_landmark.x) >= 0.05) or (
            abs(shoulder.y - last_shoulder_landmark.y) >= 0.05) or (
            abs(shoulder.z - last_shoulder_landmark.z) >= 0.05):
        return True

def conf_shoulder_distance(shoulder_sx,shoulder_dx):
    if(abs(shoulder_sx.x - shoulder_dx.x) <= 0.12):
        return True

def conf_hip(last_hip_landmark,hip):
    if (abs(hip.x - last_hip_landmark.x) >= 0.07) or (abs(hip.y - last_hip_landmark.y) >= 0.07) or (abs(hip.z - last_hip_landmark.z) >= 0.07):
        return True

def conf_elbow(last_elbow_landmark, elbow):
    if (abs(elbow.x - last_elbow_landmark.x) >= 0.04) or (abs(elbow.y - last_elbow_landmark.y) >= 0.04) or (abs(elbow.z - last_elbow_landmark.z) >= 0.04):
        return True



video_Draw_Landmarks(vid,out)