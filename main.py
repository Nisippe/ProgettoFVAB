import cv2
import mediapipe as mp

# initialize Pose estimator
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False,model_complexity=0, min_detection_confidence=0.5,min_tracking_confidence=0.5)

vid=cv2.VideoCapture('train+validation/VID_RGB_033.mp4')
fps = vid.get(cv2.CAP_PROP_FPS)
print(fps)
width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
codec = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
out = cv2.VideoWriter('CSV/output_video1.avi',codec,fps,(int(width),int(height)))

last_shoulder_landmark = 0
last_shoulder2_landmark = 0
while vid.isOpened():
    # if frame is read correctly ret is True
    ret, image = vid.read()
    if ret is True:
        results=pose.process(image)
        if results.pose_landmarks:
            #mp_drawing.plot_landmarks(results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)
            shoulder_sx=results.pose_landmarks.landmark[11]
            shoulder_dx=results.pose_landmarks.landmark[12]
            if last_shoulder_landmark == 0 and last_shoulder2_landmark == 0:
                last_shoulder_landmark=shoulder_sx
                last_shoulder2_landmark=shoulder_dx
            else:
                #confronto
                if (abs(shoulder_sx.x - last_shoulder_landmark.x) >= 0.03) or (abs(shoulder_sx.y - last_shoulder_landmark.y) >= 0.03) or (abs(shoulder_sx.z - last_shoulder_landmark.z) >= 0.03):
                    pass
                if (abs(shoulder_dx.x - last_shoulder2_landmark.x) >= 0.03) or (abs(shoulder_dx.y - last_shoulder2_landmark.y) >= 0.03) or (abs(shoulder_dx.z - last_shoulder2_landmark.z) >= 0.03):
                    pass
                else:
                    print(shoulder_sx.z - last_shoulder_landmark.z)
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                    out.write(image)
                last_shoulder_landmark=shoulder_sx
                last_shoulder2_landmark=shoulder_dx

        cv2.waitKey(1)

    else:
        break

out.release()