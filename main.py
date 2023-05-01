import cv2
import mediapipe as mp

# initialize Pose estimator
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False,model_complexity=0, min_detection_confidence=0.5,min_tracking_confidence=0.5)

vid=cv2.VideoCapture('train+validation/VID_RGB_033.mp4')
fps = vid.get(cv2.CAP_PROP_FPS)
width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
codec = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
out = cv2.VideoWriter('CSV/output_video1.avi',codec,fps,(int(width),int(height)))
while vid.isOpened():
    # if frame is read correctly ret is True
    ret, image = vid.read()

    if ret is True:
        results=pose.process(image)
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            mp_drawing.plot_landmarks(results.pose_world_landmarks.LEFT_SHOULDER, mp_pose.POSE_CONNECTIONS)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image=cv2.resize(image,(1920,1080))

        cv2.waitKey(1)
        out.write(image)
    else:
        break

out.release()