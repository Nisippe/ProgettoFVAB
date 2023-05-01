import cv2
import mediapipe as mp

# initialize Pose estimator
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False,min_detection_confidence=0.7,min_tracking_confidence=0.7)

vid=cv2.VideoCapture('train+validation/VID_RGB_000.mp4')
fps = vid.get(cv2.CAP_PROP_FPS)
width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
codec = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
#frameSize = (1920,1080)
out = cv2.VideoWriter('CSV/output_video.avi',codec,fps,(int(width),int(height)))
while vid.isOpened():
    # if frame is read correctly ret is True
    ret, image = vid.read()
    #image=cv2.resize(image,(1920,1080))
    if ret is True:
        results=pose.process(image)
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image=image, landmark_list=results.pose_landmarks,
                                      connections=mp_pose.POSE_CONNECTIONS,
                                      landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255),
                                                                                   thickness=3, circle_radius=3),
                                      connection_drawing_spec=mp_drawing.DrawingSpec(color=(49, 125, 237),
                                                                                     thickness=2, circle_radius=2))

        #mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        out.write(image)
    else:
        break

out.release()