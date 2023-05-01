import mediapipe as mp
import cv2
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5,model_complexity=0,static_image_mode=False)

vid=cv2.VideoCapture('train+validation/VID_RGB_033.mp4')
fps = vid.get(cv2.CAP_PROP_FPS)
width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
codec = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
out = cv2.VideoWriter('CSV/output_video1.avi',codec,fps,(int(width),int(height)))

while vid.isOpened():
    # if frame is read correctly ret is True
    ret, image = vid.read()
    #image=cv2.resize(image,(1920,1080))
    if ret is True:
        results=holistic.process(image)
        #mp_drawing.draw_landmarks(image,results.left_hand_landmarks,mp_holistic.HAND_CONNECTIONS)
        #mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        cv2.imshow('mammt',image)
        out.write(image)
    else:
        break

out.release()