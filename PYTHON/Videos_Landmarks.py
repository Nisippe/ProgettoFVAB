import cv2
import Funzioni as f


list=f.getAllVideo()
indice_video=0
for video in list:
    vid=cv2.VideoCapture('train+validation/'+video)
    fps = vid.get(cv2.CAP_PROP_FPS)
    width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
    codec = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    out = cv2.VideoWriter('Video_Train/VID_RGB_CUT_'+str(indice_video)+'.avi', codec, fps, (int(width), int(height)))
    f.video_Draw_Landmarks(vid,out)
    indice_video+=1
