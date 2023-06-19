import cv2
import Funzioni as f


videos=f.getAllVideo()
path='C:/Users/drugo/PycharmProjects/ProgettoFVAB/train+validation/'
indice=0


#video_cut
for video in videos:
    vid=cv2.VideoCapture(path+video)
    fps = vid.get(cv2.CAP_PROP_FPS)
    width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
    codec = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter('C:/Users/drugo/PycharmProjects/ProgettoFVAB/Video_Cut/VID_RGB_CUT_'+str(indice)+'.mp4', codec, fps, (int(width), int(height)))
    frames=f.getAllFramesFromVideo(vid)
    f.video_cut_val(frames,out)
    indice=indice+1

path2='C:/Users/drugo/PycharmProjects/ProgettoFVAB/Video_Cut/'
videos2=f.getAllTestVideos()
indice=0

#video_landmarks
for video in videos2:
    vid=cv2.VideoCapture(path2+video)
    fps = vid.get(cv2.CAP_PROP_FPS)
    width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
    codec = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter('C:/Users/drugo/PycharmProjects/ProgettoFVAB/Video_Cut_Landmarks/VID_RGB_CUT_'+str(indice)+'.mp4', codec, fps, (int(width), int(height)))
    f.video_Draw_Landmarks(vid,out)
    indice=indice+1
