import cv2
import Funzioni as f

videos=f.getAllTestVideos()
path='C:/Users/drugo/PycharmProjects/ProgettoFVAB/test/'
indice=61
for video in videos:
    vid=cv2.VideoCapture(path+video)
    fps = vid.get(cv2.CAP_PROP_FPS)
    width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
    codec = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter('C:/Users/drugo/PycharmProjects/ProgettoFVAB/Video_Cut_Test/VID_RGB_CUT_' + str(indice) + '.mp4', codec, fps, (int(width), int(height)))
    frames=f.getAllFramesFromVideo(vid)
    f.video_cut_val(frames, out)
    indice = indice + 1


path2='C:/Users/drugo/PycharmProjects/ProgettoFVAB/Video_Cut_Test/'
for i in range(61,76):
    vid=cv2.VideoCapture(path2+'VID_RGB_CUT_'+str(i)+'.mp4')
    fps = vid.get(cv2.CAP_PROP_FPS)
    width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
    codec = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter('C:/Users/drugo/PycharmProjects/ProgettoFVAB/Video_Test_Landmarks/VID_RGB_CUT_' + str(i) + '.mp4', codec, fps, (int(width), int(height)))
    f.video_Draw_Landmarks(vid,out)



