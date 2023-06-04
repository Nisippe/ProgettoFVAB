import cv2
import Funzioni as f


list=f.getAllVideo()
indice_video=0
max=0
min=100
frames=[]

for video in list:
    vid=cv2.VideoCapture('train+validation/'+video)
    fps = vid.get(cv2.CAP_PROP_FPS)
    width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
    codec = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter('Video_Cut_Landmarks/VID_RGB_CUT_'+str(indice_video)+'.mp4', codec, fps, (int(width), int(height)))
    frames=f.video_DrawCut_Landmarks(vid)
    i=f.video_cut_val(frames,out)
    if i > max:
        max=i
    if i < min:
        min=i
    indice_video+=1

print('MAX: '+str(max))
print('MIN: '+str(min))

