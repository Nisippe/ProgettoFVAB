import Funzioni as f
import cv2
videos=f.getAllVideo()
vid = cv2.VideoCapture('C:/Users/anton/Desktop/ProgettoFVAB/train+validation/' + videos[0])
fps = vid.get(cv2.CAP_PROP_FPS)
width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
codec = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
out = cv2.VideoWriter('Video_Cut_Landmarks/TEST'+'.mp4', codec, fps,(int(width), int(height)))
f.video_DrawCut_Landmarks(vid,out)

out = cv2.VideoWriter('Video_Gait_Landmarks/TEST'+'.mp4', codec, fps,(int(width), int(height)))