import cv2
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import Funzioni as f

# Carica il modello di rete neurale convoluzionale preaddestrato (es. VGG16)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))


# Estrai le features dai fotogrammi del video utilizzando il modello
def extract_features_from_video(vid):
    frames = f.getAllFramesFromVideo(vid)
    list_frames=[]
    for frame in frames:
        frame = cv2.resize(frame, (224, 224))
        frame = preprocess_input(frame)
        list_frames.append(frame)
    frames = np.array(list_frames)
    features = base_model.predict(frames)
    features = features.reshape(features.shape[0], -1)
    return features


