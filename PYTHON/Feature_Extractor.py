from keras import Sequential
from keras.layers import SimpleRNN, Dense
from tensorflow import keras
import cv2
import numpy as np
import Funzioni as f
IMG_WIDTH=1920
IMG_HEIGHT=1080


def build_feature_extractor():
    feature_extractor = keras.applications.InceptionV3(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMG_WIDTH, IMG_HEIGHT, 3),
    )
    preprocess_input = keras.applications.inception_v3.preprocess_input

    inputs = keras.Input((IMG_WIDTH, IMG_HEIGHT, 3))
    preprocessed = preprocess_input(inputs)

    outputs = feature_extractor(preprocessed)
    return keras.Model(inputs, outputs, name="feature_extractor")

NUM_FEATURES=2048
MAX_SEQ_LENGTH=22
def prepare_all_videos():
    num_samples = 61
    video_paths = f.getAllVideoGait()
    # `frame_masks` and `frame_features` are what we will feed to our sequence model.
    # `frame_masks` will contain a bunch of booleans denoting if a timestep is
    # masked with padding or not.
    frame_masks = np.zeros(shape=(num_samples, MAX_SEQ_LENGTH), dtype="bool")
    frame_features = np.zeros(
        shape=(num_samples, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32"
    )

    # For each video.
    for idx,video in enumerate(video_paths):
        vid = cv2.VideoCapture('Video_Cut_Landmarks/' + video)
        frames = f.getAllFramesFromVideo(vid)
        frames = np.array(frames)
        frames = frames[None, ...]

        # Initialize placeholders to store the masks and features of the current video.
        temp_frame_mask = np.zeros(shape=(1, MAX_SEQ_LENGTH,), dtype="bool")
        temp_frame_features = np.zeros(
            shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32"
        )

        # Extract features from the frames of the current video.
        for i, batch in enumerate(frames):
            video_length = batch.shape[0]
            length = min(MAX_SEQ_LENGTH, video_length)
            for j in range(length):
                temp_frame_features[i, j, :] = feature_extractor.predict(
                    batch[None, j, :]
                )
            temp_frame_mask[i, :length] = 1  # 1 = not masked, 0 = masked
        print(idx)
        frame_features[idx,] = temp_frame_features.squeeze()
        frame_masks[idx,] = temp_frame_mask.squeeze()

    return (frame_features, frame_masks)


def prepare_single_video(frames):
    frames = np.array(frames)
    frames = frames[None, ...]
    frame_mask = np.zeros(shape=(1, MAX_SEQ_LENGTH,), dtype="bool")
    frame_features = np.zeros(shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")

    for i, batch in enumerate(frames):
        video_length = batch.shape[0]
        length = min(MAX_SEQ_LENGTH, video_length)
        for j in range(length):
            frame_features[i, j, :] = feature_extractor.predict(batch[None, j, :])
        frame_mask[i, :length] = 1  # 1 = not masked, 0 = masked

    return frame_features, frame_mask

def create_model():
    frame_features_input=keras.Input((MAX_SEQ_LENGTH,NUM_FEATURES))
    mask_input=keras.Input((MAX_SEQ_LENGTH,),dtype='bool')
    print(frame_features_input.shape)
    print(mask_input.shape)

    model=Sequential()
    model.add(SimpleRNN(4, input_shape=(22,2048)))
    model.add(Dense(units=1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

    return model




def train_model(emozione):
    modello=create_model()
    x_train=prepare_all_videos()
    y_train=f.getListEtichettaFromVideos(emozione)
    y_train=np.array(y_train)
    modello.fit(x_train[0],y_train,validation_split=0.3,epochs=500)
    modello.save(str(emozione)+'Model.keras')
    return modello

def test_video(frames):
    modello1 = keras.models.load_model('HappyModel.keras')
    modello2 = keras.models.load_model('AngryModel.keras')
    modello3 = keras.models.load_model('SadModel.keras')
    modello4 = keras.models.load_model('NeutralModel.keras')
    frame_features, frame_mask=prepare_single_video(frames)
    print('Happy: '+ str(modello1.predict(frame_features)))
    print('Angry: '+ str(modello2.predict(frame_features)))
    print('Sad: ' + str(modello3.predict(frame_features)))
    print('Neutral: ' + str(modello4.predict(frame_features)))


feature_extractor = build_feature_extractor()
train_model('happy')
train_model('angry')
train_model('sad')
train_model('neutral')
videos=f.getAllTestVideos()
vid = cv2.VideoCapture('test/' + videos[1])
frames=f.video_DrawCut_Landmarks(vid)
frames=frames[:22]
test_video(frames)

