import keras
import matplotlib.pyplot as plt
from keras import Sequential
import Feature_Extractor as fe
import Funzioni as f
import numpy as np
import cv2



def create_X_train():
    """
    Crea e salva X_train(Features dei video di train)
    """
    videos = f.getAllVideoGait()
    path = 'C:/Users/drugo/PycharmProjects/ProgettoFVAB/Video_Cut_Landmarks/'
    list_features = []
    for video in videos:
        vid = cv2.VideoCapture(path + video)
        features = fe.extract_features_from_video(vid)
        list_features.append(features)
    X_train = np.array(list_features)
    np.save('X_train.npy', X_train)

def create_model():
    """
    Crea il modello
    :return: ritorna il modello
    """
    model = Sequential()
    inputs = keras.layers.Input(shape=(24, 25088))
    lstm_out = keras.layers.LSTM(512)(inputs)
    dense_out = keras.layers.Dense(256)(lstm_out)
    dense2_out = keras.layers.Dense(128)(dense_out)
    dense3_out = keras.layers.Dense(64)(dense2_out)
    dense4_out = keras.layers.Dense(32)(dense3_out)
    outputs = keras.layers.Dense(1)(dense4_out)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mae', metrics=['mse'])
    model.summary()
    return model

def train_model(model,emozione):
    """
    Allena il modello
    :param model: modello
    :param emozione: quale emozione scegliere (happy,angry,sad,neutral)
    """
    X_train = np.load('X_train.npy')
    y_train = f.getListEtichettaFromVideos(emozione)
    y_train = np.array(y_train)
    history = model.fit(X_train, y_train, epochs=100, validation_data=(X_train, y_train))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.legend()
    plt.show()
    #model.save('Modello'+str(emozione)+'.keras')