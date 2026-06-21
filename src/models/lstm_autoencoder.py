"""
LSTM Autoencoder : traite le signal ECG comme une séquence temporelle.
Plus puissant pour capturer les dépendances temporelles du signal cardiaque.
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_lstm_autoencoder(timesteps=187, features=1, lstm_units=[128, 64], encoding_dim=32):
    inp = keras.Input(shape=(timesteps, features))
    # Encodeur LSTM
    x = layers.LSTM(lstm_units[0], return_sequences=True)(inp)
    x = layers.LSTM(lstm_units[1], return_sequences=False)(x)
    encoded = layers.Dense(encoding_dim, activation="relu")(x)
    # Décodeur
    x = layers.RepeatVector(timesteps)(encoded)
    x = layers.LSTM(lstm_units[1], return_sequences=True)(x)
    x = layers.LSTM(lstm_units[0], return_sequences=True)(x)
    decoded = layers.TimeDistributed(layers.Dense(features, activation="sigmoid"))(x)

    model = keras.Model(inp, decoded, name="LSTM_AE")
    return model

def reshape_for_lstm(X):
    """(N, 187) → (N, 187, 1)"""
    return X.reshape(X.shape[0], X.shape[1], 1)

def reconstruction_error(model, X):
    X_3d = reshape_for_lstm(X)
    X_pred = model.predict(X_3d, verbose=0)
    return np.mean(np.power(X_3d - X_pred, 2), axis=(1, 2))