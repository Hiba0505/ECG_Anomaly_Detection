"""
Autoencoder Dense (MLP) pour détection d'anomalies ECG.
Principe : entraîné sur normaux → erreur de reconstruction
élevée sur les anomalies.
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import yaml, os

def build_autoencoder(input_dim=187, encoding_dim=32, hidden=[128, 64, 32], dropout=0.2):
    inp = keras.Input(shape=(input_dim,))
    x = inp
    # Encodeur
    for units in hidden:
        x = layers.Dense(units, activation="relu")(x)
        x = layers.Dropout(dropout)(x)
    encoded = layers.Dense(encoding_dim, activation="relu", name="bottleneck")(x)
    # Décodeur
    for units in reversed(hidden):
        x = layers.Dense(units, activation="relu")(encoded if x is encoded else x)
        x = layers.Dropout(dropout)(x)
    decoded = layers.Dense(input_dim, activation="sigmoid")(x)

    autoencoder = keras.Model(inp, decoded, name="AE_Dense")
    encoder     = keras.Model(inp, encoded, name="Encoder")
    return autoencoder, encoder

def reconstruction_error(model, X):
    X_pred = model.predict(X, verbose=0)
    return np.mean(np.power(X - X_pred, 2), axis=1)  # MSE par sample

def get_threshold(errors, percentile=95):
    return np.percentile(errors, percentile)

def predict_anomaly(errors, threshold):
    return (errors > threshold).astype(int)