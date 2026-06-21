"""
Script principal d'entraînement de tous les modèles.
Ordre d'exécution : APRÈS data_preprocessing.py
"""
import numpy as np
import os, yaml
import tensorflow as tf
from tensorflow import keras

from models.autoencoder      import build_autoencoder
from models.lstm_autoencoder import build_lstm_autoencoder, reshape_for_lstm
from models.isolation_forest import build_isolation_forest, save_model as save_if
from models.one_class_svm    import build_ocsvm, fit_and_predict as ocsvm_fit
import joblib

def load_config(path="config.yaml"):
    with open(path) as f: return yaml.safe_load(f)

def load_data(cfg):
    d = cfg["data"]["processed_dir"]
    return np.load(os.path.join(d, "X_train_normal.npy"))

def train_autoencoder(X_train, cfg):
    print("\n[AE Dense] Entraînement...")
    c = cfg["autoencoder"]; t = cfg["training"]
    ae, enc = build_autoencoder(
        input_dim=187,
        encoding_dim=c["encoding_dim"],
        hidden=c["hidden_layers"],
        dropout=c["dropout"]
    )
    ae.compile(optimizer=keras.optimizers.Adam(t["learning_rate"]), loss="mse")
    ae.summary()
    history = ae.fit(
        X_train, X_train,
        epochs=t["epochs"],
        batch_size=t["batch_size"],
        validation_split=t["validation_split"],
        callbacks=[
            keras.callbacks.EarlyStopping(patience=7, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(patience=4, factor=0.5)
        ],
        verbose=1
    )
    os.makedirs("outputs/models", exist_ok=True)
    ae.save("outputs/models/autoencoder_dense.keras")
    print("[✓] Autoencoder Dense sauvegardé.")
    return ae, history

def train_lstm_autoencoder(X_train, cfg):
    print("\n[LSTM AE] Entraînement...")
    c = cfg["lstm_autoencoder"]; t = cfg["training"]
    model = build_lstm_autoencoder(
        timesteps=187, features=1,
        lstm_units=c["lstm_units"],
        encoding_dim=c["encoding_dim"]
    )
    model.compile(optimizer=keras.optimizers.Adam(t["learning_rate"]), loss="mse")
    model.summary()
    X_3d = reshape_for_lstm(X_train)
    history = model.fit(
        X_3d, X_3d,
        epochs=t["epochs"],
        batch_size=t["batch_size"],
        validation_split=t["validation_split"],
        callbacks=[
            keras.callbacks.EarlyStopping(patience=7, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(patience=4, factor=0.5)
        ],
        verbose=1
    )
    model.save("outputs/models/lstm_autoencoder.keras")
    print("[✓] LSTM Autoencoder sauvegardé.")
    return model, history

def train_isolation_forest(X_train, cfg):
    print("\n[Isolation Forest] Entraînement...")
    c = cfg["isolation_forest"]
    model = build_isolation_forest(
        n_estimators=c["n_estimators"],
        contamination=c["contamination"],
        random_state=c["random_state"]
    )
    model.fit(X_train)
    save_if(model)
    print("[✓] Isolation Forest sauvegardé.")
    return model

def train_ocsvm(X_train, cfg):
    print("\n[One-Class SVM] Entraînement (sur sous-échantillon 10k)...")
    c = cfg["one_class_svm"]
    # OCSVM ne passe pas à l'échelle : sous-échantillonnage
    idx = np.random.choice(len(X_train), size=min(10000, len(X_train)), replace=False)
    X_sub = X_train[idx]
    model = build_ocsvm(kernel=c["kernel"], nu=c["nu"], gamma=c["gamma"])
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_sub_s = scaler.fit_transform(X_sub)
    model.fit(X_sub_s)
    joblib.dump({"model": model, "scaler": scaler}, "outputs/models/ocsvm.pkl")
    print("[✓] One-Class SVM sauvegardé.")
    return model, scaler

def run():
    cfg = load_config()
    X_train = load_data(cfg)
    print(f"[INFO] X_train_normal : {X_train.shape}")

    train_autoencoder(X_train, cfg)
    train_lstm_autoencoder(X_train, cfg)
    train_isolation_forest(X_train, cfg)
    train_ocsvm(X_train, cfg)
    print("\n[✓✓] Tous les modèles entraînés et sauvegardés !")

if __name__ == "__main__":
    run()