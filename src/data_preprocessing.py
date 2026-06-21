"""
Chargement, nettoyage et préparation des données ECG.
Sépare les battements normaux (pour entraînement non supervisé)
des données de test complètes.
"""
import numpy as np
import pandas as pd
import yaml
import os
from sklearn.model_selection import train_test_split

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_mitbih(cfg):
    """Charge MIT-BIH. Retourne signaux + labels."""
    train_df = pd.read_csv(os.path.join(cfg["data"]["raw_dir"], "mitbih_train.csv"), header=None)
    test_df  = pd.read_csv(os.path.join(cfg["data"]["raw_dir"], "mitbih_test.csv"),  header=None)

    X_train_all = train_df.iloc[:, :187].values.astype(np.float32)
    y_train_all = train_df.iloc[:, 187].values.astype(int)

    X_test  = test_df.iloc[:, :187].values.astype(np.float32)
    y_test  = test_df.iloc[:, 187].values.astype(int)

    # Entraînement non supervisé : UNIQUEMENT classe normale (0)
    normal_mask = y_train_all == 0
    X_train_normal = X_train_all[normal_mask]

    return X_train_normal, X_test, y_test

def load_ptbdb(cfg):
    """Charge PTBDB (binaire: 0=normal, 1=anormal)."""
    abn = pd.read_csv(os.path.join(cfg["data"]["raw_dir"], "ptbdb_abnormal.csv"), header=None)
    nrm = pd.read_csv(os.path.join(cfg["data"]["raw_dir"], "ptbdb_normal.csv"),   header=None)

    df = pd.concat([nrm, abn], ignore_index=True).sample(frac=1, random_state=42)
    X = df.iloc[:, :187].values.astype(np.float32)
    y = df.iloc[:, 187].values.astype(int)
    return X, y

def save_processed(cfg, X_train_normal, X_test_mitbih, y_test_mitbih, X_ptb, y_ptb):
    out = cfg["data"]["processed_dir"]
    os.makedirs(out, exist_ok=True)
    np.save(os.path.join(out, "X_train_normal.npy"), X_train_normal)
    np.save(os.path.join(out, "X_test.npy"),         X_test_mitbih)
    np.save(os.path.join(out, "y_test.npy"),         y_test_mitbih)
    np.save(os.path.join(out, "X_ptb.npy"),          X_ptb)
    np.save(os.path.join(out, "y_ptb.npy"),          y_ptb)
    print(f"[✓] Données sauvegardées dans {out}/")
    print(f"    X_train_normal : {X_train_normal.shape}")
    print(f"    X_test (MIT-BIH): {X_test_mitbih.shape}")
    print(f"    X_ptb           : {X_ptb.shape}")

def run():
    cfg = load_config()
    print("[1/3] Chargement MIT-BIH...")
    X_train_normal, X_test, y_test = load_mitbih(cfg)
    print("[2/3] Chargement PTBDB...")
    X_ptb, y_ptb = load_ptbdb(cfg)
    print("[3/3] Sauvegarde...")
    save_processed(cfg, X_train_normal, X_test, y_test, X_ptb, y_ptb)

if __name__ == "__main__":
    run()