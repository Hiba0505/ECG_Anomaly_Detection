"""
Évaluation complète de tous les modèles sur MIT-BIH et PTBDB.
"""
import numpy as np
import os, yaml, joblib
import matplotlib
matplotlib.use('Agg')
import tensorflow as tf
from tensorflow import keras

from models.autoencoder      import reconstruction_error as ae_error, get_threshold, predict_anomaly
from models.lstm_autoencoder import reconstruction_error as lstm_error, reshape_for_lstm
from models.isolation_forest import fit_and_predict as if_predict
from utils import (evaluate_model, plot_roc_curves, plot_confusion_matrix,
                   plot_error_distribution, plot_reconstruction, save_results)

def load_config(path="config.yaml"):
    with open(path) as f: return yaml.safe_load(f)

def load_test_data(cfg, dataset="mitbih"):
    d = cfg["data"]["processed_dir"]
    if dataset == "mitbih":
        X = np.load(os.path.join(d, "X_test.npy"))
        y = np.load(os.path.join(d, "y_test.npy"))
    else:
        X = np.load(os.path.join(d, "X_ptb.npy"))
        y = np.load(os.path.join(d, "y_ptb.npy"))
    return X, y

def run():
    cfg = load_config()
    fig_dir = cfg["evaluation"]["figures_dir"]
    os.makedirs(fig_dir, exist_ok=True)

    for dataset_name in ["mitbih", "ptbdb"]:
        print(f"\n{'#'*60}")
        print(f"  ÉVALUATION SUR {dataset_name.upper()}")
        print(f"{'#'*60}")
        X_test, y_test = load_test_data(cfg, dataset_name)

        all_results   = {}
        metrics_list  = []
        perc = cfg["evaluation"]["threshold_percentile"]

        # --- 1. Autoencoder Dense ---
        ae = keras.models.load_model("outputs/models/autoencoder_dense.keras")
        errors_ae = ae_error(ae, X_test)
        threshold_ae = get_threshold(errors_ae, perc)
        met = evaluate_model(y_test, errors_ae, perc, "Autoencoder Dense")
        metrics_list.append(met)
        all_results["AE Dense"] = ((y_test != 0).astype(int), errors_ae)
        # Visualisations
        normal_idx = np.where(y_test == 0)[0][:500]
        anomaly_idx = np.where(y_test != 0)[0][:500]
        plot_error_distribution(
            errors_ae[normal_idx], errors_ae[anomaly_idx], threshold_ae,
            "Autoencoder Dense",
            save_path=f"{fig_dir}/{dataset_name}_ae_error_dist.png"
        )
        plot_confusion_matrix(
            (y_test != 0).astype(int),
            predict_anomaly(errors_ae, threshold_ae),
            "Autoencoder Dense",
            save_path=f"{fig_dir}/{dataset_name}_ae_confusion.png"
        )

        # --- 2. LSTM Autoencoder ---
        lstm_ae = keras.models.load_model("outputs/models/lstm_autoencoder.keras")
        errors_lstm = lstm_error(lstm_ae, X_test)
        threshold_lstm = get_threshold(errors_lstm, perc)
        met = evaluate_model(y_test, errors_lstm, perc, "LSTM Autoencoder")
        metrics_list.append(met)
        all_results["LSTM AE"] = ((y_test != 0).astype(int), errors_lstm)
        plot_error_distribution(
            errors_lstm[normal_idx], errors_lstm[anomaly_idx], threshold_lstm,
            "LSTM Autoencoder",
            save_path=f"{fig_dir}/{dataset_name}_lstm_error_dist.png"
        )

        # --- 3. Isolation Forest ---
        if_model = joblib.load("outputs/models/isolation_forest.pkl")
        scores_if, preds_if = fit_and_predict = (
            -if_model.score_samples(X_test),
            (if_model.predict(X_test) == -1).astype(int)
        )
        met = evaluate_model(y_test, scores_if, perc, "Isolation Forest")
        metrics_list.append(met)
        all_results["Isolation Forest"] = ((y_test != 0).astype(int), scores_if)

        # --- 4. One-Class SVM ---
        ocsvm_data = joblib.load("outputs/models/ocsvm.pkl")
        ocsvm_model, scaler = ocsvm_data["model"], ocsvm_data["scaler"]
        X_test_s = scaler.transform(X_test)
        scores_svm = -ocsvm_model.score_samples(X_test_s)
        met = evaluate_model(y_test, scores_svm, perc, "One-Class SVM")
        metrics_list.append(met)
        all_results["OC-SVM"] = ((y_test != 0).astype(int), scores_svm)

        # --- Courbes ROC comparatives ---
        plot_roc_curves(all_results,
                        save_path=f"{fig_dir}/{dataset_name}_roc_comparison.png")

        save_results(metrics_list,
                     path=f"outputs/results/{dataset_name}_metrics.json")

    print("\n[✓✓] Évaluation complète terminée !")

if __name__ == "__main__":
    run()