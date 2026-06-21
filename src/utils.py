"""
Fonctions utilitaires : visualisation, sauvegarde figures, logging.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os, json
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              classification_report, confusion_matrix,
                              roc_curve, precision_recall_curve)

def plot_ecg_samples(X, y, n=5, title="Exemples ECG", save_path=None):
    classes = {0: "Normal", 1: "Supraventriculaire",
               2: "Ventriculaire", 3: "Fusion", 4: "Autre"}
    fig, axes = plt.subplots(n, 1, figsize=(14, 2.5*n))
    for i in range(n):
        axes[i].plot(X[i], linewidth=1.2, color="steelblue")
        axes[i].set_title(f"Classe {y[i]} : {classes.get(y[i], '?')}", fontsize=10)
        axes[i].set_ylabel("Amplitude")
        axes[i].grid(True, alpha=0.3)
    plt.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.close()

def plot_reconstruction(original, reconstructed, label, save_path=None):
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(original,      label="Original",    linewidth=1.5, color="steelblue")
    ax.plot(reconstructed, label="Reconstruit", linewidth=1.5, color="tomato", linestyle="--")
    ax.set_title(f"Reconstruction ECG — Label: {label}", fontsize=11)
    ax.legend(); ax.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

def plot_error_distribution(errors_normal, errors_anomaly, threshold, model_name, save_path=None):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(errors_normal,  bins=80, alpha=0.6, label="Normal",  color="steelblue", density=True)
    ax.hist(errors_anomaly, bins=80, alpha=0.6, label="Anomalie",color="tomato",    density=True)
    ax.axvline(threshold, color="black", linestyle="--", linewidth=2, label=f"Seuil ({threshold:.4f})")
    ax.set_xlabel("Erreur de reconstruction (MSE)"); ax.set_ylabel("Densité")
    ax.set_title(f"{model_name} — Distribution des erreurs", fontsize=12)
    ax.legend()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

def plot_roc_curves(results_dict, save_path=None):
    """results_dict = {'Model': (y_true, scores), ...}"""
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, (y_true, scores) in results_dict.items():
        fpr, tpr, _ = roc_curve(y_true, scores)
        auc = roc_auc_score(y_true, scores)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", linewidth=2)
    ax.plot([0,1],[0,1],"k--", linewidth=1)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title("Courbes ROC — Comparaison des modèles", fontsize=12)
    ax.legend()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

def plot_confusion_matrix(y_true, y_pred, model_name, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal","Anomalie"],
                yticklabels=["Normal","Anomalie"], ax=ax)
    ax.set_title(f"Matrice de confusion — {model_name}")
    ax.set_xlabel("Prédit"); ax.set_ylabel("Réel")
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

def evaluate_model(y_true, scores, threshold_percentile=95, model_name="Model"):
    """Calcule et affiche toutes les métriques."""
    # Binariser y_true pour détection d'anomalies (0=normal → 0, tout autre → 1)
    y_binary = (y_true != 0).astype(int)

    threshold = np.percentile(scores, threshold_percentile)
    y_pred = (scores > threshold).astype(int)

    auc_roc = roc_auc_score(y_binary, scores)
    auc_pr  = average_precision_score(y_binary, scores)

    print(f"\n{'='*50}")
    print(f"  {model_name}")
    print(f"{'='*50}")
    print(f"  AUC-ROC  : {auc_roc:.4f}")
    print(f"  AUC-PR   : {auc_pr:.4f}")
    print(f"  Seuil    : {threshold:.4f} (percentile {threshold_percentile})")
    print(classification_report(y_binary, y_pred,
                                 target_names=["Normal", "Anomalie"]))
    return {
        "model": model_name,
        "auc_roc": round(auc_roc, 4),
        "auc_pr":  round(auc_pr, 4),
        "threshold": round(float(threshold), 6)
    }

def save_results(results_list, path="outputs/results/metrics.json"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(results_list, f, indent=2)
    print(f"[✓] Métriques sauvegardées : {path}")