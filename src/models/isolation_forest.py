"""
Isolation Forest : méthode ensembliste non supervisée.
Isole les anomalies en construisant des arbres aléatoires.
Score d'anomalie = profondeur moyenne d'isolation.
"""
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib, os

def build_isolation_forest(n_estimators=200, contamination=0.15, random_state=42):
    return IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1
    )

def fit_and_predict(model, X_train, X_test):
    model.fit(X_train)
    scores = -model.score_samples(X_test)   # Plus haut = plus anormal
    preds  = model.predict(X_test)           # -1 = anomalie, 1 = normal
    preds_binary = (preds == -1).astype(int) # 0/1
    return scores, preds_binary

def save_model(model, path="outputs/models/isolation_forest.pkl"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)

def load_model(path):
    return joblib.load(path)