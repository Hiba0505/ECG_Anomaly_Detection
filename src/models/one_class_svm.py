"""
One-Class SVM : apprend une frontière autour des données normales.
Tout point hors frontière = anomalie.
"""
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import joblib, os

def build_ocsvm(kernel="rbf", nu=0.1, gamma="scale"):
    return OneClassSVM(kernel=kernel, nu=nu, gamma=gamma)

def fit_and_predict(model, X_train, X_test, scaler=None):
    if scaler is None:
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
    else:
        X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    model.fit(X_train_s)
    scores = -model.score_samples(X_test_s)
    preds  = model.predict(X_test_s)
    preds_binary = (preds == -1).astype(int)
    return scores, preds_binary, scaler