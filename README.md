# Détection d'Anomalies ECG par Approches Non Supervisées

> Projet de Fin de Module — Deep Learning  
> FSBM — Université Hassan II de Casablanca — 2024/2025

---

## Description

Ce projet implémente et compare **4 modèles non supervisés** pour détecter automatiquement des anomalies dans des signaux ECG (électrocardiogrammes), sans utiliser de données annotées lors de l'entraînement.

**Principe** : les modèles apprennent uniquement la structure des battements normaux. Tout signal s'écartant significativement de cette représentation est détecté comme anomalie.

---

## Structure du projet

```
ECG_Anomaly_Detection/
├── data/
│   ├── raw/                  ← Placer ici les 4 fichiers CSV (Kaggle)
│   └── processed/            ← Généré automatiquement (.npy)
├── notebooks/
│   ├── 01_EDA.ipynb          ← Exploration des données
│   └── 02_results_visualization.ipynb
├── src/
│   ├── data_preprocessing.py ← Étape 1
│   ├── train.py              ← Étape 2
│   ├── evaluate.py           ← Étape 3
│   ├── utils.py
│   └── models/
│       ├── autoencoder.py
│       ├── lstm_autoencoder.py
│       ├── isolation_forest.py
│       └── one_class_svm.py
├── outputs/
│   ├── models/               ← Poids sauvegardés
│   ├── figures/              ← Graphiques générés
│   └── results/              ← Métriques JSON
├── config.yaml
├── requirements.txt
└── README.md
```

---

## Données

Télécharger depuis Kaggle : [ECG Heartbeat Categorization](https://www.kaggle.com/datasets/shayanfazeli/heartbeat)

Placer les 4 fichiers dans `data/raw/` :
- `mitbih_train.csv` — 87 554 battements (5 classes)
- `mitbih_test.csv` — 21 892 battements
- `ptbdb_normal.csv` — 4 046 battements normaux
- `ptbdb_abnormal.csv` — 10 506 battements anormaux

---

**requirements.txt**
```
numpy
pandas
scikit-learn
tensorflow
keras
matplotlib
seaborn
plotly
pyyaml
joblib
tqdm
jupyter
ipykernel
```

---

## Ordre d'exécution

```bash
# Étape 1 — Prétraitement des données
python src/data_preprocessing.py

# Étape 2 — Entraînement des 4 modèles
python src/train.py

# Étape 3 — Évaluation et génération des résultats
python src/evaluate.py
```

---

## Modèles implémentés

| Modèle | Type | Principe |
|--------|------|----------|
| **Autoencoder Dense** | Deep Learning | Erreur de reconstruction MSE |
| **LSTM Autoencoder** | Deep Learning | Dépendances temporelles du signal |
| **Isolation Forest** | Machine Learning | Score d'isolation aléatoire |
| **One-Class SVM** | Machine Learning | Frontière de normalité (kernel RBF) |

---

## Résultats

### MIT-BIH Arrhythmia Database

| Modèle | AUC-ROC | AUC-PR | Accuracy |
|--------|---------|--------|----------|
| **LSTM Autoencoder** | **0.9504** | **0.8270** | **0.87** |
| Autoencoder Dense | 0.8804 | 0.6366 | 0.86 |
| Isolation Forest | 0.7532 | 0.3381 | 0.80 |
| One-Class SVM | 0.7049 | 0.3839 | 0.82 |

### PTBDB

| Modèle | AUC-ROC | AUC-PR |
|--------|---------|--------|
| **LSTM Autoencoder** | **0.6872** | **0.8605** |
| One-Class SVM | 0.6110 | 0.7994 |
| Autoencoder Dense | 0.5903 | 0.8165 |
| Isolation Forest | 0.5402 | 0.7794 |

> Le **LSTM Autoencoder** est le meilleur modèle sur les deux datasets.

---

## Environnement

- Python 3.12
- TensorFlow 2.21 / Keras 3.14
- scikit-learn
- VS Code

---
## 🖥️ Interface Streamlit

# Interface Streamlit — Outil de Détection d'Anomalies ECG

Outil interactif pour **utiliser** les modèles entraînés : diagnostiquer un signal ECG,
traiter un lot de battements, ou comparer les 4 modèles entre eux.

## 📁 Où placer ce fichier

Place `app.py` et `requirements.txt` **à la racine** de ton projet `ECG_Anomaly_Detection`,
à côté des dossiers `data/`, `src/`, `outputs/`.

```
ECG_Anomaly_Detection/
├── app.py                  
├── requirements.txt        
├── data/
├── src/
├── outputs/
└── ...
```

## ⚙️ Installation

```bash
pip install streamlit
```

(les autres dépendances — numpy, pandas, tensorflow, etc. — sont déjà installées
si tu as suivi le reste du projet)

## 🚀 Lancement

Depuis la racine du projet, dans le terminal :

```bash
streamlit run app.py
```

Une page va s'ouvrir automatiquement dans ton navigateur à l'adresse
`http://localhost:8501`

## 🧭 Pages de l'interface

1. **🔍 Diagnostiquer un signal** — fournis un battement ECG (saisie manuelle, sélection
   dans le jeu de test, ou fichier importé) et obtiens un diagnostic immédiat avec un modèle au choix
2. **📁 Traiter un lot (CSV)** — importe un fichier CSV contenant plusieurs battements
   et diagnostique-les tous en une fois ; export des résultats en CSV
3. **⚖️ Comparer les modèles** — fait tourner les 4 modèles sur le même signal pour
   voir s'ils sont d'accord
4. **🛠️ État des modèles** — vérifie que tous les fichiers nécessaires (modèles
   entraînés, données, seuils) sont bien présents

## ⚠️ Prérequis

Pour que toutes les pages fonctionnent, il faut avoir déjà exécuté dans l'ordre :

```bash
python src/data_preprocessing.py
python src/train.py
python src/evaluate.py
```

Si un fichier est manquant, l'interface affichera un message d'avertissement clair
indiquant quelle étape exécuter.

## 🎨 Personnalisation

- Les couleurs des graphiques peuvent être modifiées dans les appels `plt.subplots()`
- Le nom du fichier de métriques utilisé pour les seuils (`outputs/results/mitbih_metrics.json`)
  peut être changé dans la fonction `load_threshold()` si besoin

## 🔐 Comptes utilisateurs

L'application démarre désormais sur un écran de **connexion / création de compte**.

- Crée un compte via l'onglet **"Créer un compte"** (nom d'utilisateur, email, mot de passe)
- Connecte-toi ensuite via l'onglet **"Connexion"**
- Un bouton **"Se déconnecter"** est disponible en bas de la barre latérale

Les comptes sont stockés localement dans `users_db.json` (créé automatiquement
au premier lancement, à la racine du projet). Les mots de passe sont hashés
(SHA-256) — jamais stockés en clair.

⚠️ Ce système est pensé pour un usage local/académique (démonstration du projet),
pas pour une mise en production avec des données sensibles.

## Auteur

Groupe :**Hiba Hanane et Ilham Elbiti** 
Encadrant : **Oumaima Guendoul**
FSBM — Université Hassan II de Casablanca