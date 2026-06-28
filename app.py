"""
Interface Streamlit — Outil de Détection d'Anomalies ECG
Projet de Fin de Module — Deep Learning — FSBM

Outil pour UTILISER les modèles entraînés :
- diagnostiquer un signal ECG (saisie, sélection ou fichier)
- traiter un lot de battements (CSV)
- comparer les 4 modèles sur un même signal
- vérifier l'état des modèles/données

Lancement : streamlit run app.py
"""

import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import joblib
import keras

from auth import require_login

# ──────────────────────────────────────────────────────────────────
# CONFIGURATION DE PAGE
# ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Outil de Détection d'Anomalies ECG",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ──────────────────────────────────────────────────────────────────
# STYLE GLOBAL
# ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Fond général */
    .stApp { background-color: #F8FAFC; }

    /* Typographie titres */
    h1 { color:#0D1B3E; font-weight:800; }
    h2 { color:#1B3A6B; font-weight:700; }
    h3 { color:#1B3A6B; font-weight:600; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0D1B3E 0%, #1B3A6B 100%);
    }
    section[data-testid="stSidebar"] * { color: #E8F0FE !important; }
    section[data-testid="stSidebar"] .stRadio label { font-size: 0.95rem; }

    /* Cartes / containers */
    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 14px 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    div[data-testid="stMetric"] label { color:#64748B !important; }

    /* Boutons */
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        border: none;
        transition: all 0.15s ease;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #2E6DB4, #0891B2);
        color: white;
    }
    .stButton > button[kind="primary"]:hover {
        box-shadow: 0 4px 14px rgba(46,109,180,0.35);
        transform: translateY(-1px);
    }

    /* Tabs */
    .stTabs [data-baseweb="tab"] { font-weight:600; }

    /* DataFrames */
    div[data-testid="stDataFrame"] { border-radius: 10px; overflow:hidden; }

    /* Header bar custom */
    .topbar {
        background: linear-gradient(90deg, #0D1B3E, #1B3A6B 60%, #0891B2);
        padding: 1.1rem 1.6rem;
        border-radius: 14px;
        margin-bottom: 1.4rem;
        display:flex; justify-content:space-between; align-items:center;
        box-shadow: 0 4px 18px rgba(13,27,62,0.18);
    }
    .topbar h1 { color:white !important; margin:0; font-size:1.6rem; }
    .topbar .sub { color:#A8C4E8; font-size:0.85rem; margin-top:2px; }
    .userpill {
        background: rgba(255,255,255,0.12);
        color: white; padding: 6px 14px; border-radius: 20px;
        font-size: 0.85rem; font-weight:600;
    }

    /* Section card */
    .card {
        background: white; border:1px solid #E2E8F0; border-radius:14px;
        padding: 1.2rem 1.4rem; box-shadow: 0 1px 4px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────
# AUTHENTIFICATION — bloque tout tant que non connecté
# ──────────────────────────────────────────────────────────────────
require_login()

# ──────────────────────────────────────────────────────────────────
# CONSTANTES
# ──────────────────────────────────────────────────────────────────
MODEL_PATHS = {
    "LSTM Autoencoder (recommandé)": "outputs/models/lstm_autoencoder.keras",
    "Autoencoder Dense": "outputs/models/autoencoder_dense.keras",
    "Isolation Forest": "outputs/models/isolation_forest.pkl",
    "One-Class SVM": "outputs/models/ocsvm.pkl",
}
SIGNAL_LENGTH = 187
ACCENT = "#2E6DB4"
ACCENT2 = "#0891B2"


# ──────────────────────────────────────────────────────────────────
# CHARGEMENT DES MODÈLES (cache)
# ──────────────────────────────────────────────────────────────────
@st.cache_resource
def load_keras_model(path):
    return keras.models.load_model(path) if os.path.exists(path) else None

@st.cache_resource
def load_pickle_model(path):
    return joblib.load(path) if os.path.exists(path) else None

@st.cache_data
def load_npy(path):
    return np.load(path) if os.path.exists(path) else None

@st.cache_resource
def load_threshold(model_name):
    path = "outputs/results/mitbih_metrics.json"
    if not os.path.exists(path):
        return None
    with open(path) as f:
        metrics = json.load(f)
    clean_name = model_name.lower().replace(" ", "").replace("(recommandé)", "")
    for m in metrics:
        mname = m["model"].lower().replace(" ", "")
        if mname in clean_name or clean_name in mname:
            return m.get("threshold")
    return None


def score_dense(model, X):
    X_pred = model.predict(X, verbose=0)
    return np.mean(np.power(X - X_pred, 2), axis=1)

def score_lstm(model, X):
    X_3d = X.reshape(X.shape[0], X.shape[1], 1)
    X_pred = model.predict(X_3d, verbose=0)
    return np.mean(np.power(X_3d - X_pred, 2), axis=(1, 2))

def score_isolation_forest(model, X):
    return -model.score_samples(X)

def score_ocsvm(model_data, X):
    scaler = model_data["scaler"]
    model = model_data["model"]
    X_s = scaler.transform(X)
    return -model.score_samples(X_s)


def compute_scores(model_name, X):
    path = MODEL_PATHS[model_name]
    if not os.path.exists(path):
        return None, None
    if "LSTM" in model_name:
        model = load_keras_model(path); scores = score_lstm(model, X)
    elif "Dense" in model_name:
        model = load_keras_model(path); scores = score_dense(model, X)
    elif "Isolation" in model_name:
        model = load_pickle_model(path); scores = score_isolation_forest(model, X)
    elif "SVM" in model_name:
        model = load_pickle_model(path); scores = score_ocsvm(model, X)
    else:
        return None, None
    threshold = load_threshold(model_name)
    return scores, threshold


def validate_signal_matrix(X):
    msgs = []
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if X.shape[1] != SIGNAL_LENGTH:
        msgs.append(("error", f"Chaque signal doit contenir exactement {SIGNAL_LENGTH} points "
                               f"(trouvé : {X.shape[1]})."))
        return None, msgs
    if np.isnan(X).any():
        msgs.append(("error", "Le fichier contient des valeurs manquantes (NaN)."))
        return None, msgs
    if X.min() < -0.01 or X.max() > 1.01:
        msgs.append(("warning", "Les valeurs ne semblent pas normalisées entre 0 et 1 — "
                                 "les résultats du modèle peuvent être peu fiables."))
    return X.astype(np.float32), msgs


def styled_plot(signal, color=ACCENT, title=None):
    fig, ax = plt.subplots(figsize=(8, 3))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#FAFBFD")
    ax.plot(signal, color=color, linewidth=1.8)
    ax.fill_between(range(len(signal)), signal, alpha=0.08, color=color)
    ax.set_xlabel("Point temporel", color="#475569")
    ax.set_ylabel("Amplitude", color="#475569")
    ax.tick_params(colors="#94A3B8")
    for spine in ax.spines.values():
        spine.set_color("#E2E8F0")
    ax.grid(True, alpha=0.25)
    if title:
        ax.set_title(title, color="#1B3A6B", fontweight="bold", fontsize=11)
    return fig


# ──────────────────────────────────────────────────────────────────
# UTILITAIRES DE DÉFILEMENT
# ──────────────────────────────────────────────────────────────────
def scroll_to(anchor_id):
    components.html(f"""
    <script>
        window.parent.document.querySelector('#{anchor_id}').scrollIntoView(
            {{behavior: 'smooth', block: 'start'}}
        );
    </script>
    """, height=0)

# ──────────────────────────────────────────────────────────────────
# TOPBAR
# ──────────────────────────────────────────────────────────────────
LOGO_B64 = "PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyMjAgNjAiIHdpZHRoPSIyMjAiIGhlaWdodD0iNjAiPgogIDxkZWZzPgogICAgPGxpbmVhckdyYWRpZW50IGlkPSJiZ0dyYWQiIHgxPSIwJSIgeTE9IjAlIiB4Mj0iMTAwJSIgeTI9IjAlIj4KICAgICAgPHN0b3Agb2Zmc2V0PSIwJSIgc3R5bGU9InN0b3AtY29sb3I6IzBEMUIzRSIvPgogICAgICA8c3RvcCBvZmZzZXQ9IjEwMCUiIHN0eWxlPSJzdG9wLWNvbG9yOiMwODkxQjIiLz4KICAgIDwvbGluZWFyR3JhZGllbnQ+CiAgICA8bGluZWFyR3JhZGllbnQgaWQ9InB1bHNlR3JhZCIgeDE9IjAlIiB5MT0iMCUiIHgyPSIxMDAlIiB5Mj0iMCUiPgogICAgICA8c3RvcCBvZmZzZXQ9IjAlIiBzdHlsZT0ic3RvcC1jb2xvcjojNjBBNUZBIi8+CiAgICAgIDxzdG9wIG9mZnNldD0iMTAwJSIgc3R5bGU9InN0b3AtY29sb3I6IzIyRDNFRSIvPgogICAgPC9saW5lYXJHcmFkaWVudD4KICAgIDxmaWx0ZXIgaWQ9Imdsb3ciPgogICAgICA8ZmVHYXVzc2lhbkJsdXIgc3RkRGV2aWF0aW9uPSIxLjUiIHJlc3VsdD0iYmx1ciIvPgogICAgICA8ZmVNZXJnZT48ZmVNZXJnZU5vZGUgaW49ImJsdXIiLz48ZmVNZXJnZU5vZGUgaW49IlNvdXJjZUdyYXBoaWMiLz48L2ZlTWVyZ2U+CiAgICA8L2ZpbHRlcj4KICA8L2RlZnM+CgogIDwhLS0gQmFja2dyb3VuZCBwaWxsIC0tPgogIDxyZWN0IHg9IjAiIHk9IjAiIHdpZHRoPSIyMjAiIGhlaWdodD0iNjAiIHJ4PSIxNCIgZmlsbD0idXJsKCNiZ0dyYWQpIi8+CgogIDwhLS0gSGVhcnQgaWNvbiBjaXJjbGUgLS0+CiAgPGNpcmNsZSBjeD0iMzIiIGN5PSIzMCIgcj0iMjAiIGZpbGw9InJnYmEoMjU1LDI1NSwyNTUsMC4wOCkiLz4KCiAgPCEtLSBIZWFydCBzaGFwZSAtLT4KICA8cGF0aCBkPSJNMzIgNDAgQzMyIDQwIDE4IDMxIDE4IDIzIEMxOCAxOC41IDIxLjUgMTUgMjUuNSAxNSBDMjggMTUgMzAuNSAxNi41IDMyIDE4LjUgQzMzLjUgMTYuNSAzNiAxNSAzOC41IDE1IEM0Mi41IDE1IDQ2IDE4LjUgNDYgMjMgQzQ2IDMxIDMyIDQwIDMyIDQwWiIKICAgICAgICBmaWxsPSJub25lIiBzdHJva2U9IiNFRjQ0NDQiIHN0cm9rZS13aWR0aD0iMS44IiBzdHJva2UtbGluZWpvaW49InJvdW5kIi8+CiAgPCEtLSBIZWFydCBmaWxsIHN1YnRsZSAtLT4KICA8cGF0aCBkPSJNMzIgMzggQzMyIDM4IDIwIDMwIDIwIDIzIEMyMCAxOS41IDIyLjUgMTcgMjUuNSAxNyBDMjggMTcgMzAgMTguNSAzMiAyMC41IEMzNCAxOC41IDM2IDE3IDM4LjUgMTcgQzQxLjUgMTcgNDQgMTkuNSA0NCAyMyBDNDQgMzAgMzIgMzggMzIgMzhaIgogICAgICAgIGZpbGw9InJnYmEoMjM5LDY4LDY4LDAuMTUpIi8+CgogIDwhLS0gRUNHIGxpbmUgdGhyb3VnaCBoZWFydCAtLT4KICA8cG9seWxpbmUgcG9pbnRzPSIxOCwyOSAyMiwyOSAyNCwyOSAyNS41LDIzIDI3LDM1IDI4LjUsMjAgMzAsMzQgMzEuNSwyOSAzMywyOSAzNSwyOSAzNywyNCAzOC41LDMyIDQwLDI5IDQ2LDI5IgogICAgICAgICAgICBmaWxsPSJub25lIiBzdHJva2U9InVybCgjcHVsc2VHcmFkKSIgc3Ryb2tlLXdpZHRoPSIxLjgiCiAgICAgICAgICAgIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgc3Ryb2tlLWxpbmVqb2luPSJyb3VuZCIKICAgICAgICAgICAgZmlsdGVyPSJ1cmwoI2dsb3cpIi8+CgogIDwhLS0gQW5vbWFseSBkb3QgaGlnaGxpZ2h0IC0tPgogIDxjaXJjbGUgY3g9IjI4LjUiIGN5PSIyMCIgcj0iMi41IiBmaWxsPSIjMjJEM0VFIiBvcGFjaXR5PSIwLjkiIGZpbHRlcj0idXJsKCNnbG93KSIvPgoKICA8IS0tIEFwcCBuYW1lIC0tPgogIDx0ZXh0IHg9IjYyIiB5PSIyNCIgZm9udC1mYW1pbHk9IidTZWdvZSBVSScsIEFyaWFsLCBzYW5zLXNlcmlmIiBmb250LXNpemU9IjEzLjUiIGZvbnQtd2VpZ2h0PSI4MDAiIGZpbGw9IndoaXRlIiBsZXR0ZXItc3BhY2luZz0iMC4zIj5DYXJkaW9TY2FuPC90ZXh0PgogIDx0ZXh0IHg9IjYyIiB5PSIzOCIgZm9udC1mYW1pbHk9IidTZWdvZSBVSScsIEFyaWFsLCBzYW5zLXNlcmlmIiBmb250LXNpemU9IjguNSIgZm9udC13ZWlnaHQ9IjQwMCIgZmlsbD0iI0E4QzRFOCIgbGV0dGVyLXNwYWNpbmc9IjEuMiI+RMOJVEVDVElPTiBEJ0FOT01BTElFUyBFQ0c8L3RleHQ+CgogIDwhLS0gU2VwYXJhdG9yIGxpbmUgLS0+CiAgPGxpbmUgeDE9IjE1NiIgeTE9IjE0IiB4Mj0iMTU2IiB5Mj0iNDYiIHN0cm9rZT0icmdiYSgyNTUsMjU1LDI1NSwwLjE1KSIgc3Ryb2tlLXdpZHRoPSIxIi8+CgogIDwhLS0gQmFkZ2UgRlNCTSAtLT4KICA8dGV4dCB4PSIxNjUiIHk9IjI3IiBmb250LWZhbWlseT0iJ1NlZ29lIFVJJywgQXJpYWwsIHNhbnMtc2VyaWYiIGZvbnQtc2l6ZT0iOCIgZm9udC13ZWlnaHQ9IjcwMCIgZmlsbD0iIzYwQTVGQSIgbGV0dGVyLXNwYWNpbmc9IjAuNSI+RlNCTTwvdGV4dD4KICA8dGV4dCB4PSIxNjMiIHk9IjM5IiBmb250LWZhbWlseT0iJ1NlZ29lIFVJJywgQXJpYWwsIHNhbnMtc2VyaWYiIGZvbnQtc2l6ZT0iNyIgZmlsbD0iIzdCQTRDQyIgbGV0dGVyLXNwYWNpbmc9IjAuMyI+SGFzc2FuIElJPC90ZXh0PgogIDx0ZXh0IHg9IjE2MSIgeT0iNDkiIGZvbnQtZmFtaWx5PSInU2Vnb2UgVUknLCBBcmlhbCwgc2Fucy1zZXJpZiIgZm9udC1zaXplPSI2LjUiIGZpbGw9IiM1QThBQUYiPkNhc2FibGFuY2E8L3RleHQ+Cjwvc3ZnPgo="

st.markdown(f"""
<div class="topbar">
    <div style="display:flex; align-items:center;">
        <img src="data:image/svg+xml;base64,{LOGO_B64}" height="52" style="filter: drop-shadow(0 2px 6px rgba(0,0,0,0.3));" alt="CardioScan Logo"/>
    </div>
    <div class="userpill">👤 {st.session_state.username}</div>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🫀 Menu")
    page = st.radio(
        "Fonctionnalité",
        ["🔍 Diagnostiquer un signal", "📁 Traiter un lot (CSV)"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.caption("Chaque signal doit contenir exactement **187 points**, normalisés entre 0 et 1.")
    st.markdown("---")
    st.markdown("""
    <style>
    [data-testid="stSidebar"] button,
    [data-testid="stSidebar"] button:focus,
    [data-testid="stSidebar"] button:active,
    [data-testid="stSidebar"] .stButton button,
    [data-testid="stSidebar"] div.stButton > button,
    section[data-testid="stSidebar"] div.stButton > button {
        background-color: #DC2626 !important;
        background: #DC2626 !important;
        color: white !important;
        border: 2px solid rgba(255,255,255,0.3) !important;
        font-weight: 700 !important;
    }
    [data-testid="stSidebar"] button:hover,
    [data-testid="stSidebar"] div.stButton > button:hover,
    section[data-testid="stSidebar"] div.stButton > button:hover {
        background-color: #B91C1C !important;
        background: #B91C1C !important;
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)
    if st.button("🚪 Se déconnecter", use_container_width=True):
        st.session_state.authenticated = False
        st.session_state.username = None
        st.rerun()


# ════════════════════════════════════════════════════════════════
# PAGE 1 — DIAGNOSTIQUER UN SIGNAL
# ════════════════════════════════════════════════════════════════
if page == "🔍 Diagnostiquer un signal":
    st.subheader("🔍 Diagnostiquer un signal ECG")
    st.caption("Fournis un battement ECG (187 points) et obtiens un diagnostic immédiat.")

    input_mode = st.radio(
        "Source du signal :",
        ["Saisir / coller les 187 valeurs", "Choisir un battement du jeu de test",
         "Importer un fichier (.csv ou .npy, 1 ligne = 1 signal)"],
        horizontal=True
    )

    signal = None

    if input_mode == "Saisir / coller les 187 valeurs":
        raw = st.text_area(
            "Colle 187 valeurs séparées par des virgules ou des espaces :",
            height=110,
            placeholder="0.12, 0.34, 0.51, 0.88, ... (187 valeurs au total)"
        )
        if raw.strip():
            try:
                values = [float(v) for v in raw.replace(",", " ").split()]
                if len(values) != SIGNAL_LENGTH:
                    st.error(f"❌ {len(values)} valeurs détectées — il en faut exactement {SIGNAL_LENGTH}.")
                else:
                    signal = np.array(values, dtype=np.float32)
            except ValueError:
                st.error("❌ Le texte contient des caractères non numériques.")

    elif input_mode == "Choisir un battement du jeu de test":
        X_test = load_npy("data/processed/X_test.npy")
        if X_test is None:
            st.warning("⚠️ `data/processed/X_test.npy` introuvable. Utilise la saisie manuelle ou un fichier importé.")
        else:
            idx = st.number_input(f"Index du signal (0 à {len(X_test)-1})",
                                    min_value=0, max_value=len(X_test)-1, value=0, step=1)
            signal = X_test[idx]
            st.caption("Ce mode sert à tester rapidement l'outil avec des signaux déjà connus.")

    else:
        uploaded = st.file_uploader("Fichier .csv ou .npy contenant 1 signal (187 valeurs, 1 ligne)",
                                      type=["csv", "npy"])
        if uploaded is not None:
            try:
                arr = np.load(uploaded) if uploaded.name.endswith(".npy") else pd.read_csv(uploaded, header=None).values
                arr = np.array(arr).flatten()
                if len(arr) != SIGNAL_LENGTH:
                    st.error(f"❌ {len(arr)} valeurs détectées — il en faut exactement {SIGNAL_LENGTH}.")
                else:
                    signal = arr.astype(np.float32)
            except Exception as e:
                st.error(f"❌ Erreur de lecture du fichier : {e}")

    if signal is not None:
        st.markdown('<div id="signal-section"></div>', unsafe_allow_html=True)
        scroll_to("signal-section")
        st.markdown("---")
        col_sig, col_diag = st.columns([1.4, 1])

        with col_sig:
            st.markdown("##### 📈 Signal fourni")
            st.pyplot(styled_plot(signal))

        with col_diag:
            st.markdown("##### 🩺 Diagnostic")
            run = st.button("🔬 Lancer le diagnostic", type="primary", use_container_width=True)

        MODEL_LSTM = "LSTM Autoencoder (recommandé)"

        if run:
            scores, threshold = compute_scores(MODEL_LSTM, signal.reshape(1, -1))
            if scores is None:
                st.error("Le modèle d'analyse est introuvable. Veuillez contacter l'administrateur.")
            else:
                score = scores[0]
                st.markdown("---")
                st.markdown('<div id="result-section"></div>', unsafe_allow_html=True)
                scroll_to("result-section")
                st.markdown("##### 🩺 Résultat du diagnostic")
                if threshold is not None:
                    is_anomaly = score > threshold
                    if is_anomaly:
                        st.error("🔴 ANOMALIE détectée — Ce signal présente des irrégularités.")
                    else:
                        st.success("✅ Signal NORMAL — Aucune anomalie détectée.")
                else:
                    st.info("Analyse effectuée. Veuillez contacter l'administrateur pour obtenir le seuil de décision.")


# ════════════════════════════════════════════════════════════════
# PAGE 2 — TRAITER UN LOT (CSV)
# ════════════════════════════════════════════════════════════════
elif page == "📁 Traiter un lot (CSV)":
    st.subheader("📁 Traiter un lot de signaux ECG")
    st.caption("Importe un fichier CSV (1 ligne = 1 signal de 187 valeurs) pour un diagnostic groupé.")

    MODEL_LSTM = "LSTM Autoencoder (recommandé)"
    uploaded = st.file_uploader("Fichier CSV (sans en-tête, 187 colonnes par ligne)", type=["csv"])

    if uploaded is not None:
        try:
            X = pd.read_csv(uploaded, header=None).values
            X_valid, msgs = validate_signal_matrix(X)
            for level, m in msgs:
                st.error(m) if level == "error" else st.warning(m)

            if X_valid is not None:
                st.success(f"✅ {X_valid.shape[0]} signaux chargés avec succès.")

                if st.button("🔬 Lancer le diagnostic du lot", type="primary"):
                    with st.spinner(f"Analyse de {X_valid.shape[0]} signaux en cours..."):
                        scores, threshold = compute_scores(MODEL_LSTM, X_valid)

                    if scores is None:
                        st.error("Le modèle d'analyse est introuvable. Veuillez contacter l'administrateur.")
                    else:
                        if threshold is not None:
                            diagnostics = np.where(scores > threshold, "🔴 Anomalie", "✅ Normal")
                            n_anom = int((scores > threshold).sum())
                            n_normal = len(scores) - n_anom

                            c1, c2, c3 = st.columns(3)
                            c1.metric("Total analysé", len(scores))
                            c2.metric("Normaux", n_normal)
                            c3.metric("Anomalies détectées", n_anom)

                            result_df = pd.DataFrame({
                                "Signal n°": range(1, len(scores) + 1),
                                "Diagnostic": diagnostics
                            })
                        else:
                            result_df = pd.DataFrame({
                                "Signal n°": range(1, len(scores) + 1),
                                "Diagnostic": ["Analyse effectuée"] * len(scores)
                            })
                            st.info("Seuil de décision indisponible. Contactez l'administrateur.")

                        st.markdown("---")
                        st.markdown("##### 📋 Résultats")
                        st.dataframe(result_df, use_container_width=True, hide_index=True)

                        st.download_button("⬇️ Télécharger les résultats (CSV)",
                            data=result_df.to_csv(index=False).encode("utf-8"),
                            file_name="diagnostic_resultats.csv", mime="text/csv")
        except Exception as e:
            st.error(f"❌ Erreur lors de la lecture du fichier : {e}")
    else:
        st.info("⬆️ Importe un fichier CSV pour commencer.")
