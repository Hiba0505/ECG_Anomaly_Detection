"""
Module d'authentification — gestion des comptes utilisateurs.
Stockage local dans un fichier JSON, mots de passe hashés (jamais en clair).
"""

import streamlit as st
import json
import os
import hashlib
import re
from datetime import datetime

USERS_FILE = "users_db.json"


def _hash_password(password: str) -> str:
    """Hash SHA-256 avec sel fixe simple — suffisant pour un usage local/académique."""
    salt = "ecg_anomaly_detection_2025"
    return hashlib.sha256((salt + password).encode()).hexdigest()


def _load_users() -> dict:
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_users(users: dict):
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2, ensure_ascii=False)


def _valid_email(email: str) -> bool:
    return re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email) is not None


def create_account(username: str, email: str, password: str, confirm: str):
    """Retourne (succès: bool, message: str)."""
    username = username.strip()
    email = email.strip().lower()

    if not username or not email or not password:
        return False, "Merci de remplir tous les champs."
    if len(username) < 3:
        return False, "Le nom d'utilisateur doit contenir au moins 3 caractères."
    if not _valid_email(email):
        return False, "Adresse email invalide."
    if len(password) < 6:
        return False, "Le mot de passe doit contenir au moins 6 caractères."
    if password != confirm:
        return False, "Les mots de passe ne correspondent pas."

    users = _load_users()
    if username in users:
        return False, "Ce nom d'utilisateur est déjà pris."
    if any(u["email"] == email for u in users.values()):
        return False, "Cette adresse email est déjà utilisée."

    users[username] = {
        "email": email,
        "password_hash": _hash_password(password),
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    _save_users(users)
    return True, "Compte créé avec succès. Tu peux maintenant te connecter."


def check_login(username: str, password: str):
    """Retourne (succès: bool, message: str)."""
    users = _load_users()
    username = username.strip()
    if username not in users:
        return False, "Nom d'utilisateur introuvable."
    if users[username]["password_hash"] != _hash_password(password):
        return False, "Mot de passe incorrect."
    return True, "Connexion réussie."


def require_login():
    """À appeler en haut de app.py. Affiche les écrans de connexion/inscription
    tant que l'utilisateur n'est pas authentifié, puis laisse passer."""

    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "username" not in st.session_state:
        st.session_state.username = None

    if st.session_state.authenticated:
        return  # déjà connecté → l'app continue normalement

    # ── Écran de connexion / inscription ──
    st.markdown(
        """
        <style>
        .auth-title {
            text-align:center; font-size:2.3rem; font-weight:800;
            background: linear-gradient(90deg,#2E6DB4,#0891B2);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            margin-bottom: 0.2rem;
        }
        .auth-subtitle { text-align:center; color:#64748B; margin-bottom:1.5rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    col_l, col_c, col_r = st.columns([1, 1.3, 1])
    with col_c:
        st.markdown('<div class="auth-title">🫀 ECG Anomaly Tool</div>', unsafe_allow_html=True)
        st.markdown('<div class="auth-subtitle">Outil de détection d\'anomalies ECG — Deep Learning</div>', unsafe_allow_html=True)

        tab_login, tab_signup = st.tabs(["🔑 Connexion", "📝 Créer un compte"])

        with tab_login:
            with st.form("login_form"):
                u = st.text_input("Nom d'utilisateur")
                p = st.text_input("Mot de passe", type="password")
                submitted = st.form_submit_button("Se connecter", type="primary", use_container_width=True)
            if submitted:
                ok, msg = check_login(u, p)
                if ok:
                    st.session_state.authenticated = True
                    st.session_state.username = u.strip()
                    st.rerun()
                else:
                    st.error(msg)

        with tab_signup:
            with st.form("signup_form"):
                su = st.text_input("Nom d'utilisateur", key="su_user")
                se = st.text_input("Adresse email", key="su_email")
                sp = st.text_input("Mot de passe", type="password", key="su_pass")
                sc = st.text_input("Confirmer le mot de passe", type="password", key="su_conf")
                signup_submitted = st.form_submit_button("Créer mon compte", type="primary", use_container_width=True)
            if signup_submitted:
                ok, msg = create_account(su, se, sp, sc)
                if ok:
                    st.success(msg + " Utilise l'onglet **Connexion** pour accéder à l'outil.")
                else:
                    st.error(msg)

    st.stop()  # bloque le reste de l'app tant que non authentifié
