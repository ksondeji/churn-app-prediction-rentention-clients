"""Dashboard Streamlit : batch CSV, scores de risque, SHAP, segmentation."""

from __future__ import annotations

import os
import tempfile
import urllib.request
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

from ml.preprocessing import prepare_features

try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    plt.style.use("ggplot")

ROOT = Path(__file__).resolve().parent
DEFAULT_BUNDLE = ROOT / "artifacts" / "churn_bundle.joblib"
LEGACY_BUNDLE = ROOT / "churn_model.pkl"
MODEL_ENV = os.environ.get("CHURN_MODEL_PATH")


def _bundle_from_path(p: Path):
    if not p.is_file():
        return None
    b = joblib.load(p)
    if isinstance(b, dict) and "pipeline" in b:
        return b
    if hasattr(b, "predict_proba"):
        return {"pipeline": b, "threshold": 0.5}
    return None


def _remote_model_url() -> str | None:
    for key in ("CHURN_MODEL_URL", "STREAMLIT_CHURN_MODEL_URL"):
        v = os.environ.get(key)
        if v and v.strip():
            return v.strip()
    try:
        if "CHURN_MODEL_URL" in st.secrets:
            return str(st.secrets["CHURN_MODEL_URL"]).strip()
    except (FileNotFoundError, RuntimeError, KeyError, TypeError):
        pass
    return None


@st.cache_resource
def load_bundle():
    paths = []
    if MODEL_ENV:
        paths.append(Path(MODEL_ENV))
    paths.extend([DEFAULT_BUNDLE, LEGACY_BUNDLE])
    for p in paths:
        b = _bundle_from_path(p)
        if b is not None:
            return b

    url = _remote_model_url()
    if url and (url.startswith("https://") or url.startswith("http://")):
        cache_path = Path(tempfile.gettempdir()) / "streamlit_churn_bundle.joblib"
        urllib.request.urlretrieve(url, cache_path)
        return _bundle_from_path(cache_path)

    return None


def segment(proba: float) -> str:
    if proba >= 0.7:
        return "Critique"
    if proba >= 0.4:
        return "Élevé"
    if proba >= 0.2:
        return "Modéré"
    return "Faible"


def predict_batch(bundle: dict, X: pd.DataFrame) -> pd.DataFrame:
    pipe = bundle["pipeline"]
    thr = float(bundle.get("threshold", 0.5))
    Xf = prepare_features(X)
    proba = pipe.predict_proba(Xf)[:, 1]
    out = X.copy()
    out["churn_probability"] = proba
    out["churn_predicted"] = (proba >= thr).astype(int)
    out["segment"] = [segment(float(p)) for p in proba]
    return out


st.set_page_config(page_title="Churn Telco — Dashboard", layout="wide", page_icon="📡")

st.title("Prédiction de churn — télécoms")
st.caption("Scores de risque, SHAP et segmentation pour actions de rétention (horizon cible ~3 mois via signaux comportementaux).")

bundle = load_bundle()
if bundle is None:
    st.error(
        "Aucun bundle modèle trouvé. **En local :** exécutez `python -m ml.train_pipeline` puis vérifiez "
        "`artifacts/churn_bundle.joblib`. **Sur Streamlit Cloud :** commitez ce fichier dans le dépôt "
        "(il est autorisé dans `.gitignore`) ou définissez le secret **`CHURN_MODEL_URL`** (lien HTTPS "
        "vers le `.joblib`, ex. release GitHub / stockage objet)."
    )
    st.stop()

tab1, tab2, tab3 = st.tabs(["Données & prédictions", "SHAP & drivers", "Segmentation"])

with tab1:
    st.subheader("Jeu de données ou CSV batch")
    default_csv = ROOT / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    uploaded = st.file_uploader("Uploader un CSV (format Telco)", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
    elif default_csv.is_file():
        df = pd.read_csv(default_csv)
        st.info("Fichier local Telco chargé par défaut.")
    else:
        st.warning("Chargez un CSV ou placez le fichier Telco à la racine du projet.")
        st.stop()

    if "Churn" in df.columns:
        X_raw = df.drop(columns=["Churn"])
        y_true = df["Churn"]
    else:
        X_raw = df.copy()
        y_true = None

    if "TotalCharges" in X_raw.columns:
        X_raw["TotalCharges"] = pd.to_numeric(X_raw["TotalCharges"], errors="coerce").fillna(0.0)

    with st.spinner("Prédictions…"):
        scored = predict_batch(bundle, X_raw)

    st.dataframe(
        scored.head(50),
        use_container_width=True,
    )
    st.download_button(
        "Télécharger résultats (CSV)",
        scored.to_csv(index=False).encode("utf-8"),
        file_name="predictions_churn.csv",
        mime="text/csv",
    )

    col_a, col_b = st.columns(2)
    with col_a:
        fig1, ax1 = plt.subplots(figsize=(6, 3))
        ax1.hist(scored["churn_probability"], bins=30, color="steelblue", edgecolor="white")
        ax1.set_xlabel("Probabilité churn")
        ax1.set_title("Distribution des scores")
        st.pyplot(fig1)
        plt.close(fig1)
    with col_b:
        seg_counts = scored["segment"].value_counts()
        fig2, ax2 = plt.subplots(figsize=(6, 3))
        seg_counts.reindex(["Faible", "Modéré", "Élevé", "Critique"]).fillna(0).plot(
            kind="bar", ax=ax2, color=["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c"]
        )
        ax2.set_title("Effectifs par segment")
        plt.xticks(rotation=25)
        st.pyplot(fig2)
        plt.close(fig2)

    if y_true is not None:
        from sklearn.metrics import roc_auc_score

        y_bin = y_true.map({"Yes": 1, "No": 0})
        mask = y_bin.notna()
        if mask.sum() > 0:
            auc = roc_auc_score(y_bin[mask], scored.loc[mask, "churn_probability"])
            st.metric("ROC-AUC (si Churn présent)", f"{auc:.3f}")

with tab2:
    st.subheader("Importance locale (SHAP — sous-échantillon)")
    pipe = bundle["pipeline"]
    prep = pipe.named_steps["prep"]
    clf = pipe.named_steps["clf"]
    X_demo = prepare_features(X_raw)
    max_rows = min(400, len(X_demo))
    X_sub = X_demo.sample(n=max_rows, random_state=42) if len(X_demo) > max_rows else X_demo
    X_trans = prep.transform(X_sub)
    try:
        explainer = shap.TreeExplainer(clf)
        sv = explainer.shap_values(X_trans)
        if isinstance(sv, list):
            sv = sv[1]
        feat_names = prep.get_feature_names_out()
        mean_abs = np.abs(sv).mean(axis=0)
        top_idx = np.argsort(mean_abs)[::-1][:15]
        fig3, ax3 = plt.subplots(figsize=(8, 5))
        ax3.barh(feat_names[top_idx][::-1], mean_abs[top_idx][::-1], color="teal")
        ax3.set_title("Top features |SHAP| moyen")
        st.pyplot(fig3)
        plt.close(fig3)
    except Exception as e:
        st.warning(f"SHAP indisponible sur ce jeu : {e}")

with tab3:
    st.subheader("Clients à risque prioritaires")
    thr_show = st.slider("Seuil d’affichage probabilité", 0.0, 1.0, 0.35, 0.05)
    at_risk = scored[scored["churn_probability"] >= thr_show].sort_values(
        "churn_probability", ascending=False
    )
    st.write(f"**{len(at_risk)}** clients ≥ {thr_show:.0%}")
    show_cols = [c for c in ["customerID", "tenure", "Contract", "MonthlyCharges", "churn_probability", "segment"] if c in at_risk.columns]
    st.dataframe(at_risk[show_cols].head(100), use_container_width=True)
