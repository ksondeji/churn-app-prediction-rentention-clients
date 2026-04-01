"""Préparation des données Telco : nettoyage et construction du préprocesseur sklearn."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

DROP_COLS = ["customerID", "Churn"]


def load_clean_dataframe(csv_path: str) -> tuple[pd.DataFrame, pd.Series]:
    """Charge le CSV Telco, convertit TotalCharges, encode la cible binaire."""
    df = pd.read_csv(csv_path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)
    y = (df["Churn"].map({"Yes": 1, "No": 0})).astype(int)
    X = df.drop(columns=["Churn"])
    return X, y


def prepare_features(X: pd.DataFrame) -> pd.DataFrame:
    """Sélectionne les colonnes modèle (sans customerID si présent)."""
    cols = [c for c in NUMERIC_FEATURES + CATEGORICAL_FEATURES if c in X.columns]
    return X[cols].copy()


def build_preprocessor() -> ColumnTransformer:
    """ColumnTransformer : standardisation numérique + one-hot catégoriel."""
    return ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                NUMERIC_FEATURES,
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )
