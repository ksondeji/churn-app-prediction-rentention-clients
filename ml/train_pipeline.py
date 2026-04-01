"""
Entraînement pipeline XGBoost + MLflow.
Split stratifié 80/20, GridSearchCV stratifié, objectif recall classe churn.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from ml.preprocessing import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    build_preprocessor,
    load_clean_dataframe,
    prepare_features,
)

RANDOM_STATE = 42
ARTIFACT_DIR = Path(__file__).resolve().parent.parent / "artifacts"
DATA_DEFAULT = Path(__file__).resolve().parent.parent / "WA_Fn-UseC_-Telco-Customer-Churn.csv"


def optimal_threshold_recall(y_true: np.ndarray, proba: np.ndarray, min_recall: float = 0.8) -> float:
    """Plus haut seuil avec recall >= min_recall (meilleure précision possible sous contrainte)."""
    grid = np.linspace(0.001, 0.999, 500)
    valid = []
    for t in grid:
        pred = (proba >= t).astype(int)
        rec = recall_score(y_true, pred, zero_division=0)
        if rec >= min_recall:
            valid.append(float(t))
    if valid:
        return max(valid)
    # Contrainte impossible sur cet échantillon : seuil bas pour maximiser le recall
    best_rec = -1.0
    best_t = 0.35
    for t in grid:
        pred = (proba >= t).astype(int)
        rec = recall_score(y_true, pred, zero_division=0)
        if rec > best_rec:
            best_rec = rec
            best_t = float(t)
    return best_t


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=str(DATA_DEFAULT))
    parser.add_argument("--experiment", type=str, default="telco_churn")
    parser.add_argument("--register", type=str, default="", help="Nom du modèle MLflow Model Registry")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Grille réduite pour itérer plus vite (désactiver pour recherche complète).",
    )
    args = parser.parse_args()

    csv_path = Path(args.data)
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)

    X_raw, y = load_clean_dataframe(str(csv_path))
    X = prepare_features(X_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    X_tr_inner, X_val, y_tr_inner, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.15,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )

    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    spw_default = max(n_neg / max(n_pos, 1), 1.0)

    preprocessor = build_preprocessor()
    xgb = XGBClassifier(
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        n_jobs=-1,
    )
    pipe = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("clf", xgb),
        ]
    )

    if args.quick:
        param_grid = {
            "clf__n_estimators": [200],
            "clf__max_depth": [5, 6],
            "clf__learning_rate": [0.1],
            "clf__subsample": [0.9],
            "clf__colsample_bytree": [0.9],
            "clf__scale_pos_weight": [spw_default, spw_default * 1.5, spw_default * 2.0],
        }
    else:
        param_grid = {
            "clf__n_estimators": [150, 300],
            "clf__max_depth": [4, 6],
            "clf__learning_rate": [0.05, 0.1],
            "clf__subsample": [0.8, 1.0],
            "clf__colsample_bytree": [0.8, 1.0],
            "clf__scale_pos_weight": [1.0, spw_default, spw_default * 1.2],
        }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    grid = GridSearchCV(
        pipe,
        param_grid,
        scoring="recall",
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=1,
    )

    mlflow.set_experiment(args.experiment)
    with mlflow.start_run(run_name="grid_xgb_recall") as run:
        grid.fit(X_tr_inner, y_tr_inner)
        best_inner: Pipeline = grid.best_estimator_

        proba_val = best_inner.predict_proba(X_val)[:, 1]
        threshold = optimal_threshold_recall(y_val.values, proba_val, min_recall=0.8)

        best: Pipeline = clone(best_inner)
        best.fit(X_train, y_train)

        mlflow.log_params(grid.best_params_)
        mlflow.log_param("train_rows", len(X_train))
        mlflow.log_param("test_rows", len(X_test))
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_metric("val_recall_at_threshold", float(recall_score(y_val, (proba_val >= threshold).astype(int))))

        proba_test = best.predict_proba(X_test)[:, 1]
        y_pred_default = best.predict(X_test)
        y_pred_tuned = (proba_test >= threshold).astype(int)

        metrics = {
            "cv_best_recall": float(grid.best_score_),
            "test_recall_default": float(recall_score(y_test, y_pred_default)),
            "test_recall_tuned": float(recall_score(y_test, y_pred_tuned)),
            "test_precision_tuned": float(precision_score(y_test, y_pred_tuned, zero_division=0)),
            "test_f1_tuned": float(f1_score(y_test, y_pred_tuned, zero_division=0)),
            "test_roc_auc": float(roc_auc_score(y_test, proba_test)),
            "threshold": float(threshold),
        }
        mlflow.log_metrics(metrics)

        print("Best params:", grid.best_params_)
        print("Threshold (validation interne, recall>=80% si possible):", threshold)
        print("Test metrics:", metrics)
        print(classification_report(y_test, y_pred_tuned, target_names=["No churn", "Churn"]))
        print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_tuned))

        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        bundle = {
            "pipeline": best,
            "threshold": threshold,
            "numeric_features": NUMERIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
        }
        bundle_path = ARTIFACT_DIR / "churn_bundle.joblib"
        joblib.dump(bundle, bundle_path)
        legacy_path = ARTIFACT_DIR.parent / "churn_model.pkl"
        joblib.dump(bundle, legacy_path)

        def _json_val(v):
            if isinstance(v, (np.integer, int)):
                return int(v)
            if isinstance(v, (np.floating, float)):
                return float(v)
            return v

        meta = {
            "metrics": metrics,
            "best_params": {k: _json_val(v) for k, v in grid.best_params_.items()},
            "mlflow_run_id": run.info.run_id,
        }
        with open(ARTIFACT_DIR / "champion_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        mlflow.log_artifact(str(bundle_path))
        mlflow.log_artifact(str(ARTIFACT_DIR / "champion_meta.json"))
        mlflow.sklearn.log_model(sk_model=best, artifact_path="sklearn_pipeline")

        if args.register:
            model_uri = f"runs:/{run.info.run_id}/sklearn_pipeline"
            mlflow.register_model(model_uri=model_uri, name=args.register)

    print(f"Artifacts écrits dans {ARTIFACT_DIR} et {legacy_path}")


if __name__ == "__main__":
    main()
