# Churn Telco — prédiction & rétention client (ML + API + dashboard)

Projet data science autour du churn dans un contexte **télécoms** : exploration des données IBM Telco, pipeline **scikit-learn / XGBoost** optimisé pour repérer les clients à risque, **API FastAPI** prête à être conteneurisée, **dashboard Streamlit** (scores, SHAP, segmentation) et **suivi d’expériences avec MLflow**.

![Python](https://img.shields.io/badge/python-3.11-blue)
![Streamlit](https://img.shields.io/badge/streamlit-1.28-red)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://first-app-cloud-vhjtmtp83vfptgbkssvy8i.streamlit.app/)

---

## Problématique

En télécoms, une partie des clients quitte l’opérateur sans signal explicite immédiat. L’enjeu est d’**anticiper** les départs pour lancer des **actions de rétention** ciblées (offres, contact humain, ajustement d’offre), plutôt que d’intervenir une fois le churn consommé.

**Objectifs du projet :**

- Repérer les profils et comportements associés au churn (EDA, corrélations, segments).
- Produire un modèle qui **limite les faux négatifs** sur la classe *churn* (priorité au **recall** côté clients qui partent).
- Industrialiser partiellement : API de scoring, dashboard métier, traçabilité des runs ML.

> Le fichier utilisé est `WA_Fn-UseC_-Telco-Customer-Churn.csv` (jeu IBM Telco). Il ne contient pas d’historique mois par mois : l’horizon « trois mois avant » se traduit ici par une **logique métier** (signaux de fragilité) et un **objectif de recall**, pas par une colonne temporelle explicite.

---

## Ce que livre le dépôt

| Livrable | Rôle |
| -------- | ---- |
| `notebooks/01_eda_churn.ipynb` | EDA : distributions, corrélations, visualisations matplotlib / seaborn, pistes d’actions |
| `ml/preprocessing.py` + `ml/train_pipeline.py` | Feature engineering via pipeline sklearn, **split stratifié 80/20**, grid search XGBoost, seuil calibré sur validation interne, **MLflow** (params, métriques, modèle) |
| `api/` | **FastAPI** : `GET /health`, `POST /predict` avec validation **Pydantic** |
| `Dockerfile` + `docker-compose.yml` | Image API + healthcheck |
| `app.py` | **Streamlit** : upload CSV batch, scores de risque, **SHAP** (top variables), segmentation |
| `ml/compare_runs.py` | Aperçu rapide des derniers runs MLflow en CLI |

Les modèles entraînés sont écrits dans `artifacts/churn_bundle.joblib` (et copie `churn_model.pkl` à la racine pour compatibilité). Pensez à lancer l’entraînement au moins une fois avant l’API ou le dashboard hors ligne.

---

## Fonctionnalités (résumé)

- **Scoring batch ou ligne** : probabilité de churn + classe selon un seuil appris sur la validation (objectif recall élevé sur les churns).
- **Dashboard** : histogrammes des scores, segments (faible → critique), export CSV des prédictions, aperçu SHAP sur un sous-échantillon.
- **API REST** : JSON aligné sur les colonnes du CSV Telco (sans `customerID` ni `Churn` pour la prédiction).
- **MLflow** : comparaison des runs, artefacts, possibilité d’enregistrer un modèle dans le registry (`--register NomDuModele`).

---

## Stack technique

| Catégorie | Technologies |
| --------- | ------------ |
| **ML** | pandas, scikit-learn, XGBoost, SHAP |
| **API** | FastAPI, Uvicorn, Pydantic |
| **UI** | Streamlit, matplotlib |
| **Ops / qualité ML** | MLflow, Docker, docker-compose |

---

## Performances (indicatif)

Les chiffres **varient** selon la graine, la grille d’hyperparamètres (`--quick` vs grille complète) et le split. Sur une exécution type (grille rapide, jeu Telco, hold-out 20 %) :

| Métrique | Ordre de grandeur |
| -------- | ----------------- |
| **Recall (classe churn)** | ≈ **80 %+** (objectif métier : ne pas rater trop de futurs churns) |
| **ROC-AUC** | ≈ **0,83** |
| **Précision (churn)** | plus modérée (compromis naturel quand on pousse le recall) |

Pour les valeurs exactes du dernier entraînement : consulter la console après `python -m ml.train_pipeline`, le fichier `artifacts/champion_meta.json` ou l’UI MLflow (`mlflow ui` → dossier `mlruns/`).

---

## Impact business (lecture cible)

Sans rejouer des chiffres de ROI non mesurés sur ce dépôt, le fil conducteur est :

- **Ciblage** : concentrer les actions de rétention sur les clients à **score élevé** plutôt que sur toute la base.
- **Priorisation** : segments « élevé / critique » dans Streamlit pour l’ordre de traitement.
- **Transparence** : SHAP et MLflow aident à **expliquer** et à **comparer** les versions du modèle devant les parties prenantes.

Les gains € réels dépendent du coût d’une action, du taux de succès des campagnes et de votre base réelle — à estimer en conditions opérationnelles.

---

## Démarrage rapide

```bash
# À la racine du projet
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS / Linux
# source .venv/bin/activate

pip install -r requirements.txt

# Entraîner le modèle + logger MLflow (grille rapide)
python -m ml.train_pipeline --quick
# Grille plus large (plus long) : sans --quick
# python -m ml.train_pipeline

# Dashboard
streamlit run app.py

# API (après entraînement : artifacts présents)
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# MLflow (autre terminal)
mlflow ui
```

**Docker (API seule)**

```bash
docker compose up --build
# Santé : http://localhost:8000/health
```

Variable utile : `CHURN_MODEL_PATH` (chemin vers le fichier `.joblib` du bundle si vous ne utilisez pas l’emplacement par défaut).

---

## Licence

Projet personnel / démonstration — MIT licence.
