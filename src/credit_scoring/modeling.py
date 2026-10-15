"""Construção dos modelos: o XGBoost escolhido e os modelos de comparação do notebook 03_modeling.

Todos são ``Pipeline(CreditPreprocessor → classificador)``, então recebem dados brutos.
"""

from __future__ import annotations

import pandas as pd
import xgboost as xgb
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from credit_scoring import config as c
from credit_scoring.features import CreditPreprocessor

CHOSEN_MODEL = "xgboost"


def scale_pos_weight(y: pd.Series) -> float:
    """Trata o desbalanceamento no XGBoost: n_negativos / n_positivos."""
    return float((y == 0).sum() / (y == 1).sum())


def build_pipeline(model_type: str, y_train: pd.Series) -> Pipeline:
    """Pipeline completo (pré-processamento + classificador) com os hiperparâmetros da exploração."""
    if model_type == "xgboost":
        clf = xgb.XGBClassifier(**c.XGB_PARAMS, scale_pos_weight=scale_pos_weight(y_train))
        return Pipeline([("prep", CreditPreprocessor()), ("model", clf)])
    if model_type == "logistic_regression":
        clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=c.SEED, C=0.1)
        # XGBoost lida com NaN nativamente; a regressão logística não (só ocorre em entradas incompletas).
        return Pipeline([("prep", CreditPreprocessor()), ("scaler", StandardScaler()), ("model", clf)])
    if model_type == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=50,
            class_weight="balanced", random_state=c.SEED, n_jobs=-1,
        )
        return Pipeline([("prep", CreditPreprocessor()), ("model", clf)])
    if model_type == "baseline":
        return Pipeline([("prep", CreditPreprocessor()), ("model", DummyClassifier(strategy="most_frequent"))])
    raise ValueError(f"modelo desconhecido: {model_type}")


COMPARISON_MODELS = ("baseline", "logistic_regression", "random_forest", CHOSEN_MODEL)


def model_params(pipeline: Pipeline) -> dict:
    """Hiperparâmetros do classificador final, para logar no MLflow."""
    params = pipeline.named_steps["model"].get_params()
    return {k: v for k, v in params.items() if isinstance(v, (int, float, str, bool)) and v is not None}


def cross_val_auc(model_type: str, X: pd.DataFrame, y: pd.Series, folds: int = 5) -> tuple[float, float]:
    """AUC média e desvio da validação cruzada estratificada no treino (estabilidade do modelo)."""
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=c.SEED)
    scores = cross_val_score(build_pipeline(model_type, y), X, y, cv=skf, scoring="roc_auc")
    return float(scores.mean()), float(scores.std())
