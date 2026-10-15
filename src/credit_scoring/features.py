"""Pré-processamento e feature engineering como um transformer do scikit-learn.

Reproduz as decisões dos notebooks 01_eda e 02_feature_engineering, com duas correções para produção:

1. **Sem leakage:** medianas, percentis e min/max são aprendidos em ``fit`` usando só o treino
   (na exploração eram calculados no dataset inteiro, antes do split).
2. **Sem divergência entre treino e serving:** o mesmo objeto (com os parâmetros aprendidos) vai
   dentro do modelo registrado, então o endpoint recebe dados brutos e aplica a mesma transformação.
   Isso substitui o ``pipeline_params.json`` + ``predictor.py`` da exploração.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from credit_scoring import config as c


def _to_numeric_frame(X: pd.DataFrame) -> pd.DataFrame:
    """Copia só as colunas usadas, como float (``None`` vindo de JSON vira NaN)."""
    missing = [col for col in c.MODEL_INPUT_COLUMNS if col not in X.columns]
    if missing:
        raise ValueError(f"colunas obrigatórias ausentes: {missing}")
    return X[c.MODEL_INPUT_COLUMNS].apply(pd.to_numeric, errors="coerce").astype("float64")


class CreditPreprocessor(BaseEstimator, TransformerMixin):
    """Dados brutos do cliente (``config.MODEL_INPUT_COLUMNS``) → as 15 features do modelo (``config.FEATURES``)."""

    def fit(self, X: pd.DataFrame, y=None):
        df = _to_numeric_frame(X)

        # Códigos 96/98 → maior contagem real vista no treino (mantém o sinal de "muito atraso"
        # sem esmagar a escala do score_risco, que na exploração ia até 98 e 294).
        self.delinquency_caps_ = {}
        for col in c.DELINQUENCY_COLUMNS:
            real = df.loc[~df[col].isin(c.DELINQUENCY_SENTINELS), col]
            self.delinquency_caps_[col] = float(real.max()) if real.notna().any() else 0.0
        df = self._fix_sentinels(df)

        # Imputação pela mediana (MonthlyIncome 19.8% e NumberOfDependents 2.6% de missing)
        self.median_income_ = float(df[c.INCOME].median())
        self.median_dependents_ = float(df[c.DEPENDENTS].median())
        df = self._impute(df)

        # Winsorização: p99 para RevolvingUtilization e MonthlyIncome; DebtRatio no p99 do grupo
        # plausível (< 2), porque o DebtRatio está corrompido quando a renda não foi informada.
        self.cap_revolving_ = float(df[c.REVOLVING].quantile(0.99))
        self.cap_income_ = float(df[c.INCOME].quantile(0.99))
        plausible = df.loc[df[c.DEBT_RATIO] < 2, c.DEBT_RATIO]
        self.cap_debt_ratio_ = float(plausible.quantile(0.99)) if len(plausible) else 2.0
        df = self._winsorize(df)

        # Limites da normalização min-max do score_risco
        total = df[c.DELINQUENCY_COLUMNS].sum(axis=1)
        self.norm_ = {
            "revolving": (float(df[c.REVOLVING].min()), float(df[c.REVOLVING].max())),
            "late_90": (float(df[c.LATE_90].min()), float(df[c.LATE_90].max())),
            "total_atrasos": (float(total.min()), float(total.max())),
        }
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = _to_numeric_frame(X)
        flag_missing_income = df[c.INCOME].isna().astype("float64")  # antes da imputação
        df = self._winsorize(self._impute(self._fix_sentinels(df)))

        total = df[c.DELINQUENCY_COLUMNS].sum(axis=1)
        df["flag_missing_income"] = flag_missing_income
        df["renda_per_capita"] = df[c.INCOME] / (df[c.DEPENDENTS] + 1)
        df["teve_atraso_90dias"] = (df[c.LATE_90] > 0).astype("float64")
        df["total_atrasos"] = total
        df["teve_qualquer_atraso"] = (total > 0).astype("float64")
        df["score_risco"] = (
            0.5 * self._normalize(df[c.REVOLVING], "revolving")
            + 0.3 * self._normalize(df[c.LATE_90], "late_90")
            + 0.2 * self._normalize(total, "total_atrasos")
        )
        return df[c.FEATURES]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(c.FEATURES, dtype=object)

    def learned_params(self) -> dict:
        """Parâmetros aprendidos no treino (equivalente ao ``pipeline_params.json`` da exploração)."""
        return {
            "delinquency_caps": self.delinquency_caps_,
            "median_income": self.median_income_,
            "median_dependents": self.median_dependents_,
            "cap_revolving": self.cap_revolving_,
            "cap_income": self.cap_income_,
            "cap_debt_ratio": self.cap_debt_ratio_,
            "score_risco_norm": self.norm_,
        }

    def _fix_sentinels(self, df: pd.DataFrame) -> pd.DataFrame:
        for col, cap in self.delinquency_caps_.items():
            df[col] = df[col].where(~df[col].isin(c.DELINQUENCY_SENTINELS), cap)
        df[c.DELINQUENCY_COLUMNS] = df[c.DELINQUENCY_COLUMNS].fillna(0.0)
        return df

    def _impute(self, df: pd.DataFrame) -> pd.DataFrame:
        df[c.INCOME] = df[c.INCOME].fillna(self.median_income_)
        df[c.DEPENDENTS] = df[c.DEPENDENTS].fillna(self.median_dependents_)
        return df

    def _winsorize(self, df: pd.DataFrame) -> pd.DataFrame:
        df[c.REVOLVING] = df[c.REVOLVING].clip(upper=self.cap_revolving_)
        df[c.INCOME] = df[c.INCOME].clip(upper=self.cap_income_)
        df[c.DEBT_RATIO] = df[c.DEBT_RATIO].clip(upper=self.cap_debt_ratio_)
        return df

    def _normalize(self, s: pd.Series, key: str) -> pd.Series:
        lo, hi = self.norm_[key]
        return (s - lo) / ((hi - lo) or 1.0)
