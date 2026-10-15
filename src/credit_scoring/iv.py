"""Information Value (IV) e Weight of Evidence (WoE), usados na seleção de features."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _iv_from_bins(binned: pd.Series, target: pd.Series) -> float:
    total_events = target.sum()
    total_non_events = len(target) - total_events

    iv_table = target.groupby(binned, observed=True).agg(events="sum", total="count")
    iv_table["non_events"] = iv_table["total"] - iv_table["events"]
    iv_table["pct_events"] = iv_table["events"] / total_events
    iv_table["pct_non_events"] = iv_table["non_events"] / total_non_events

    # Evitar log(0)
    iv_table = iv_table[(iv_table["pct_events"] > 0) & (iv_table["pct_non_events"] > 0)]

    woe = np.log(iv_table["pct_events"] / iv_table["pct_non_events"])
    return float(((iv_table["pct_events"] - iv_table["pct_non_events"]) * woe).sum())


def calcular_iv(df: pd.DataFrame, feature: str, target: str, bins: int = 10) -> float:
    """IV de uma feature: contínuas (> 10 valores distintos) são discretizadas em ``bins`` quantis."""
    col = df[feature]
    binned = pd.qcut(col, q=bins, duplicates="drop") if col.nunique() > 10 else col
    return _iv_from_bins(binned, df[target])


def iv_contagem(df: pd.DataFrame, col: str, target: str) -> float:
    """IV para variáveis de contagem concentradas em zero, com bins de negócio fixos: 0, 1, 2, 3+."""
    binned = pd.cut(df[col], bins=[-1, 0, 1, 2, np.inf], labels=["0", "1", "2", "3+"])
    return _iv_from_bins(binned, df[target])


def classificar_iv(iv: float) -> str:
    """Classificação padrão de mercado: < 0.02 fraco, < 0.1 médio, < 0.3 forte, acima disso suspeito."""
    if iv < 0.02:
        return "Fraco"
    if iv < 0.1:
        return "Médio"
    if iv < 0.3:
        return "Forte"
    return "Suspeito (investigar leakage)"
