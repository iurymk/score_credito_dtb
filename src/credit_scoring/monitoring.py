"""Monitoramento de estabilidade: PSI (Population Stability Index) entre a base de treino e uma safra de produção.

O PSI compara distribuições, então não precisa do target: a safra pode ser escorada e monitorada
antes de se saber quem inadimpliu. Faixas das features = optimal binning ajustado no treino;
faixas do score = decis do score no treino.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from credit_scoring import binning

PSI_ESTAVEL, PSI_ATENCAO = 0.10, 0.25  # referência de mercado
EPS = 1e-4  # evita log(0) em faixas vazias


def psi_status(psi: float) -> str:
    if psi < PSI_ESTAVEL:
        return "estável"
    if psi < PSI_ATENCAO:
        return "atenção"
    return "instável"


def psi_table(esperado: pd.Series, atual: pd.Series, variavel: str, ordem: list[str] | None = None) -> pd.DataFrame:
    """PSI por faixa: (%atual - %esperado) * ln(%atual / %esperado). ``esperado`` e ``atual`` são rótulos de faixa."""
    vistos = set(esperado.dropna().astype(str)) | set(atual.dropna().astype(str))
    ordem = list(ordem or [])
    faixas = [f for f in ordem if f in vistos] + sorted(vistos - set(ordem))
    pe = esperado.astype(str).value_counts(normalize=True).reindex(faixas, fill_value=0.0)
    pa = atual.astype(str).value_counts(normalize=True).reindex(faixas, fill_value=0.0)
    pe_s, pa_s = pe.clip(lower=EPS), pa.clip(lower=EPS)
    return pd.DataFrame(
        {
            "variavel": variavel,
            "faixa": [f"{i + 1:02d}. {f}" if not f[:2].isdigit() else f for i, f in enumerate(faixas)],
            "pct_esperado": pe.to_numpy(),
            "pct_atual": pa.to_numpy(),
            "psi": ((pa_s - pe_s) * np.log(pa_s / pe_s)).to_numpy(),
        }
    )


def score_bins(scores: np.ndarray, n: int = 10) -> np.ndarray:
    """Limites dos decis do score no treino (as extremidades viram ±inf para cobrir qualquer valor novo)."""
    edges = np.unique(np.quantile(scores, np.linspace(0, 1, n + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


def label_score(scores, edges: np.ndarray) -> pd.Series:
    idx = np.clip(np.searchsorted(edges, scores, side="right") - 1, 0, len(edges) - 2)
    labels = [f"{i + 1:02d}. [{edges[i]:.3f}, {edges[i + 1]:.3f})" for i in range(len(edges) - 1)]
    return pd.Series([labels[i] for i in idx])


def monitor(
    train_features: pd.DataFrame,
    y_train: pd.Series,
    safra_features: pd.DataFrame,
    train_scores: np.ndarray,
    safra_scores: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """PSI de cada feature do modelo e do score. Retorna (detalhe por faixa, resumo por variável)."""
    detalhes = []
    for col in train_features.columns:
        optb = binning.fit_binning(train_features[col], y_train, col)
        esperado = pd.Series(optb.transform(train_features[col].to_numpy(), metric="bins"))
        atual = pd.Series(optb.transform(safra_features[col].to_numpy(), metric="bins"))
        detalhes.append(psi_table(esperado, atual, col, ordem=binning.bin_order(optb)))

    edges = score_bins(train_scores)
    detalhes.append(psi_table(label_score(train_scores, edges), label_score(safra_scores, edges), "score_modelo"))

    detalhe = pd.concat(detalhes, ignore_index=True)
    resumo = detalhe.groupby("variavel", as_index=False)["psi"].sum().sort_values("psi", ascending=False)
    resumo["status"] = resumo["psi"].apply(psi_status)
    return detalhe, resumo.reset_index(drop=True)
