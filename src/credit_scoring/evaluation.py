"""Métricas de crédito, escolha do threshold, fairness por segmento, impacto financeiro e portões de qualidade."""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from credit_scoring import config as c

AGE_BINS = [0, 25, 35, 45, 55, 65, np.inf]
AGE_LABELS = ["<25", "25-35", "35-45", "45-55", "55-65", "65+"]
MIN_EVENTS_PER_SEGMENT = 10  # abaixo disso a AUC do segmento não é confiável


def ks_statistic(y_true, y_proba) -> float:
    """KS: separação máxima entre as distribuições de score de bons e maus (métrica padrão do mercado de crédito)."""
    y_true, y_proba = np.asarray(y_true), np.asarray(y_proba)
    return float(ks_2samp(y_proba[y_true == 1], y_proba[y_true == 0]).statistic)


def best_f1_threshold(y_true, y_proba) -> float:
    """Threshold que maximiza o F1 na curva precision-recall (critério do notebook 04_evaluation)."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    f1 = 2 * precisions * recalls / (precisions + recalls + 1e-8)
    return float(thresholds[int(np.argmax(f1[:-1]))])


def classification_metrics(y_true, y_proba, threshold: float) -> dict:
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    auc = roc_auc_score(y_true, y_proba)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "auc": float(auc),
        "gini": float(2 * auc - 1),
        "ks": ks_statistic(y_true, y_proba),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def segment_auc(raw: pd.DataFrame, y_true, y_proba) -> pd.DataFrame:
    """AUC por faixa etária e por quartil de renda (renda não informada vira um segmento próprio)."""
    df = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(y_proba)}, index=raw.index)
    df["faixa_etaria"] = pd.cut(raw[c.AGE], bins=AGE_BINS, labels=AGE_LABELS)
    income = raw[c.INCOME]
    quartis = pd.qcut(income, q=4, labels=["Q1 baixa", "Q2 média-baixa", "Q3 média-alta", "Q4 alta"])
    df["faixa_renda"] = quartis.cat.add_categories("não informada").fillna("não informada")

    rows = []
    for segment in ("faixa_etaria", "faixa_renda"):
        for band, g in df.groupby(segment, observed=True):
            if g["y"].sum() >= MIN_EVENTS_PER_SEGMENT and g["y"].nunique() == 2:
                rows.append({
                    "segmento": segment, "faixa": str(band), "n": len(g),
                    "taxa_inadimplencia": float(g["y"].mean()), "auc": float(roc_auc_score(g["y"], g["p"])),
                })
    return pd.DataFrame(rows)


def fairness_gaps(segments: pd.DataFrame) -> dict:
    """Diferença entre a maior e a menor AUC dentro de cada segmento."""
    return {
        f"auc_gap_{seg}": float(g["auc"].max() - g["auc"].min()) for seg, g in segments.groupby("segmento")
    }


def financial_impact(fn: int, fp: int) -> dict:
    """Simulação de impacto com as premissas de negócio da exploração (``config``)."""
    perda_fn = fn * c.TICKET_MEDIO * c.TAXA_PERDA_INADIMPLENCIA
    receita_perdida_fp = fp * c.RECEITA_POR_CLIENTE
    return {
        "perda_inadimplentes_aprovados": perda_fn,
        "receita_perdida_bons_reprovados": receita_perdida_fp,
        "custo_total_erros": perda_fn + receita_perdida_fp,
    }


@dataclass(frozen=True)
class Gates:
    """Critérios mínimos para um modelo virar @champion."""

    min_auc: float
    min_ks: float
    max_fairness_gap: float


def gate_failures(metrics: dict, gaps: dict, gates: Gates) -> list[str]:
    failures = []
    if metrics["auc"] < gates.min_auc:
        failures.append(f"AUC {metrics['auc']:.4f} < {gates.min_auc}")
    if metrics["ks"] < gates.min_ks:
        failures.append(f"KS {metrics['ks']:.4f} < {gates.min_ks}")
    for name, gap in gaps.items():
        if gap > gates.max_fairness_gap:
            failures.append(f"{name} {gap:.4f} > {gates.max_fairness_gap}")
    return failures


# ── Gráficos (logados no MLflow) ───────────────────────────────────────────


def roc_figure(y_true, y_proba, title: str):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#e74c3c", linewidth=2, label=f"AUC={roc_auc_score(y_true, y_proba):.4f}")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Aleatório")
    ax.set_xlabel("Taxa de falsos positivos")
    ax.set_ylabel("Taxa de verdadeiros positivos")
    ax.set_title(title, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def threshold_figure(y_true, y_proba, threshold: float):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    f1 = 2 * precisions * recalls / (precisions + recalls + 1e-8)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, precisions[:-1], "#e74c3c", label="Precision", linewidth=2)
    ax.plot(thresholds, recalls[:-1], "#2ecc71", label="Recall", linewidth=2)
    ax.plot(thresholds, f1[:-1], "#3498db", label="F1", linewidth=2)
    ax.axvline(threshold, color="orange", linestyle="--", label=f"Threshold escolhido ({threshold:.3f})")
    ax.set_xlabel("Threshold")
    ax.set_title("Precision, Recall e F1 por threshold (validação)", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def confusion_figure(metrics: dict, threshold: float):
    cm = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v:,}", ha="center", va="center", color="white" if v > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["Pred: Adimplente", "Pred: Inadimplente"])
    ax.set_yticks([0, 1], ["Real: Adimplente", "Real: Inadimplente"])
    ax.set_title(f"Matriz de confusão (teste) — threshold {threshold:.3f}", fontweight="bold")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig
