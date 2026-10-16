"""Optimal binning: análise bivariada (IV, WoE, odds, lift por faixa) e faixas para o monitoramento de PSI.

Versão de biblioteca da ``analise_binning2`` usada na seleção de variáveis: as faixas são escolhidas pelo
``optbinning`` para maximizar o IV com taxa de evento monotônica, em vez de quantis fixos.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from optbinning import OptimalBinning

from credit_scoring import config as c
from credit_scoring.iv import classificar_iv

# Parâmetros padrão (os mesmos da seleção de variáveis original)
SOLVER = "cp"
MIN_BIN_SIZE = 0.01  # cada faixa com pelo menos 1% da base
MAX_N_BINS = 5
PREBINNING = "quantile"

# Variáveis de contagem concentradas em zero (> 90% de zeros): o prebinning por quantis falha,
# então os cortes são de negócio: 0, 1, 2, 3+.
USER_SPLITS = {col: [0.5, 1.5, 2.5] for col in [*c.DELINQUENCY_COLUMNS, "total_atrasos"]}
# Flags 0/1: tratadas como categóricas.
BINARY_FEATURES = [
    "flag_missing_income", "flag_missing_dependents",
    "teve_atraso_30dias", "teve_atraso_60dias", "teve_atraso_90dias", "teve_qualquer_atraso",
]


def fit_binning(x: pd.Series, y: pd.Series, name: str) -> OptimalBinning:
    """Ajusta o optimal binning de uma variável contra o target binário."""
    fitted = False
    if name in BINARY_FEATURES:
        optb = OptimalBinning(name=name, dtype="categorical", solver=SOLVER, min_bin_size=MIN_BIN_SIZE)
    elif name in USER_SPLITS:
        # Cortes de negócio fixos: sem tamanho mínimo de faixa nem restrição de monotonicidade (o "3+" de
        # atrasos tem ~0,5% da base; com essas restrições o solver fica inviável e devolve uma faixa só, IV 0).
        splits = USER_SPLITS[name]
        optb = OptimalBinning(
            name=name, dtype="numerical", solver=SOLVER, min_prebin_size=1e-4, min_bin_size=1e-4,
            monotonic_trend=None, user_splits=splits, user_splits_fixed=[True] * len(splits),
        )
        try:
            optb.fit(x.to_numpy(), y.to_numpy())
            fitted = True
        except ValueError:
            # Um corte fixo gerou uma faixa sem eventos (ou sem não-eventos): os cortes viram sugestões
            # e o solver pode fundir faixas.
            optb = OptimalBinning(
                name=name, dtype="numerical", solver=SOLVER, min_prebin_size=1e-4, min_bin_size=1e-4,
                monotonic_trend=None, user_splits=splits,
            )
    else:
        optb = OptimalBinning(
            name=name, dtype="numerical", solver=SOLVER, min_prebin_size=MIN_BIN_SIZE, min_bin_size=MIN_BIN_SIZE,
            max_n_prebins=MAX_N_BINS, max_n_bins=MAX_N_BINS, prebinning_method=PREBINNING,
        )
    if not fitted:
        optb.fit(x.to_numpy(), y.to_numpy())
    if optb.status != "OPTIMAL":
        raise RuntimeError(f"optimal binning de {name} terminou com status {optb.status}: revise os parâmetros")
    return optb


def bin_order(optb: OptimalBinning) -> list[str]:
    """Rótulos das faixas na ordem natural, no mesmo formato de ``optb.transform(metric="bins")``."""
    table = optb.binning_table.build()
    return [str(b) for i, b in zip(table.index, table["Bin"]) if i != "Totals" and str(b) != "Special"]


def bivariate_table(optb: OptimalBinning) -> pd.DataFrame:
    """Tabela bivariada por faixa: volume, taxa de evento, odds, WoE, IV e lift.

    Mantém a faixa ``Missing`` quando ela tem registros (em MonthlyIncome é ~20% da base e informativa).
    """
    raw = optb.binning_table.build()
    raw = raw[~raw.index.isin(["Totals"]) & (raw["Bin"].astype(str) != "Special") & (raw["Count"] > 0)].copy()

    evento, nao_evento, qtd = raw["Event"].astype(float), raw["Non-event"].astype(float), raw["Count"].astype(float)
    pct_evento, pct_nao_evento = evento / evento.sum(), nao_evento / nao_evento.sum()
    taxa_media = evento.sum() / qtd.sum()

    table = pd.DataFrame(
        {
            "variavel": optb.name,
            "faixa": [f"{i + 1}. {b}" if not isinstance(b, (list, np.ndarray)) else f"{i + 1}. {list(b)}"
                      for i, b in enumerate(raw["Bin"])],
            "qtd": qtd.astype(int).to_numpy(),
            "pct_total": (qtd / qtd.sum()).to_numpy(),
            "evento": evento.astype(int).to_numpy(),
            "nao_evento": nao_evento.astype(int).to_numpy(),
            "taxa_evento": (evento / qtd).to_numpy(),
            # Odds: eventos por não-evento dentro da faixa (= taxa / (1 - taxa))
            "odds": (evento / nao_evento.replace(0, np.nan)).to_numpy(),
            "pct_evento": pct_evento.to_numpy(),
            "pct_nao_evento": pct_nao_evento.to_numpy(),
            # WoE positivo = faixa com mais risco que a média (mesma convenção do iv.py)
            "woe": np.log((pct_evento + 1e-9) / (pct_nao_evento + 1e-9)).to_numpy(),
            "lift": ((evento / qtd) / taxa_media).to_numpy(),
        }
    )
    table["iv"] = (table["pct_evento"] - table["pct_nao_evento"]) * table["woe"]
    return table


def analise_binning(df: pd.DataFrame, features: list[str], target: str) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Análise bivariada de várias variáveis. Retorna ({variavel: tabela}, ranking de IV)."""
    bivariadas = {f: bivariate_table(fit_binning(df[f], df[target], f)) for f in features}
    ranking = (
        pd.DataFrame({"variavel": list(bivariadas), "iv": [t["iv"].sum() for t in bivariadas.values()]})
        .sort_values("iv", ascending=False)
        .reset_index(drop=True)
    )
    ranking["classificacao"] = ranking["iv"].apply(classificar_iv)
    return bivariadas, ranking


def plot_lift(table: pd.DataFrame):
    """% de eventos e % da base por faixa (barras) com o lift (linha), como no painel de bivariadas."""
    x = np.arange(len(table))
    fig, ax1 = plt.subplots(figsize=(10, 5))
    width = 0.38
    ax1.bar(x - width / 2, table["pct_evento"] * 100, width, label="% dos eventos", color="#e74c3c")
    ax1.bar(x + width / 2, table["pct_total"] * 100, width, label="% da base", color="#95a5a6")
    ax1.set_xticks(x, table["faixa"], rotation=20, ha="right")
    ax1.set_ylabel("%")
    ax2 = ax1.twinx()
    ax2.plot(x, table["lift"], "o-", color="black", label="Lift")
    for i, v in enumerate(table["lift"]):
        ax2.text(i, v + 0.05, f"{v:.2f}", ha="center")
    ax2.axhline(1, color="gray", linestyle="--", linewidth=1)
    ax2.set_ylabel("Lift")
    iv = table["iv"].sum()
    ax1.set_title(f"Optimal binning — {table['variavel'].iloc[0]} (IV={iv:.3f})", fontweight="bold")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left")
    fig.tight_layout()
    return fig
