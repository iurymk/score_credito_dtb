"""Leitura do dataset, validações de qualidade e splits estratificados train/val/test."""

from __future__ import annotations

import hashlib

import pandas as pd
from sklearn.model_selection import train_test_split

from credit_scoring import config as c

# Faixas aceitáveis: fora delas o dado de origem mudou e o pipeline para antes de treinar.
MIN_ROWS = 1_000
TARGET_RATE_RANGE = (0.02, 0.20)
MAX_NULL_FRACTION = {c.INCOME: 0.30, c.DEPENDENTS: 0.10}


def read_training_csv(path: str) -> pd.DataFrame:
    """Lê o ``cs-training.csv`` ou o ``cs-test.csv`` (a primeira coluna, sem nome, vira ``id``)."""
    df = pd.read_csv(path)
    return df.rename(columns={df.columns[0]: c.ID_COL})


def check_quality(df: pd.DataFrame, has_target: bool = True) -> dict:
    """Falha rápido em problemas de dados, antes de qualquer treino ou escoragem. Retorna estatísticas resumidas.

    ``has_target=False`` para a safra de produção, que ainda não tem o desfecho.
    """
    errors = []
    expected = [c.ID_COL, *([c.TARGET] if has_target else []), *c.RAW_COLUMNS]
    missing = [col for col in expected if col not in df.columns]
    if missing:
        raise ValueError(f"colunas ausentes no dataset: {missing}")

    n = len(df)
    if n < MIN_ROWS:
        errors.append(f"apenas {n} linhas (mínimo {MIN_ROWS})")
    if df[c.ID_COL].duplicated().any():
        errors.append(f"{int(df[c.ID_COL].duplicated().sum())} ids duplicados")

    target_rate = None
    if has_target:
        if not set(df[c.TARGET].dropna().unique()) <= {0, 1}:
            errors.append(f"target não binário: {sorted(df[c.TARGET].dropna().unique())[:10]}")
        if df[c.TARGET].isna().any():
            errors.append(f"{int(df[c.TARGET].isna().sum())} linhas sem target")
        target_rate = float(df[c.TARGET].mean())
        if not TARGET_RATE_RANGE[0] <= target_rate <= TARGET_RATE_RANGE[1]:
            errors.append(f"taxa de inadimplência {target_rate:.2%} fora de {TARGET_RATE_RANGE}")

    null_fraction = {col: float(df[col].isna().mean()) for col in c.RAW_COLUMNS}
    for col, frac in null_fraction.items():
        limit = MAX_NULL_FRACTION.get(col, 0.0)
        if frac > limit:
            errors.append(f"{col}: {frac:.2%} nulos (máximo {limit:.0%})")

    negatives = {col: int((df[col] < 0).sum()) for col in c.RAW_COLUMNS if (df[col] < 0).any()}
    if negatives:
        errors.append(f"valores negativos: {negatives}")

    if errors:
        raise ValueError("validação de qualidade falhou:\n  - " + "\n  - ".join(errors))
    return {
        "n_rows": n,
        "target_rate": round(target_rate, 4) if target_rate is not None else None,
        "null_fraction": {k: round(v, 4) for k, v in null_fraction.items() if v > 0},
        "n_age_zero": int((df[c.AGE] == 0).sum()),
    }


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Regras de linha (sem estatísticas, então sem risco de leakage): remove ``age <= 0``, um registro inválido."""
    return df[df[c.AGE] > 0].reset_index(drop=True)


def assign_splits(df: pd.DataFrame) -> pd.Series:
    """Coluna ``split`` (train/val/test), estratificada pelo target e determinística (ordem por ``id`` + seed)."""
    ordered = df.sort_values(c.ID_COL)
    rest, test = train_test_split(
        ordered, test_size=c.TEST_FRACTION, random_state=c.SEED, stratify=ordered[c.TARGET]
    )
    train, val = train_test_split(
        rest, test_size=c.VAL_FRACTION, random_state=c.SEED, stratify=rest[c.TARGET]
    )
    split = pd.Series("train", index=df.index, name="split")
    split[val.index] = "val"
    split[test.index] = "test"
    return split


def split_digest(df: pd.DataFrame, split: str) -> str:
    """Impressão digital estável dos ids de um split: prova que duas avaliações usaram o mesmo conjunto."""
    ids = df.loc[df["split"] == split, c.ID_COL].sort_values().astype(str)
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()[:16]
