import os

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
import pytest

from credit_scoring import config as c

REAL_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "raw", c.TRAINING_FILE)


def make_credit_data(n: int = 4000, seed: int = 0) -> pd.DataFrame:
    """Dataset sintético com o mesmo schema do Give Me Some Credit (nulos, códigos 96/98, outliers)."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            c.ID_COL: np.arange(1, n + 1),
            c.REVOLVING: rng.beta(0.6, 1.4, n) * 1.2,
            c.AGE: rng.integers(21, 95, n),
            c.LATE_30_59: rng.poisson(0.3, n),
            c.DEBT_RATIO: rng.lognormal(-1, 1.5, n),
            c.INCOME: rng.lognormal(8.5, 0.6, n).round(),
            c.OPEN_CREDIT: rng.poisson(8, n),
            c.LATE_90: rng.poisson(0.1, n),
            c.REAL_ESTATE: rng.poisson(1, n),
            c.LATE_60_89: rng.poisson(0.08, n),
            c.DEPENDENTS: rng.poisson(0.8, n).astype(float),
        }
    )
    df.loc[rng.random(n) < 0.2, c.INCOME] = np.nan
    df.loc[rng.random(n) < 0.03, c.DEPENDENTS] = np.nan
    df.loc[rng.choice(n, 10, replace=False), c.DELINQUENCY_COLUMNS] = 98
    df.loc[rng.choice(n, 5, replace=False), c.REVOLVING] = 5000.0

    logit = -3.2 + 2.0 * df[c.REVOLVING].clip(upper=1.2) + 0.8 * df[c.LATE_90].clip(upper=5) - 0.02 * (df[c.AGE] - 50)
    df[c.TARGET] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return df[[c.ID_COL, c.TARGET, *c.RAW_COLUMNS]]


@pytest.fixture
def credit_df() -> pd.DataFrame:
    return make_credit_data()


@pytest.fixture
def real_df() -> pd.DataFrame:
    if not os.path.exists(REAL_CSV):
        pytest.skip(f"{REAL_CSV} não encontrado (veja o README)")
    from credit_scoring import data

    return data.read_training_csv(REAL_CSV)
