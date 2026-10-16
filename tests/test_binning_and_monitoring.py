import numpy as np
import pandas as pd
import pytest

from credit_scoring import binning, config as c, data, evaluation, monitoring
from credit_scoring.features import CreditPreprocessor


@pytest.fixture
def features(credit_df):
    return CreditPreprocessor().fit(credit_df).transform(credit_df), credit_df[c.TARGET]


def test_bivariate_table_is_consistent(features):
    X, y = features
    table = binning.bivariate_table(binning.fit_binning(X[c.REVOLVING], y, c.REVOLVING))
    assert table["qtd"].sum() == len(X)
    assert table["evento"].sum() == y.sum()
    assert table["pct_total"].sum() == pytest.approx(1)
    assert (table["odds"] == table["evento"] / table["nao_evento"]).all()
    # Revolving é monotônico com o risco no dataset sintético: lift cresce ao longo das faixas
    assert table["lift"].is_monotonic_increasing


def test_business_splits_keep_rare_bins(features):
    """Regressão: com mínimo de 1% por faixa, o '3+' de atrasos deixava o solver inviável (IV = 0)."""
    X, y = features
    optb = binning.fit_binning(X[c.LATE_90], y, c.LATE_90)
    assert list(optb.splits) == [0.5, 1.5, 2.5]
    assert binning.bivariate_table(optb)["iv"].sum() > 0


def test_iv_ranking_sorted(features):
    X, y = features
    _, ranking = binning.analise_binning(X.assign(y=y.values), [c.REVOLVING, c.AGE, "flag_missing_income"], "y")
    assert ranking["iv"].is_monotonic_decreasing
    assert set(ranking.columns) == {"variavel", "iv", "classificacao"}


def test_psi_detects_shift_and_stability(features):
    X, y = features
    scores = np.random.default_rng(0).random(len(X))
    _, stable = monitoring.monitor(X, y, X, scores, scores)
    assert (stable["psi"] < 1e-9).all() and set(stable["status"]) == {"estável"}

    shifted = X.copy()
    shifted[c.REVOLVING] = shifted[c.REVOLVING] * 3  # população bem mais alavancada
    _, resumo = monitoring.monitor(X, y, shifted, scores, scores * 0.5)
    by_var = resumo.set_index("variavel")["status"]
    assert by_var[c.REVOLVING] == "instável"
    assert by_var["score_modelo"] == "instável"
    assert by_var[c.AGE] == "estável"


def test_faixas_and_cortes_tables():
    rng = np.random.default_rng(1)
    y = (rng.random(4000) < 0.1).astype(int)
    p = np.clip(y * 0.4 + rng.random(4000) * 0.6, 0, 1)

    faixas = evaluation.tabela_faixas(y, p)
    assert len(faixas) == 40 and faixas["qtd_total"].sum() == 4000
    assert faixas["pct_maus_acum"].iloc[-1] == pytest.approx(1)
    assert faixas["lift"].iloc[0] > 1 > faixas["lift"].iloc[-1]

    cortes = evaluation.tabela_cortes(y, p)
    assert cortes["recall_maus"].is_monotonic_increasing  # do mais restritivo ao mais permissivo
    assert cortes["pct_reprovados"].iloc[-1] == pytest.approx(1)


def test_threshold_for_min_recall_meets_target():
    rng = np.random.default_rng(2)
    y = (rng.random(5000) < 0.1).astype(int)
    p = np.clip(y * 0.3 + rng.random(5000) * 0.7, 0, 1)
    for target in (0.5, 0.7, 0.9):
        th = evaluation.threshold_for_min_recall(y, p, target)
        recall = ((p >= th) & (y == 1)).sum() / y.sum()
        assert recall >= target
        # é o maior corte possível: subir um pouco já fica abaixo da meta
        assert ((p >= th + 1e-6) & (y == 1)).sum() / y.sum() < target


def test_quality_check_for_unlabeled_safra(credit_df):
    safra = credit_df.drop(columns=[c.TARGET])
    assert data.check_quality(safra, has_target=False)["target_rate"] is None
