import numpy as np
import pytest

from credit_scoring import config as c, data, evaluation


def test_quality_check_passes_on_valid_data(credit_df):
    stats = data.check_quality(credit_df)
    assert stats["n_rows"] == len(credit_df)


def test_quality_check_fails_fast_on_broken_data(credit_df):
    broken = credit_df.copy()
    broken.loc[:10, c.TARGET] = 2
    broken.loc[:3000, c.AGE] = np.nan
    with pytest.raises(ValueError) as exc:
        data.check_quality(broken)
    assert "target não binário" in str(exc.value) and "age" in str(exc.value)


def test_clean_removes_invalid_age(credit_df):
    credit_df.loc[0, c.AGE] = 0
    assert (data.clean(credit_df)[c.AGE] > 0).all()


def test_splits_are_stratified_and_deterministic(credit_df):
    split = data.assign_splits(credit_df)
    assert split.value_counts(normalize=True)["test"] == pytest.approx(c.TEST_FRACTION, abs=0.01)
    rates = credit_df.groupby(split)[c.TARGET].mean()
    assert rates.max() - rates.min() < 0.01
    shuffled = credit_df.sample(frac=1, random_state=1)  # ordem de leitura do Delta não é garantida
    assert (data.assign_splits(shuffled).sort_index() == split).all()


def test_ks_and_threshold():
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    assert evaluation.ks_statistic(y, p) == pytest.approx(1.0)
    assert 0.3 < evaluation.best_f1_threshold(y, p) <= 0.7


def test_gates():
    gates = evaluation.Gates(min_auc=0.85, min_ks=0.4, max_fairness_gap=0.05)
    ok = {"auc": 0.87, "ks": 0.58}
    assert evaluation.gate_failures(ok, {"auc_gap_faixa_etaria": 0.02}, gates) == []
    failures = evaluation.gate_failures({"auc": 0.80, "ks": 0.3}, {"auc_gap_faixa_renda": 0.09}, gates)
    assert len(failures) == 3


def test_segment_auc_and_financial_impact(credit_df):
    rng = np.random.default_rng(0)
    proba = np.clip(credit_df[c.TARGET] * 0.5 + rng.random(len(credit_df)) * 0.5, 0, 1)
    seg = evaluation.segment_auc(credit_df, credit_df[c.TARGET], proba)
    assert set(seg["segmento"]) == {"faixa_etaria", "faixa_renda"}
    assert "não informada" in set(seg["faixa"])
    impact = evaluation.financial_impact(fn=10, fp=20)
    assert impact["custo_total_erros"] == 10 * c.TICKET_MEDIO * c.TAXA_PERDA_INADIMPLENCIA + 20 * c.RECEITA_POR_CLIENTE
