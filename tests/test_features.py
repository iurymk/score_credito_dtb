import numpy as np
import pandas as pd
import pytest

from credit_scoring import config as c
from credit_scoring.features import CreditPreprocessor


def test_output_has_the_15_features_in_model_order(credit_df):
    out = CreditPreprocessor().fit(credit_df).transform(credit_df)
    assert list(out.columns) == c.FEATURES
    assert len(out) == len(credit_df)
    assert not out.isna().any().any()


def test_statistics_are_learned_only_from_fit_data(credit_df):
    train, other = credit_df.iloc[:1000], credit_df.iloc[1000:].copy()
    other[c.INCOME] = other[c.INCOME] * 100  # se vazasse, a mediana/p99 mudaria

    prep = CreditPreprocessor().fit(train)
    assert prep.median_income_ == pytest.approx(train[c.INCOME].median())
    prep.transform(other)  # transformar não pode alterar o que foi aprendido
    assert prep.median_income_ == pytest.approx(train[c.INCOME].median())


def test_missing_flag_is_computed_before_imputation(credit_df):
    prep = CreditPreprocessor().fit(credit_df)
    out = prep.transform(credit_df)
    missing = credit_df[c.INCOME].isna()
    assert (out.loc[missing, "flag_missing_income"] == 1).all()
    assert (out.loc[~missing, "flag_missing_income"] == 0).all()
    assert (out.loc[missing, c.INCOME] == min(prep.median_income_, prep.cap_income_)).all()


def test_sentinel_codes_96_98_are_capped_at_real_max(credit_df):
    prep = CreditPreprocessor().fit(credit_df)
    out = prep.transform(credit_df)
    for col in c.DELINQUENCY_COLUMNS:
        real_max = credit_df.loc[~credit_df[col].isin(c.DELINQUENCY_SENTINELS), col].max()
        assert out[col].max() == real_max


def test_outliers_are_winsorized(credit_df):
    prep = CreditPreprocessor().fit(credit_df)
    out = prep.transform(credit_df)
    assert out[c.REVOLVING].max() <= prep.cap_revolving_
    assert out[c.DEBT_RATIO].max() <= prep.cap_debt_ratio_ < 2


def test_single_row_matches_batch_training_serving_parity(credit_df):
    """O endpoint recebe 1 cliente por vez; o resultado tem que ser idêntico ao do lote."""
    prep = CreditPreprocessor().fit(credit_df)
    batch = prep.transform(credit_df)
    for i in (0, 7, 123):
        single = prep.transform(credit_df.iloc[[i]])
        pd.testing.assert_series_equal(single.iloc[0], batch.iloc[i], check_names=False)


def test_accepts_json_like_input_with_nulls(credit_df):
    prep = CreditPreprocessor().fit(credit_df)
    record = {col: float(credit_df[col].iloc[0]) for col in c.MODEL_INPUT_COLUMNS}
    record[c.INCOME] = None
    record[c.DEPENDENTS] = None
    out = prep.transform(pd.DataFrame([record], dtype=object))
    assert out["flag_missing_income"].iloc[0] == 1
    assert np.isfinite(out.to_numpy(dtype=float)).all()


def test_missing_required_column_fails_loudly(credit_df):
    prep = CreditPreprocessor().fit(credit_df)
    with pytest.raises(ValueError, match=c.AGE):
        prep.transform(credit_df.drop(columns=[c.AGE]))
