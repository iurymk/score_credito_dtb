"""O modelo logado no MLflow recebe dados brutos (como o endpoint) e devolve probabilidade, score, decisão e motivos."""

import json

import mlflow
import pandas as pd
import pytest

from credit_scoring import config as c, modeling, serving


@pytest.fixture
def logged_model(credit_df, tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    mlflow.set_experiment("test")
    X, y = credit_df[c.RAW_COLUMNS], credit_df[c.TARGET]
    pipe = modeling.build_pipeline(modeling.CHOSEN_MODEL, y).fit(X, y)
    with mlflow.start_run():
        info = serving.log_model(pipe, threshold=0.6, X_example=X.head(10))
    return mlflow.pyfunc.load_model(info.model_uri), pipe


def test_pyfunc_matches_pipeline_and_applies_threshold(logged_model, credit_df):
    model, pipe = logged_model
    X = credit_df[c.MODEL_INPUT_COLUMNS].head(200).astype("float64")
    out = model.predict(X)

    assert list(out.columns) == ["prob_inadimplencia", "score", "decisao", "motivos"]
    expected = pipe.predict_proba(X)[:, 1]
    assert out["prob_inadimplencia"].to_numpy() == pytest.approx(expected, rel=1e-6)
    assert ((out["prob_inadimplencia"] >= 0.6) == (out["decisao"] == c.REJECT)).all()
    assert out["score"].between(0, 1000).all()
    for motivos in out["motivos"]:
        assert set(json.loads(motivos)) <= set(c.FEATURES)


def test_pyfunc_accepts_int_and_float_payloads(logged_model, credit_df):
    """Regressão: com a entrada declarada como double, o MLflow recusava int64 (JSON "age": 32) no endpoint."""
    model, _ = logged_model
    as_ints = credit_df[c.MODEL_INPUT_COLUMNS].head(5).fillna(0).round().astype("int64")
    as_floats = as_ints.astype("float64")
    out_int, out_float = model.predict(as_ints), model.predict(as_floats)
    pd.testing.assert_frame_equal(out_int, out_float)
    # um único cliente como chegaria por JSON, misturando inteiros e decimais
    record = {col: int(as_ints[col].iloc[0]) for col in c.MODEL_INPUT_COLUMNS}
    record[c.REVOLVING] = 0.95
    assert model.predict(pd.DataFrame([record]))["decisao"].iloc[0] in (c.APPROVE, c.REJECT)


def test_pyfunc_accepts_endpoint_payload_with_nulls(logged_model, credit_df):
    model, _ = logged_model
    record = {col: float(credit_df[col].iloc[0]) for col in c.MODEL_INPUT_COLUMNS}
    record[c.INCOME] = None
    record[c.DEPENDENTS] = None
    out = model.predict(pd.DataFrame([record]))
    assert out["decisao"].iloc[0] in (c.APPROVE, c.REJECT)
    assert 0 <= out["prob_inadimplencia"].iloc[0] <= 1
