"""Wrapper pyfunc do MLflow: dados brutos do cliente entram → probabilidade, score, decisão e motivos saem.

Por que pyfunc e não ``mlflow.sklearn.log_model``: servido como pyfunc, o modelo sklearn chama ``predict()``,
que devolve 0/1 com threshold 0.5 — ou seja, nem a probabilidade nem o threshold de negócio escolhido.
Aqui o pré-processamento, o modelo e o threshold viajam juntos, então o endpoint, a inferência em lote
e a avaliação executam exatamente o mesmo código.
"""

from __future__ import annotations

import importlib
import json
import os
import tempfile
from importlib.metadata import version

import joblib
import mlflow
import numpy as np
import pandas as pd
from mlflow.models import ModelSignature, infer_signature
from mlflow.types.schema import AnyType, ColSpec, Schema

from credit_scoring import config as c

N_MOTIVOS = 3


def _pkg(dist: str, module: str | None = None) -> str:
    """Fixa a versão instalada no ambiente de treino (``__version__`` primeiro, depois o dist-info)."""
    try:
        v = importlib.import_module(module or dist).__version__
    except (ImportError, AttributeError):
        v = version(dist)
    return f"{dist}=={v.split('+')[0]}"


def pip_requirements() -> list[str]:
    """Requirements embutidos no modelo para o container de serving (mesmas versões do treino)."""
    return [
        _pkg("mlflow"),
        _pkg("scikit-learn", "sklearn"),
        _pkg("xgboost"),
        _pkg("pandas"),
        _pkg("numpy"),
        _pkg("joblib"),
    ]


class CreditScoringModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.pipeline = joblib.load(context.artifacts["pipeline"])
        with open(context.artifacts["model_config"]) as f:
            cfg = json.load(f)
        self.threshold = float(cfg["threshold"])

    def predict(self, context, model_input: pd.DataFrame, params=None) -> pd.DataFrame:
        proba = self.pipeline.predict_proba(model_input)[:, 1]
        return pd.DataFrame(
            {
                "prob_inadimplencia": proba.astype("float64"),
                # Score de crédito 0–1000: quanto maior, menor o risco.
                "score": np.round((1 - proba) * 1000).astype("int32"),
                "decisao": np.where(proba >= self.threshold, c.REJECT, c.APPROVE),
                "motivos": self._motivos(model_input),
            }
        )

    def _motivos(self, model_input: pd.DataFrame) -> list[str]:
        """As features que mais aumentaram o risco de cada cliente (contribuições SHAP nativas do XGBoost)."""
        model = self.pipeline.named_steps["model"]
        if not hasattr(model, "get_booster"):
            return ["[]"] * len(model_input)
        import xgboost as xgb

        features = self.pipeline.named_steps["prep"].transform(model_input)
        contribs = model.get_booster().predict(xgb.DMatrix(features), pred_contribs=True)[:, :-1]
        motivos = []
        for row in contribs:
            top = np.argsort(-row)[:N_MOTIVOS]
            motivos.append(json.dumps([c.FEATURES[i] for i in top if row[i] > 0]))
        return motivos


def log_model(pipeline, threshold: float, X_example: pd.DataFrame, registered_model_name: str | None = None):
    """Loga (e opcionalmente registra no Unity Catalog) o modelo como um pyfunc autocontido.

    Precisa estar dentro de um ``mlflow.start_run``. Com ``registered_model_name`` no formato
    ``catalogo.schema.modelo`` (e ``mlflow.set_registry_uri("databricks-uc")``), cria uma nova versão.
    """
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    with tempfile.TemporaryDirectory() as tmp:
        pipeline_path = os.path.join(tmp, "pipeline.joblib")
        joblib.dump(pipeline, pipeline_path)
        cfg_path = os.path.join(tmp, "model_config.json")
        with open(cfg_path, "w") as f:
            json.dump(
                {"threshold": threshold, "input_columns": c.MODEL_INPUT_COLUMNS, "features": c.FEATURES}, f, indent=2
            )
        artifacts = {"pipeline": pipeline_path, "model_config": cfg_path}

        input_example = X_example[c.MODEL_INPUT_COLUMNS].reset_index(drop=True)

        # Signature: a saída é inferida de uma predição real do wrapper (também serve de teste de carga) e vira
        # ColSpec com "name" e "type". A entrada é declarada com tipo "any": o MLflow recusa converter int64 → double
        # ("Can not safely convert"), então um JSON com "age": 32 ou "MonthlyIncome": 3200 seria rejeitado pelo
        # endpoint. A conversão numérica fica no CreditPreprocessor (pd.to_numeric), que aceita inteiros, decimais e nulos.
        wrapper = CreditScoringModel()
        wrapper.load_context(type("Ctx", (), {"artifacts": artifacts})())
        output = infer_signature(model_input=input_example, model_output=wrapper.predict(None, input_example)).outputs
        inputs = Schema(
            [ColSpec(AnyType(), col, required=col not in c.NULLABLE_COLUMNS) for col in c.MODEL_INPUT_COLUMNS]
        )
        signature = ModelSignature(inputs=inputs, outputs=output)

        return mlflow.pyfunc.log_model(
            name="model",
            python_model=CreditScoringModel(),
            artifacts=artifacts,
            code_paths=[pkg_dir],
            signature=signature,
            input_example=input_example.head(5),
            pip_requirements=pip_requirements(),
            registered_model_name=registered_model_name,
        )
