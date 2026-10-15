# Databricks notebook source
# MAGIC %md
# MAGIC # 3 · Treino e registro no Unity Catalog
# MAGIC Treina o modelo escolhido na experimentação (**XGBoost**, hiperparâmetros do `03_modeling`) e o registra
# MAGIC no Unity Catalog. Não há seleção automática nem retreino: o modelo é o que foi decidido na exploração.
# MAGIC
# MAGIC 1. **Comparação** (registro histórico): baseline, regressão logística, random forest e XGBoost, cada um em um
# MAGIC    run filho do MLflow, avaliados na validação — documenta por que o XGBoost foi o escolhido.
# MAGIC 2. **Modelo final**: XGBoost dentro de `Pipeline(CreditPreprocessor → XGBClassifier)`, validação cruzada no treino,
# MAGIC    threshold escolhido na **validação** (maximiza F1) e registro como pyfunc: dados brutos → probabilidade + decisão.

# COMMAND ----------

import json, os, sys

sys.path.insert(0, os.path.abspath(".."))

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

from credit_scoring import config as c, evaluation, modeling, serving

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
model_name = dbutils.widgets.get("model_name")  # catalogo.schema.modelo
experiment_path = dbutils.widgets.get("experiment_path")
job_run_id = dbutils.widgets.get("job_run_id")

silver_table = f"{catalog}.{schema}.silver_credit"

# Unity Catalog como registry; log manual (sem autolog) para controlar exatamente o que é registrado.
mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(experiment_path)
mlflow.sklearn.autolog(disable=True)

# COMMAND ----------
# MAGIC %md ## Dados (silver)

# COMMAND ----------

silver_sdf = spark.table(silver_table)
silver = silver_sdf.toPandas()
split = {s: silver[silver["split"] == s] for s in c.SPLITS}
X_train, y_train = split["train"][c.RAW_COLUMNS], split["train"][c.TARGET]
X_val, y_val = split["val"][c.RAW_COLUMNS], split["val"][c.TARGET]
print({s: len(d) for s, d in split.items()})

# COMMAND ----------
# MAGIC %md ## 1 · Comparação de modelos (validação)

# COMMAND ----------

comparison = []
with mlflow.start_run(run_name="comparacao_modelos") as parent:
    mlflow.set_tags({"stage": "comparison", "job_run_id": job_run_id})
    for model_type in modeling.COMPARISON_MODELS:
        with mlflow.start_run(run_name=model_type, nested=True):
            pipe = modeling.build_pipeline(model_type, y_train).fit(X_train, y_train)
            proba = pipe.predict_proba(X_val)[:, 1]
            metrics = evaluation.classification_metrics(y_val, proba, threshold=0.5)
            mlflow.log_params({"model_type": model_type, **modeling.model_params(pipe)})
            mlflow.log_metrics({"val_auc": metrics["auc"], "val_ks": metrics["ks"], "val_gini": metrics["gini"]})
            comparison.append({"modelo": model_type, "val_auc": metrics["auc"], "val_ks": metrics["ks"]})
    comparison = pd.DataFrame(comparison).sort_values("val_auc", ascending=False)
    mlflow.log_table(comparison, "comparacao/modelos.json")

display(comparison)

# COMMAND ----------
# MAGIC %md ## 2 · Modelo final: XGBoost → Unity Catalog

# COMMAND ----------

final = modeling.build_pipeline(modeling.CHOSEN_MODEL, y_train).fit(X_train, y_train)
cv_auc_mean, cv_auc_std = modeling.cross_val_auc(modeling.CHOSEN_MODEL, X_train, y_train)

val_proba = final.predict_proba(X_val)[:, 1]
threshold = evaluation.best_f1_threshold(y_val, val_proba)
val_metrics = evaluation.classification_metrics(y_val, val_proba, threshold)
print(f"threshold={threshold:.4f} val_auc={val_metrics['auc']:.4f} val_ks={val_metrics['ks']:.4f} cv_auc={cv_auc_mean:.4f}±{cv_auc_std:.4f}")

params_finais = {
    **modeling.model_params(final),
    "model_type": modeling.CHOSEN_MODEL,
    "threshold": round(threshold, 6),
    "threshold_criterio": "max F1 na validação",
    "n_train": len(X_train),
    "n_val": len(X_val),
    "n_features": len(c.FEATURES),
}

with mlflow.start_run(run_name="xgboost_final") as run:
    mlflow.set_tags({"stage": "train", "job_run_id": job_run_id, "model_type": modeling.CHOSEN_MODEL})

    # 1. Parâmetros
    mlflow.log_params(params_finais)

    # 2. Métricas (validação + estabilidade na validação cruzada)
    mlflow.log_metrics(
        {f"val_{k}": v for k, v in val_metrics.items() if isinstance(v, float)}
        | {"cv_auc_mean": cv_auc_mean, "cv_auc_std": cv_auc_std}
    )

    # Linhagem e artefatos de apoio
    mlflow.log_input(mlflow.data.from_spark(silver_sdf, table_name=silver_table), context="training")
    mlflow.log_dict(final.named_steps["prep"].learned_params(), "preprocessamento/parametros_aprendidos.json")
    mlflow.log_dict({"features": c.FEATURES, "input_columns": c.MODEL_INPUT_COLUMNS}, "preprocessamento/features.json")
    mlflow.log_figure(evaluation.threshold_figure(y_val, val_proba, threshold), "validacao/threshold.png")

    # 3. Modelo: log + registro no Unity Catalog (nova versão de catalogo.schema.modelo)
    model_info = serving.log_model(final, threshold, X_val.head(20), registered_model_name=model_name)

version = str(model_info.registered_model_version)
client = MlflowClient(registry_uri="databricks-uc")
for key, value in {"model_type": modeling.CHOSEN_MODEL, "threshold": f"{threshold:.6f}",
                   "val_auc": f"{val_metrics['auc']:.6f}", "job_run_id": job_run_id}.items():
    client.set_model_version_tag(model_name, version, key, value)

print(f"registrado: {model_name} v{version}")
print(f"  URI: {model_info.model_uri}")
print(f"  Run ID: {run.info.run_id}")

# COMMAND ----------

dbutils.jobs.taskValues.set("model_version", version)
dbutils.jobs.taskValues.set("run_id", run.info.run_id)
dbutils.notebook.exit(json.dumps({"model_version": version, "threshold": threshold, "val_auc": val_metrics["auc"]}))
