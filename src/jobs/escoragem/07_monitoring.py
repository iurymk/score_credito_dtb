# Databricks notebook source
# MAGIC %md
# MAGIC # Esteira de escoragem · 7 · Monitoramento de estabilidade (PSI)
# MAGIC Compara a população da safra com a população de treino (**esperado × atual**), para cada feature do modelo e
# MAGIC para o score, usando o **PSI** (Population Stability Index):
# MAGIC
# MAGIC | PSI | Leitura |
# MAGIC |---|---|
# MAGIC | < 0,10 | estável |
# MAGIC | 0,10 – 0,25 | atenção: investigar |
# MAGIC | > 0,25 | instável: o modelo pode não representar mais a população |
# MAGIC
# MAGIC Faixas das features = optimal binning ajustado no treino; faixas do score = decis do treino. O PSI não precisa
# MAGIC do target, então a safra é monitorada antes de se saber quem inadimpliu. Resultado em `gold_monitoramento_psi`
# MAGIC (por faixa) e `gold_monitoramento_resumo` (por variável), que alimentam o painel de monitoramento.

# COMMAND ----------

import json, os, sys

sys.path.insert(0, os.path.abspath("../.."))

import mlflow
from pyspark.sql import functions as F

from credit_scoring import config as c, monitoring

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
model_name = dbutils.widgets.get("model_name")
safra = dbutils.widgets.get("safra")
version = dbutils.jobs.taskValues.get("batch_scoring", "model_version", debugValue="1")

mlflow.set_registry_uri("databricks-uc")


def table(name: str) -> str:
    return f"{catalog}.{schema}.{name}"


# COMMAND ----------

model = mlflow.pyfunc.load_model(f"models:/{model_name}/{version}")
pipeline = model.unwrap_python_model().pipeline
prep = pipeline.named_steps["prep"]

train = spark.table(table(c.SILVER_TREINO)).where("split = 'train'").toPandas()
atual = spark.table(table(c.SILVER_SAFRA)).where(F.col("safra") == safra).toPandas()

detalhe, resumo = monitoring.monitor(
    train_features=prep.transform(train[c.MODEL_INPUT_COLUMNS]),
    y_train=train[c.TARGET],
    safra_features=prep.transform(atual[c.MODEL_INPUT_COLUMNS]),
    train_scores=pipeline.predict_proba(train[c.MODEL_INPUT_COLUMNS])[:, 1],
    safra_scores=pipeline.predict_proba(atual[c.MODEL_INPUT_COLUMNS])[:, 1],
)
display(resumo)

# COMMAND ----------


def write_gold(pdf, name: str, comment: str) -> None:
    sdf = (
        spark.createDataFrame(pdf)
        .withColumn("safra", F.lit(safra))
        .withColumn("model_version", F.lit(version))
        .withColumn("monitored_at", F.current_timestamp())
    )
    writer = sdf.write.format("delta").mode("overwrite")
    if spark.catalog.tableExists(table(name)):
        writer = writer.option("replaceWhere", f"safra = '{safra}' AND model_version = '{version}'")
    writer.saveAsTable(table(name))
    spark.sql(f"COMMENT ON TABLE {table(name)} IS '{comment}'")


write_gold(detalhe, c.GOLD_PSI, "PSI por faixa (treino = esperado, safra = atual) de cada feature e do score")
write_gold(resumo, c.GOLD_PSI_RESUMO, "PSI total e status de estabilidade por variável, safra e versão do modelo")

alertas = resumo[resumo["status"] != "estável"]
if len(alertas):
    print(f"⚠️ {len(alertas)} variável(is) fora da faixa estável na safra {safra}:")
    display(alertas)
else:
    print(f"safra {safra}: todas as variáveis e o score estão estáveis (PSI < {monitoring.PSI_ESTAVEL})")

# COMMAND ----------

dbutils.notebook.exit(json.dumps({
    "safra": safra, "model_version": version,
    "psi_score": float(resumo.loc[resumo["variavel"] == "score_modelo", "psi"].iloc[0]),
    "n_alertas": int(len(alertas)),
}))
