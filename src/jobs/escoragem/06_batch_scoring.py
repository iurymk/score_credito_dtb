# Databricks notebook source
# MAGIC %md
# MAGIC # Esteira de escoragem · 6 · Escoragem da safra (gold)
# MAGIC Pontua a safra de produção com a versão **`@champion`** do Unity Catalog → `gold_credit_scores`
# MAGIC (probabilidade, score 0–1000, decisão e motivos por cliente).
# MAGIC
# MAGIC Esta esteira **não treina nada**: ela só consome o modelo promovido pela esteira de treino. As linhas são
# MAGIC substituídas por safra e versão do modelo, então rodar de novo é idempotente.

# COMMAND ----------

import json, os, sys

sys.path.insert(0, os.path.abspath("../.."))

import mlflow
from mlflow.tracking import MlflowClient
from pyspark.sql import functions as F

from credit_scoring import config as c

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
model_name = dbutils.widgets.get("model_name")
safra = dbutils.widgets.get("safra")
job_run_id = dbutils.widgets.get("job_run_id")

mlflow.set_registry_uri("databricks-uc")
client = MlflowClient(registry_uri="databricks-uc")
gold_table = f"{catalog}.{schema}.{c.GOLD_SCORES}"

# O alias é resolvido uma vez para um número de versão fixo: as duas tarefas desta esteira usam o mesmo modelo,
# mesmo que outra execução mova o @champion no meio do caminho.
version = client.get_model_version_by_alias(model_name, "champion").version
print(f"escorando a safra {safra} com {model_name} v{version} (@champion)")

# COMMAND ----------

carteira = spark.table(f"{catalog}.{schema}.{c.SILVER_SAFRA}").where(F.col("safra") == safra).toPandas()
assert len(carteira), f"nenhum cliente na safra '{safra}'"

model = mlflow.pyfunc.load_model(f"models:/{model_name}/{version}")
preds = model.predict(carteira[c.MODEL_INPUT_COLUMNS])
scored = carteira[[c.ID_COL, "safra"]].reset_index(drop=True).join(preds)

gold = (
    spark.createDataFrame(scored)
    .withColumn("model_version", F.lit(version))
    .withColumn("job_run_id", F.lit(job_run_id))
    .withColumn("scored_at", F.current_timestamp())
)

writer = gold.write.format("delta").mode("overwrite")
if spark.catalog.tableExists(gold_table):
    writer = writer.option("replaceWhere", f"safra = '{safra}' AND model_version = '{version}'")
writer.saveAsTable(gold_table)
spark.sql(f"COMMENT ON TABLE {gold_table} IS 'Scores de crédito por cliente, safra e versão do modelo'")

# COMMAND ----------

resumo = (
    spark.table(gold_table)
    .where((F.col("safra") == safra) & (F.col("model_version") == version))
    .groupBy("decisao")
    .agg(F.count("*").alias("clientes"), F.round(F.avg("score"), 0).alias("score_medio"))
    .withColumn("pct_clientes", F.round(F.col("clientes") / len(scored), 4))
)
display(resumo)

# COMMAND ----------

dbutils.jobs.taskValues.set("model_version", version)
dbutils.notebook.exit(json.dumps({"safra": safra, "model_version": version, "rows": len(scored)}))
