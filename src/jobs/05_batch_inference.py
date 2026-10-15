# Databricks notebook source
# MAGIC %md
# MAGIC # 5a · Inferência em lote (gold)
# MAGIC Pontua a carteira com a versão `@champion` → tabela **gold** `gold_credit_scores`
# MAGIC (probabilidade, score, decisão e motivos por cliente). As linhas são substituídas por versão do modelo,
# MAGIC então rodar de novo é idempotente.
# MAGIC
# MAGIC Por padrão pontua o split de teste (clientes que o modelo nunca viu), o que também permite comparar a decisão
# MAGIC com o desfecho real.

# COMMAND ----------

import json, os, sys

sys.path.insert(0, os.path.abspath(".."))

import mlflow
from pyspark.sql import functions as F

from credit_scoring import config as c

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
model_name = dbutils.widgets.get("model_name")
version = dbutils.widgets.get("model_version")
score_split = dbutils.widgets.get("score_split")
job_run_id = dbutils.widgets.get("job_run_id")

mlflow.set_registry_uri("databricks-uc")
gold_table = f"{catalog}.{schema}.gold_credit_scores"
model_uri = f"models:/{model_name}/{version}"  # versão fixa (== @champion), não o alias: sem corrida com outro run

# COMMAND ----------

carteira = spark.table(f"{catalog}.{schema}.silver_credit").where(F.col("split") == score_split).toPandas()
assert len(carteira), f"nenhum cliente no split '{score_split}'"

model = mlflow.pyfunc.load_model(model_uri)
preds = model.predict(carteira[c.MODEL_INPUT_COLUMNS])

scored = carteira[[c.ID_COL, c.TARGET]].rename(columns={c.TARGET: "inadimplente_real"}).reset_index(drop=True)
scored = scored.join(preds)

gold = (
    spark.createDataFrame(scored)
    .withColumn("model_version", F.lit(version))
    .withColumn("job_run_id", F.lit(job_run_id))
    .withColumn("scored_at", F.current_timestamp())
)

# COMMAND ----------

writer = gold.write.format("delta").mode("overwrite")
if spark.catalog.tableExists(gold_table):
    writer = writer.option("replaceWhere", f"model_version = '{version}'")
writer.saveAsTable(gold_table)
spark.sql(f"COMMENT ON TABLE {gold_table} IS 'Scores de crédito por cliente e versão do modelo (probabilidade, score, decisão, motivos)'")

resumo = (
    spark.table(gold_table)
    .where(F.col("model_version") == version)
    .groupBy("decisao")
    .agg(
        F.count("*").alias("clientes"),
        F.round(F.avg("inadimplente_real"), 4).alias("taxa_inadimplencia_real"),
        F.round(F.avg("score"), 0).alias("score_medio"),
    )
    .orderBy("decisao")
)
display(resumo)
display(spark.table(gold_table).where(F.col("model_version") == version).limit(20))

# COMMAND ----------

n = int(gold.count())
dbutils.notebook.exit(json.dumps({"rows": n, "model_version": version, "gold_table": gold_table}))
