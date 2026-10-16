# Databricks notebook source
# MAGIC %md
# MAGIC # Esteira de dados · 2 · Qualidade e splits (silver)
# MAGIC **bronze** → validações de qualidade → **silver**:
# MAGIC
# MAGIC - `silver_credit_treino`: dados limpos + coluna `split` (train / val / test), estratificada e determinística.
# MAGIC - `silver_credit_safra`: safra de produção validada (sem target), pronta para a escoragem.
# MAGIC
# MAGIC Falha rápido se o dado de origem mudou (colunas, taxa de inadimplência, nulos, negativos), antes de treinar ou
# MAGIC escorar. Aqui só entram regras de linha (remover `age = 0` do treino). Imputação, winsorização e features
# MAGIC dependem de estatísticas, então ficam no `CreditPreprocessor`, que aprende só com o treino (sem leakage).

# COMMAND ----------

import json, os, sys

sys.path.insert(0, os.path.abspath("../.."))

from pyspark.sql import functions as F

from credit_scoring import config as c, data

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")


def table(name: str) -> str:
    return f"{catalog}.{schema}.{name}"


def write_silver(pdf, name: str, comment: str) -> None:
    sdf = spark.createDataFrame(pdf)
    for col in c.NULLABLE_COLUMNS:  # NaN do pandas → NULL de verdade no Delta
        sdf = sdf.withColumn(col, F.when(F.isnan(col), None).otherwise(F.col(col)))
    (
        sdf.withColumn("_prepared_at", F.current_timestamp())
        .write.mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(table(name))
    )
    spark.sql(f"COMMENT ON TABLE {table(name)} IS '{comment}'")


# COMMAND ----------
# MAGIC %md ## Treino: qualidade → limpeza → splits

# COMMAND ----------

raw = spark.table(table(c.BRONZE_TREINO)).drop("_source_file", "_ingested_at").toPandas()  # 150k linhas
stats = {"treino": data.check_quality(raw)}  # lança exceção → a tarefa falha → nada depois dela roda

treino = data.clean(raw)
treino["split"] = data.assign_splits(treino)
stats["treino"]["n_removed"] = len(raw) - len(treino)
stats["treino"]["test_digest"] = data.split_digest(treino, "test")
write_silver(treino, c.SILVER_TREINO, "Dados de desenvolvimento validados (age > 0) com split estratificado train/val/test")

display(
    spark.table(table(c.SILVER_TREINO))
    .groupBy("split")
    .agg(F.count("*").alias("clientes"), F.round(F.avg(c.TARGET), 4).alias("taxa_inadimplencia"))
    .orderBy("split")
)

# COMMAND ----------
# MAGIC %md ## Safra de produção: qualidade (sem target)

# COMMAND ----------

safra = spark.table(table(c.BRONZE_SAFRA)).drop("_source_file", "_ingested_at").toPandas()
stats["safra"] = data.check_quality(safra, has_target=False)
write_silver(safra, c.SILVER_SAFRA, "Safra de produção validada, pronta para escoragem (sem target)")
print(json.dumps(stats, indent=2))

# COMMAND ----------

dbutils.notebook.exit(json.dumps(stats))
