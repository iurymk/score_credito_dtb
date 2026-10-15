# Databricks notebook source
# MAGIC %md
# MAGIC # 2 · Qualidade e splits (silver)
# MAGIC **bronze** → validações de qualidade → **silver** com os dados limpos e a coluna `split` (train / val / test).
# MAGIC
# MAGIC - Falha rápido se o dado de origem mudou (colunas, taxa de inadimplência, nulos, negativos), antes de treinar.
# MAGIC - Aqui só entram regras de linha (remover `age = 0`). Imputação, winsorização e features dependem de
# MAGIC   estatísticas, então ficam no `CreditPreprocessor`, que aprende só com o treino (sem leakage).
# MAGIC - Splits estratificados e determinísticos: teste 20% (como na exploração) e validação para o threshold.

# COMMAND ----------

import json, os, sys

sys.path.insert(0, os.path.abspath(".."))

from pyspark.sql import functions as F

from credit_scoring import config as c, data

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

bronze_table = f"{catalog}.{schema}.bronze_credit"
silver_table = f"{catalog}.{schema}.silver_credit"

# COMMAND ----------

raw = spark.table(bronze_table).drop("_source_file", "_ingested_at").toPandas()  # 150k linhas: cabe na memória
stats = data.check_quality(raw)  # lança exceção → a tarefa falha → nada depois dela roda
print(json.dumps(stats, indent=2))

clean = data.clean(raw)
clean["split"] = data.assign_splits(clean)
stats["n_removed"] = len(raw) - len(clean)
stats["test_digest"] = data.split_digest(clean, "test")

# COMMAND ----------

silver = spark.createDataFrame(clean)
for col in c.NULLABLE_COLUMNS:  # NaN do pandas → NULL de verdade no Delta
    silver = silver.withColumn(col, F.when(F.isnan(col), None).otherwise(F.col(col)))

(
    silver.withColumn("_prepared_at", F.current_timestamp())
    .write.mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(silver_table)
)
spark.sql(f"COMMENT ON TABLE {silver_table} IS 'Dados validados (age > 0) com split estratificado train/val/test'")

display(
    spark.table(silver_table)
    .groupBy("split")
    .agg(F.count("*").alias("clientes"), F.round(F.avg(c.TARGET), 4).alias("taxa_inadimplencia"))
    .orderBy("split")
)

# COMMAND ----------

dbutils.jobs.taskValues.set("test_digest", stats["test_digest"])
dbutils.notebook.exit(json.dumps(stats))
