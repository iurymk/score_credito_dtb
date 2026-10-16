# Databricks notebook source
# MAGIC %md
# MAGIC # Esteira de dados · 1 · Ingestão (bronze)
# MAGIC Arquivos brutos do UC Volume → tabelas Delta **bronze**, sem nenhuma transformação, com metadados de linhagem
# MAGIC (`_source_file`, `_ingested_at`):
# MAGIC
# MAGIC | Arquivo | Tabela | Papel |
# MAGIC |---|---|---|
# MAGIC | `cs-training.csv` | `bronze_credit_treino` | Desenvolvimento do modelo (com target) |
# MAGIC | `cs-test.csv` | `bronze_credit_safra` | Safra de produção a ser escorada e monitorada (sem target) |
# MAGIC
# MAGIC Os CSVs não vão para o Git (regras do Kaggle): eles são enviados uma vez para o Volume — veja o README.

# COMMAND ----------

import json, os, sys

sys.path.insert(0, os.path.abspath("../.."))  # src/ do bundle → import credit_scoring

from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType, StructField, StructType

from credit_scoring import config as c

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
volume = dbutils.widgets.get("volume")
safra = dbutils.widgets.get("safra")

volume_root = f"/Volumes/{catalog}/{schema}/{volume}"
SOURCES = [
    (c.TRAINING_FILE, c.BRONZE_TREINO, "treino"),
    (c.SAFRA_FILE, c.BRONZE_SAFRA, "safra"),
]

missing = [f for f, _, _ in SOURCES if not os.path.exists(f"{volume_root}/{f}")]
if missing:
    raise FileNotFoundError(
        f"arquivos ausentes em {volume_root}: {missing}. Envie-os para o Volume antes de rodar o job:\n"
        + "\n".join(f"  databricks fs cp data/raw/{f} dbfs:{volume_root}/" for f in missing)
    )

# COMMAND ----------
# MAGIC %md ## Leitura com schema explícito
# MAGIC Schema fixo em vez de `inferSchema`: se o arquivo de origem mudar de formato, a leitura falha aqui.

# COMMAND ----------

INT_COLUMNS = {c.ID_COL, c.TARGET, c.AGE, c.LATE_30_59, c.OPEN_CREDIT, c.LATE_90, c.REAL_ESTATE, c.LATE_60_89}
csv_schema = StructType(
    [StructField(col, IntegerType() if col in INT_COLUMNS else DoubleType()) for col in [c.ID_COL, c.TARGET, *c.RAW_COLUMNS]]
)

counts = {}
for file_name, table_name, role in SOURCES:
    table = f"{catalog}.{schema}.{table_name}"
    bronze = (
        spark.read.csv(f"{volume_root}/{file_name}", header=True, schema=csv_schema, nullValue="NA", mode="FAILFAST")
        .withColumn("_source_file", F.col("_metadata.file_path"))
        .withColumn("_ingested_at", F.current_timestamp())
    )
    if role == "safra":
        bronze = bronze.drop(c.TARGET).withColumn("safra", F.lit(safra))  # desfecho ainda desconhecido
    bronze.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(table)
    spark.sql(f"COMMENT ON TABLE {table} IS 'Give Me Some Credit (Kaggle) — dados brutos do {file_name} ({role})'")
    counts[table_name] = spark.table(table).count()
    print(f"{table}: {counts[table_name]:,} linhas")

display(spark.table(f"{catalog}.{schema}.{c.BRONZE_SAFRA}").limit(5))

# COMMAND ----------

dbutils.notebook.exit(json.dumps(counts))
