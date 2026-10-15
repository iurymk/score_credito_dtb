# Databricks notebook source
# MAGIC %md
# MAGIC # 1 · Ingestão (bronze)
# MAGIC `cs-training.csv` no UC Volume → tabela Delta **bronze** com os dados brutos, sem nenhuma transformação,
# MAGIC mais metadados de linhagem (`_source_file`, `_ingested_at`).
# MAGIC
# MAGIC O CSV não vai para o Git (regras do Kaggle): ele é enviado uma vez para o Volume — veja o README.

# COMMAND ----------

import json, os, sys

sys.path.insert(0, os.path.abspath(".."))  # src/ do bundle → import credit_scoring

from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType, StructField, StructType

from credit_scoring import config as c

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
volume = dbutils.widgets.get("volume")

source_path = f"/Volumes/{catalog}/{schema}/{volume}/{c.TRAINING_FILE}"
bronze_table = f"{catalog}.{schema}.bronze_credit"

if not os.path.exists(source_path):
    raise FileNotFoundError(
        f"{source_path} não encontrado. Envie o arquivo para o Volume antes de rodar o job:\n"
        f"  databricks fs cp data/raw/{c.TRAINING_FILE} dbfs:/Volumes/{catalog}/{schema}/{volume}/"
    )

# COMMAND ----------
# MAGIC %md ## Leitura com schema explícito
# MAGIC Schema fixo em vez de `inferSchema`: se o arquivo de origem mudar de formato, a leitura falha aqui.

# COMMAND ----------

INT_COLUMNS = {c.ID_COL, c.TARGET, c.AGE, c.LATE_30_59, c.OPEN_CREDIT, c.LATE_90, c.REAL_ESTATE, c.LATE_60_89}
csv_schema = StructType(
    [StructField(col, IntegerType() if col in INT_COLUMNS else DoubleType()) for col in [c.ID_COL, c.TARGET, *c.RAW_COLUMNS]]
)

bronze = (
    spark.read.csv(source_path, header=True, schema=csv_schema, nullValue="NA", mode="FAILFAST")
    .withColumn("_source_file", F.col("_metadata.file_path"))
    .withColumn("_ingested_at", F.current_timestamp())
)

(
    bronze.write.mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(bronze_table)
)
spark.sql(f"COMMENT ON TABLE {bronze_table} IS 'Give Me Some Credit (Kaggle) — dados brutos do {c.TRAINING_FILE}'")

n_rows = spark.table(bronze_table).count()
print(f"{bronze_table}: {n_rows:,} linhas")
display(spark.table(bronze_table).limit(10))

# COMMAND ----------

dbutils.jobs.taskValues.set("n_rows", n_rows)
dbutils.notebook.exit(json.dumps({"bronze_table": bronze_table, "n_rows": n_rows}))
