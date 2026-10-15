# Databricks notebook source
# MAGIC %md
# MAGIC # 5b · Endpoint em tempo real (CPU)
# MAGIC Cria ou atualiza um endpoint de Model Serving para a versão `@champion` — sem downtime na atualização (a versão
# MAGIC antiga responde até a nova ficar pronta). A inference table grava toda requisição/resposta em Delta.
# MAGIC Espera os **dois** sinais de prontidão e faz uma consulta de teste antes de declarar sucesso.

# COMMAND ----------

import json, math, os, sys, time

sys.path.insert(0, os.path.abspath(".."))

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound
from databricks.sdk.service.serving import (
    AiGatewayConfig,
    AiGatewayInferenceTableConfig,
    EndpointCoreConfigInput,
    EndpointTag,
    ServedEntityInput,
    ServingModelWorkloadType,
)

from credit_scoring import config as c

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
model_name = dbutils.widgets.get("model_name")
version = dbutils.widgets.get("model_version")
endpoint_name = dbutils.widgets.get("endpoint_name")
workload_size = dbutils.widgets.get("workload_size")
timeout_min = int(dbutils.widgets.get("deploy_timeout_minutes"))
enable_inference_table = dbutils.widgets.get("enable_inference_table").lower() == "true"

w = WorkspaceClient()
served = ServedEntityInput(
    entity_name=model_name,
    entity_version=version,
    workload_size=workload_size,
    workload_type=ServingModelWorkloadType.CPU,
    scale_to_zero_enabled=True,
)

# COMMAND ----------

try:
    current = w.serving_endpoints.get(endpoint_name)
except NotFound:
    current = None

if current is None:
    print(f"criando endpoint {endpoint_name} → {model_name} v{version}")
    # Inference table: toda requisição/resposta vai para <catalog>.<schema>.credit_endpoint_payload.
    # Desligue com enable_inference_table=false se o workspace não suportar (ex.: algumas edições gratuitas).
    ai_gateway = AiGatewayConfig(
        inference_table_config=AiGatewayInferenceTableConfig(
            catalog_name=catalog, schema_name=schema, table_name_prefix="credit_endpoint", enabled=True
        )
    ) if enable_inference_table else None
    w.serving_endpoints.create(
        name=endpoint_name,
        config=EndpointCoreConfigInput(served_entities=[served]),
        ai_gateway=ai_gateway,
        tags=[EndpointTag(key="project", value="credit-scoring-mlops")],
    )
else:
    live = {(e.entity_name, e.entity_version) for e in (current.config.served_entities or [])} if current.config else set()
    if (model_name, version) in live:
        print(f"endpoint já serve a v{version}; nada a atualizar")
    else:
        print(f"atualizando endpoint {endpoint_name}: {live} → v{version}")
        w.serving_endpoints.update_config(endpoint_name, served_entities=[served])

# COMMAND ----------
# MAGIC %md ## Espera ficar pronto (`state.ready == READY` **e** `state.config_update == NOT_UPDATING`)

# COMMAND ----------

deadline = time.time() + timeout_min * 60
while True:
    st = w.serving_endpoints.get(endpoint_name).state
    ready, upd = st.ready.value if st.ready else None, st.config_update.value if st.config_update else None
    print(time.strftime("%H:%M:%S"), ready, upd)
    if upd == "UPDATE_FAILED":
        raise RuntimeError(f"falha na atualização do endpoint — veja os build logs de {endpoint_name}")
    if ready == "READY" and upd == "NOT_UPDATING":
        break
    if time.time() > deadline:
        raise TimeoutError(f"{endpoint_name} não ficou pronto em {timeout_min} min (ready={ready}, config_update={upd})")
    time.sleep(30)

entities = w.serving_endpoints.get(endpoint_name).config.served_entities
assert any(e.entity_version == version for e in entities), f"o endpoint não está servindo a v{version}: {entities}"

# COMMAND ----------
# MAGIC %md ## Consulta de teste com um cliente real do conjunto de teste

# COMMAND ----------

row = spark.table(f"{catalog}.{schema}.silver_credit").where("split = 'test'").limit(1).toPandas().iloc[0]
record = {col: (None if row[col] is None or (isinstance(row[col], float) and math.isnan(row[col])) else float(row[col]))
          for col in c.MODEL_INPUT_COLUMNS}
resp = w.serving_endpoints.query(endpoint_name, dataframe_records=[record])
pred = resp.predictions[0]
print(json.dumps({"entrada": record, "saida": pred}, indent=2, ensure_ascii=False))
assert pred["decisao"] in (c.APPROVE, c.REJECT) and 0 <= pred["prob_inadimplencia"] <= 1

dbutils.notebook.exit(json.dumps({"endpoint": endpoint_name, "model_version": version, "smoke_decisao": pred["decisao"]}))
