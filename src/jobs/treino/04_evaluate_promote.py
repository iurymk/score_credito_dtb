# Databricks notebook source
# MAGIC %md
# MAGIC # Esteira de treino · 4 · Avaliação no teste e promoção para `@champion`
# MAGIC Avalia a **versão registrada** no Unity Catalog (o artefato exato que vai para produção) no conjunto de teste,
# MAGIC que não foi usado nem no treino nem na escolha do threshold.
# MAGIC
# MAGIC Loga no run de treino da versão: métricas (KS, AUC, PR-AUC, lift), tabelas de faixas e de cortes, ROC, matriz de
# MAGIC confusão, SHAP, AUC por segmento (fairness) e impacto financeiro. Grava a análise bivariada (optimal binning) das
# MAGIC features em `gold_bivariadas`, que alimenta o painel. Só promove para `@champion` se passar nos
# MAGIC **portões de qualidade**; senão a tarefa falha e nada é publicado (o endpoint não é atualizado).

# COMMAND ----------

import json, os, sys

sys.path.insert(0, os.path.abspath("../.."))

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import shap
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from pyspark.sql import functions as F

from credit_scoring import binning, config as c, data, evaluation

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
model_name = dbutils.widgets.get("model_name")
version = dbutils.widgets.get("model_version")
experiment_path = dbutils.widgets.get("experiment_path")
gates = evaluation.Gates(
    min_auc=float(dbutils.widgets.get("min_auc")),
    min_ks=float(dbutils.widgets.get("min_ks")),
    max_fairness_gap=float(dbutils.widgets.get("max_fairness_gap")),
)
CHAMPION_ALIAS = "champion"
CHAMPION_AUC_TOLERANCE = 0.005  # não promove uma versão pior que o champion atual

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(experiment_path)
client = MlflowClient(registry_uri="databricks-uc")

# COMMAND ----------
# MAGIC %md ## Predições da versão registrada no teste

# COMMAND ----------

silver = spark.table(f"{catalog}.{schema}.{c.SILVER_TREINO}").toPandas()
train, test = silver[silver["split"] == "train"], silver[silver["split"] == "test"]
model = mlflow.pyfunc.load_model(f"models:/{model_name}/{version}")
wrapper = model.unwrap_python_model()  # acesso ao pipeline sklearn e ao threshold registrados
threshold = wrapper.threshold

preds = model.predict(test[c.MODEL_INPUT_COLUMNS])  # mesmo caminho do endpoint
y_true, y_proba = test[c.TARGET], preds["prob_inadimplencia"]

metrics = evaluation.classification_metrics(y_true, y_proba, threshold)
segments = evaluation.segment_auc(test, y_true, y_proba)
gaps = evaluation.fairness_gaps(segments)
impacto = evaluation.financial_impact(metrics["fn"], metrics["fp"])

faixas = evaluation.tabela_faixas(y_true, y_proba)
cortes = evaluation.tabela_cortes(y_true, y_proba)

print(json.dumps({"threshold": threshold, **metrics, **gaps, **impacto}, indent=2))
display(segments)
display(faixas)

# COMMAND ----------
# MAGIC %md ## Explicabilidade (SHAP)

# COMMAND ----------

sample = test.sample(n=min(2000, len(test)), random_state=c.SEED)
features = wrapper.pipeline.named_steps["prep"].transform(sample[c.MODEL_INPUT_COLUMNS])
shap_values = shap.TreeExplainer(wrapper.pipeline.named_steps["model"]).shap_values(features)

shap.summary_plot(shap_values, features, plot_type="bar", show=False)
fig_shap_bar = plt.gcf()
plt.title("Importância global das features (SHAP)", fontweight="bold")
plt.tight_layout()
plt.show()

shap.summary_plot(shap_values, features, show=False)
fig_shap_summary = plt.gcf()
plt.title("Direção do impacto das features (SHAP)", fontweight="bold")
plt.tight_layout()
plt.show()

top_features = (
    pd.Series(np.abs(shap_values).mean(axis=0), index=c.FEATURES).sort_values(ascending=False).head(5).index.tolist()
)

# COMMAND ----------
# MAGIC %md ## Análise bivariada das features (optimal binning no treino) → `gold_bivariadas`
# MAGIC Para cada feature do modelo: faixas, volume, taxa de inadimplência, odds, WoE, IV e lift — a mesma análise
# MAGIC da seleção de variáveis, agora versionada junto com o modelo e exposta no painel.

# COMMAND ----------

train_features = wrapper.pipeline.named_steps["prep"].transform(train[c.MODEL_INPUT_COLUMNS])
bivariadas, iv_ranking = binning.analise_binning(
    train_features.assign(**{c.TARGET: train[c.TARGET].to_numpy()}), c.FEATURES, c.TARGET
)
display(iv_ranking)

bivariadas_table = f"{catalog}.{schema}.{c.GOLD_BIVARIADAS}"
gold_biv = (
    spark.createDataFrame(pd.concat(bivariadas.values(), ignore_index=True))
    .withColumn("model_version", F.lit(version))
    .withColumn("_created_at", F.current_timestamp())
)
writer = gold_biv.write.format("delta").mode("overwrite")
if spark.catalog.tableExists(bivariadas_table):
    writer = writer.option("replaceWhere", f"model_version = '{version}'")
writer.saveAsTable(bivariadas_table)

fig_lift_top = binning.plot_lift(bivariadas[iv_ranking["variavel"].iloc[0]])
plt.show()

# COMMAND ----------
# MAGIC %md ## Log no run de treino da versão (um run = a história completa do modelo)

# COMMAND ----------

mv = client.get_model_version(model_name, version)
with mlflow.start_run(run_id=mv.run_id):
    mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items()} | {f"test_{k}": v for k, v in gaps.items()} | impacto)
    mlflow.log_figure(evaluation.roc_figure(y_true, y_proba, f"ROC — teste (v{version})"), "teste/roc.png")
    mlflow.log_figure(evaluation.confusion_figure(metrics, threshold), "teste/matriz_confusao.png")
    mlflow.log_figure(fig_shap_bar, "teste/shap_importancia.png")
    mlflow.log_figure(fig_shap_summary, "teste/shap_summary.png")
    mlflow.log_table(segments, "teste/auc_por_segmento.json")
    mlflow.log_table(faixas, "teste/tabela_faixas.json")
    mlflow.log_table(cortes, "teste/tabela_cortes.json")
    mlflow.log_table(iv_ranking, "bivariadas/iv_ranking.json")
    mlflow.log_figure(fig_lift_top, "bivariadas/lift_top_feature.png")

# COMMAND ----------
# MAGIC %md ## Portões de qualidade → `@champion`

# COMMAND ----------

failures = evaluation.gate_failures(metrics, gaps, gates)

try:
    champion = client.get_model_version_by_alias(model_name, CHAMPION_ALIAS)
except MlflowException:
    champion = None
if champion is not None and champion.version != version and "test_auc" in champion.tags:
    champion_auc = float(champion.tags["test_auc"])
    if metrics["auc"] < champion_auc - CHAMPION_AUC_TOLERANCE:
        failures.append(f"AUC {metrics['auc']:.4f} pior que o champion atual v{champion.version} ({champion_auc:.4f})")

tags = {
    "test_auc": f"{metrics['auc']:.6f}",
    "test_ks": f"{metrics['ks']:.6f}",
    "test_recall": f"{metrics['recall']:.6f}",
    "test_precision": f"{metrics['precision']:.6f}",
    "test_pr_auc": f"{metrics['pr_auc']:.6f}",
    "test_digest": data.split_digest(silver, "test"),
    "gate_status": "failed" if failures else "passed",
}
for k, v in tags.items():
    client.set_model_version_tag(model_name, version, k, v)

if failures:
    raise RuntimeError(f"v{version} reprovada nos portões de qualidade — não promovida:\n  - " + "\n  - ".join(failures))

model_card = f"""Credit scoring (Give Me Some Credit) — XGBoost com pré-processamento embutido.

Entrada: {len(c.MODEL_INPUT_COLUMNS)} colunas brutas do cliente ({", ".join(c.MODEL_INPUT_COLUMNS)}); nulos aceitos em MonthlyIncome e NumberOfDependents.
Saída: prob_inadimplencia, score (0-1000, maior = menor risco), decisao (APROVAR/REPROVAR), motivos (top-3 features que aumentaram o risco).

Threshold: {threshold:.4f} ({mv.tags.get('threshold_criterio', 'regra de negócio')}).
Teste (n={len(test):,}): AUC {metrics['auc']:.4f} | KS {metrics['ks']:.4f} | Gini {metrics['gini']:.4f} | PR-AUC {metrics['pr_auc']:.4f} | Lift top 10% {metrics['lift_top10']:.2f}x | Recall {metrics['recall']:.4f} | Precision {metrics['precision']:.4f}.
Fairness: variação de AUC por faixa etária {gaps.get('auc_gap_faixa_etaria', 0):.4f}, por faixa de renda {gaps.get('auc_gap_faixa_renda', 0):.4f}.
Features mais importantes (SHAP): {", ".join(top_features)}.
Portões: AUC >= {gates.min_auc}, KS >= {gates.min_ks}, variação de AUC por segmento <= {gates.max_fairness_gap}.
Limitações: dataset público e histórico (Kaggle); premissas financeiras ilustrativas; sem monitoramento de drift automatizado."""
client.update_model_version(model_name, version, description=model_card)
client.set_registered_model_alias(model_name, CHAMPION_ALIAS, version)
print(f"@{CHAMPION_ALIAS} → {model_name} v{version}")

# COMMAND ----------

dbutils.jobs.taskValues.set("champion_version", version)
dbutils.notebook.exit(json.dumps({"champion_version": version, "test_auc": metrics["auc"], "test_ks": metrics["ks"]}))
