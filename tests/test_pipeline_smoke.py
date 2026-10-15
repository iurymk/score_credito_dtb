"""Smoke test local ponta a ponta com o dataset real: mesmo código que as tarefas do job executam.

Requer data/raw/cs-training.csv (veja o README). Caso contrário, é pulado.
"""

from credit_scoring import config as c, data, evaluation, modeling

# Mesmos portões padrão do databricks.yml
GATES = evaluation.Gates(min_auc=0.85, min_ks=0.40, max_fairness_gap=0.05)


def test_real_data_passes_quality_and_gates(real_df):
    data.check_quality(real_df)
    df = data.clean(real_df)
    df["split"] = data.assign_splits(df)
    train, val, test = (df[df["split"] == s] for s in c.SPLITS)

    pipe = modeling.build_pipeline(modeling.CHOSEN_MODEL, train[c.TARGET]).fit(train[c.RAW_COLUMNS], train[c.TARGET])
    threshold = evaluation.best_f1_threshold(val[c.TARGET], pipe.predict_proba(val[c.RAW_COLUMNS])[:, 1])

    proba = pipe.predict_proba(test[c.RAW_COLUMNS])[:, 1]
    metrics = evaluation.classification_metrics(test[c.TARGET], proba, threshold)
    gaps = evaluation.fairness_gaps(evaluation.segment_auc(test, test[c.TARGET], proba))

    assert evaluation.gate_failures(metrics, gaps, GATES) == [], (metrics, gaps)
