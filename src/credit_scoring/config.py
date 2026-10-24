"""Constantes compartilhadas por todas as etapas do pipeline (decisões da fase de experimentação)."""

SEED = 42

# Give Me Some Credit (Kaggle): a primeira coluna do CSV é um índice sem nome → vira ``id``.
ID_COL = "id"
TARGET = "SeriousDlqin2yrs"
TRAINING_FILE = "cs-training.csv"  # 150 mil clientes com target: desenvolvimento do modelo
SAFRA_FILE = "cs-test.csv"  # 101 mil clientes sem target: a "safra de produção" escorada e monitorada

# Tabelas por camada da arquitetura medalhão (nomes dentro de <catalog>.<schema>)
BRONZE_TREINO, BRONZE_SAFRA = "bronze_credit_treino", "bronze_credit_safra"
SILVER_TREINO, SILVER_SAFRA = "silver_credit_treino", "silver_credit_safra"
GOLD_SCORES = "gold_credit_scores"
GOLD_BIVARIADAS = "gold_bivariadas"
GOLD_PSI, GOLD_PSI_RESUMO = "gold_monitoramento_psi", "gold_monitoramento_resumo"
# Desempenho de cada versão avaliada no teste (alimentam a página "Desempenho do modelo" do painel)
GOLD_DESEMPENHO = "gold_desempenho_modelo"
GOLD_DESEMPENHO_FAIXAS, GOLD_DESEMPENHO_CORTES = "gold_desempenho_faixas", "gold_desempenho_cortes"

REVOLVING = "RevolvingUtilizationOfUnsecuredLines"
AGE = "age"
LATE_30_59 = "NumberOfTime30-59DaysPastDueNotWorse"
DEBT_RATIO = "DebtRatio"
INCOME = "MonthlyIncome"
OPEN_CREDIT = "NumberOfOpenCreditLinesAndLoans"
LATE_90 = "NumberOfTimes90DaysLate"
REAL_ESTATE = "NumberRealEstateLoansOrLines"
LATE_60_89 = "NumberOfTime60-89DaysPastDueNotWorse"
DEPENDENTS = "NumberOfDependents"

# Colunas brutas do dataset (ordem do CSV), sem id e target.
RAW_COLUMNS = [
    REVOLVING, AGE, LATE_30_59, DEBT_RATIO, INCOME, OPEN_CREDIT, LATE_90, REAL_ESTATE, LATE_60_89, DEPENDENTS,
]
# Colunas brutas que o modelo consome: é o contrato de entrada do endpoint.
# NumberRealEstateLoansOrLines foi descartada na exploração (IV 0.012).
MODEL_INPUT_COLUMNS = [c for c in RAW_COLUMNS if c != REAL_ESTATE]
NULLABLE_COLUMNS = [INCOME, DEPENDENTS]
DELINQUENCY_COLUMNS = [LATE_30_59, LATE_60_89, LATE_90]

# As 15 features selecionadas no notebook 02_feature_engineering, na ordem do modelo.
FEATURES = [
    REVOLVING,
    LATE_90,
    LATE_30_59,
    LATE_60_89,
    AGE,
    INCOME,
    OPEN_CREDIT,
    DEBT_RATIO,
    DEPENDENTS,
    "flag_missing_income",
    "renda_per_capita",
    "teve_atraso_90dias",
    "teve_qualquer_atraso",
    "total_atrasos",
    "score_risco",
]

# No Give Me Some Credit, 96 e 98 nas colunas de atraso são códigos especiais, não contagens reais.
DELINQUENCY_SENTINELS = (96, 98)

# Splits estratificados: teste = 20% (igual à exploração); validação = 20% do restante,
# usada só para escolher o threshold, para que o teste fique limpo para a avaliação final.
TEST_FRACTION = 0.20
VAL_FRACTION = 0.20
SPLITS = ("train", "val", "test")

# Hiperparâmetros do XGBoost escolhido no notebook 03_modeling.
XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "auc",
    "random_state": SEED,
    "n_jobs": -1,
}

APPROVE, REJECT = "APROVAR", "REPROVAR"

# Premissas de negócio da simulação de impacto financeiro (notebook 04_evaluation).
TICKET_MEDIO = 15_000.0
TAXA_PERDA_INADIMPLENCIA = 0.60
RECEITA_POR_CLIENTE = 800.0
