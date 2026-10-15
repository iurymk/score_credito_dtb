"""Envia um cliente para o endpoint de credit scoring e imprime a resposta + a latência de ida e volta.

    python scripts/query_endpoint.py --profile <profile> --endpoint credit-scoring-dev
    python scripts/query_endpoint.py --profile <profile> --cliente cliente.json
"""

import argparse
import json
import time

from databricks.sdk import WorkspaceClient

# Cliente de exemplo (dados brutos, como chegariam do sistema de concessão). Nulos são aceitos
# em MonthlyIncome e NumberOfDependents.
CLIENTE_EXEMPLO = {
    "RevolvingUtilizationOfUnsecuredLines": 0.95,
    "age": 32,
    "NumberOfTime30-59DaysPastDueNotWorse": 2,
    "DebtRatio": 0.45,
    "MonthlyIncome": 3200,
    "NumberOfOpenCreditLinesAndLoans": 7,
    "NumberOfTimes90DaysLate": 1,
    "NumberOfTime60-89DaysPastDueNotWorse": 0,
    "NumberOfDependents": None,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True, help="profile da Databricks CLI")
    ap.add_argument("--endpoint", default="credit-scoring-dev")
    ap.add_argument("--cliente", help="arquivo JSON com um cliente (padrão: cliente de exemplo)")
    args = ap.parse_args()

    cliente = CLIENTE_EXEMPLO
    if args.cliente:
        with open(args.cliente, encoding="utf-8") as f:
            cliente = json.load(f)

    w = WorkspaceClient(profile=args.profile)
    t0 = time.perf_counter()
    pred = w.serving_endpoints.query(args.endpoint, dataframe_records=[cliente]).predictions[0]
    ms = (time.perf_counter() - t0) * 1000
    print(json.dumps(pred, indent=2, ensure_ascii=False))
    print(f"latência: {ms:.0f} ms")


if __name__ == "__main__":
    main()
