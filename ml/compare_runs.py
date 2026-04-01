"""Compare quelques runs MLflow d’une expérience (CLI rapide)."""

from __future__ import annotations

import argparse

import mlflow
from mlflow.tracking import MlflowClient


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", type=str, default="telco_churn")
    p.add_argument("--max-runs", type=int, default=10)
    args = p.parse_args()

    client = MlflowClient()
    exp = client.get_experiment_by_name(args.experiment)
    if exp is None:
        raise SystemExit(f"Expérience introuvable : {args.experiment}")

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=args.max_runs,
    )
    if not runs:
        print("Aucun run.")
        return

    print(f"{'run_id':<32} {'recall_test':>12} {'roc_auc':>10} {'threshold':>10}")
    for r in runs:
        mid = r.info.run_id[:8] + "…"
        m = r.data.metrics
        rec = m.get("test_recall_tuned", m.get("test_recall_default", float("nan")))
        auc = m.get("test_roc_auc", float("nan"))
        thr = m.get("threshold", float("nan"))
        print(f"{mid:<32} {rec:12.4f} {auc:10.4f} {thr:10.4f}")


if __name__ == "__main__":
    main()
