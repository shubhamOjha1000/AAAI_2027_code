"""Accuracy metric: score predictions with the LLM-as-judge, then build reports.

CLI:
  python -m wearvqa_bench.runners.run_metrics                      # all prediction files
  python -m wearvqa_bench.runners.run_metrics --models smolvlm2 internvl2_5_1b
"""
import argparse
from pathlib import Path

import pandas as pd

from .run_judge import judge_all
from .aggregate import build_reports


def run_metrics(models=None) -> Path:
    judge_all(models=models)
    reports_dir = build_reports()
    overall = pd.read_csv(reports_dir / "overall_accuracy.csv")
    print("\n=== Overall accuracy ===")
    print(overall.to_string(index=False))
    return reports_dir


def main():
    ap = argparse.ArgumentParser(description="Judge predictions and build accuracy reports.")
    ap.add_argument("--models", nargs="*", default=None,
                    help="only score these model keys (default: every predictions/*.jsonl)")
    args = ap.parse_args()
    run_metrics(models=args.models)


if __name__ == "__main__":
    main()
