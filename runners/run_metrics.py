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


def run_metrics(models=None, judges=None) -> Path:
    judge_all(models=models, judges=judges)
    reports_dir = build_reports()
    print("\n=== Per-judge accuracy (+ majority_vote) ===")
    print(pd.read_csv(reports_dir / "judge_accuracy.csv").to_string(index=False))
    print("\n=== Overall accuracy (majority-vote consensus) ===")
    print(pd.read_csv(reports_dir / "overall_accuracy.csv").to_string(index=False))
    return reports_dir


def main():
    ap = argparse.ArgumentParser(description="Judge predictions and build accuracy reports.")
    ap.add_argument("--models", nargs="*", default=None,
                    help="only score these model keys (default: every predictions/*.jsonl)")
    ap.add_argument("--judges", nargs="*", default=None,
                    help="only use these judge keys (default: all in JUDGE_REGISTRY)")
    args = ap.parse_args()
    run_metrics(models=args.models, judges=args.judges)


if __name__ == "__main__":
    main()
