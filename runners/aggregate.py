"""Aggregate predictions + multi-judge judgments into paper-style breakdown reports.

With a judge ensemble, each judge writes judgments/<judge_key>.jsonl. We report:
  - judge_accuracy.csv : overall accuracy per (model, judge) + a majority_vote row.
  - the detailed breakdowns (domain / question type / quality / hi-vs-lo / overall)
    computed on the MAJORITY-VOTE consensus correctness (one verdict per sample).
"""
from pathlib import Path
from typing import Optional

import pandas as pd

from .. import config
from ..metrics.accuracy import (
    accuracy_by_field,
    accuracy_by_quality,
    efficiency_summary,
    hi_vs_lo_quality,
    overall_accuracy,
)
from ..utils.io import ensure_dir, read_jsonl


def _load_predictions(predictions_dir: Path) -> pd.DataFrame:
    rows = []
    for pf in sorted(Path(predictions_dir).glob("*.jsonl")):
        rows.extend(read_jsonl(pf))
    return pd.DataFrame(rows)


def _load_judgments(judgments_dir: Path) -> pd.DataFrame:
    """One row per (model, sample_id, judge). judge key = the 'judge' field or filename."""
    rows = []
    for jf in sorted(Path(judgments_dir).glob("*.jsonl")):
        for r in read_jsonl(jf):
            rows.append({
                "model": r["model"],
                "sample_id": r["sample_id"],
                "judge": r.get("judge", jf.stem),
                "correct": bool(r["correct"]),
            })
    return pd.DataFrame(rows)


def build_reports(
    predictions_dir: Optional[Path] = None,
    judgments_dir: Optional[Path] = None,
    reports_dir: Optional[Path] = None,
) -> Path:
    predictions_dir = Path(predictions_dir or config.PREDICTIONS_DIR)
    judgments_dir = Path(judgments_dir or config.JUDGMENTS_DIR)
    reports_dir = Path(reports_dir or config.REPORTS_DIR)
    ensure_dir(reports_dir)

    pdf = _load_predictions(predictions_dir)
    jdf = _load_judgments(judgments_dir)

    # --- per-judge overall accuracy ---
    per_judge = (jdf.groupby(["model", "judge"])
                    .agg(n=("correct", "size"), correct=("correct", "sum")).reset_index())
    per_judge["accuracy_pct"] = 100.0 * per_judge["correct"] / per_judge["n"]

    # --- majority-vote consensus per (model, sample_id): correct if >= half agree ---
    cons = jdf.groupby(["model", "sample_id"])["correct"].mean().reset_index()
    cons["correct"] = cons["correct"] >= 0.5
    maj = (cons.groupby("model")
              .agg(n=("correct", "size"), correct=("correct", "sum")).reset_index())
    maj["accuracy_pct"] = 100.0 * maj["correct"] / maj["n"]
    maj.insert(1, "judge", "majority_vote")

    judge_accuracy = (pd.concat([per_judge, maj], ignore_index=True)
                        .sort_values(["model", "judge"]).reset_index(drop=True))
    judge_accuracy.to_csv(reports_dir / "judge_accuracy.csv", index=False)

    # --- detailed breakdowns use the consensus correctness ---
    merged = pdf.merge(cons[["model", "sample_id", "correct"]],
                       on=["model", "sample_id"], how="inner")
    merged.to_csv(reports_dir / "joined_full.csv", index=False)

    overall_rows, eff_rows = [], []
    domain_dfs, qtype_dfs, quality_dfs, hi_lo_dfs = [], [], [], []
    for model_name, df in merged.groupby("model"):
        recs = df.to_dict(orient="records")

        overall = overall_accuracy(recs)
        overall["model"] = model_name
        overall_rows.append(overall)

        eff = efficiency_summary(recs)
        eff["model"] = model_name
        eff_rows.append(eff)

        for dfs, field in ((domain_dfs, "domain"), (qtype_dfs, "question_type")):
            t = accuracy_by_field(recs, field)
            t.insert(0, "model", model_name)
            dfs.append(t)

        ql = accuracy_by_quality(recs)
        ql.insert(0, "model", model_name)
        quality_dfs.append(ql)

        hl = hi_vs_lo_quality(recs)
        hl.insert(0, "model", model_name)
        hi_lo_dfs.append(hl)

    overall_df = pd.DataFrame(overall_rows)[["model", "n", "correct", "accuracy_pct"]]
    eff_df = pd.DataFrame(eff_rows)
    cols = ["model", "n", "mean_ttft_s", "mean_throughput_tok_s",
            "mean_peak_vram_gb", "mean_visual_tokens", "mean_output_tokens",
            "mean_total_generation_s"]
    eff_df = eff_df[[c for c in cols if c in eff_df.columns]]

    overall_df.to_csv(reports_dir / "overall_accuracy.csv", index=False)
    eff_df.to_csv(reports_dir / "efficiency_summary.csv", index=False)
    pd.concat(domain_dfs, ignore_index=True).to_csv(reports_dir / "accuracy_by_domain.csv", index=False)
    pd.concat(qtype_dfs, ignore_index=True).to_csv(reports_dir / "accuracy_by_question_type.csv", index=False)
    pd.concat(quality_dfs, ignore_index=True).to_csv(reports_dir / "accuracy_by_quality_issue.csv", index=False)
    pd.concat(hi_lo_dfs, ignore_index=True).to_csv(reports_dir / "hi_vs_lo_quality.csv", index=False)

    md_lines = ["# WearVQA Benchmark Summary", ""]
    md_lines.append("## Overall accuracy (majority-vote consensus)")
    md_lines.append(overall_df.to_markdown(index=False))
    md_lines.append("")
    md_lines.append("## Per-judge accuracy")
    md_lines.append(judge_accuracy.to_markdown(index=False))
    md_lines.append("")
    md_lines.append("## High vs low quality images (consensus)")
    md_lines.append(pd.concat(hi_lo_dfs, ignore_index=True).to_markdown(index=False))
    (reports_dir / "summary_table.md").write_text("\n".join(md_lines), encoding="utf-8")
    print(f"[aggregate] reports written under {reports_dir}")
    return reports_dir
