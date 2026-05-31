"""Aggregate prediction + judgment JSONLs into the paper-style breakdown reports."""
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


def _join_predictions_and_judgments(predictions_dir: Path, judgments_path: Path) -> pd.DataFrame:
    judgments = list(read_jsonl(judgments_path))
    jdf = pd.DataFrame(judgments)

    pred_rows = []
    for pf in sorted(Path(predictions_dir).glob("*.jsonl")):
        for r in read_jsonl(pf):
            pred_rows.append(r)
    pdf = pd.DataFrame(pred_rows)

    merged = pdf.merge(jdf, on=["model", "sample_id"], how="inner", suffixes=("", "_j"))
    return merged


def build_reports(
    predictions_dir: Optional[Path] = None,
    judgments_path: Optional[Path] = None,
    reports_dir: Optional[Path] = None,
) -> Path:
    predictions_dir = Path(predictions_dir or config.PREDICTIONS_DIR)
    judgments_path = Path(judgments_path or (config.JUDGMENTS_DIR / "qwen_judge.jsonl"))
    reports_dir = Path(reports_dir or config.REPORTS_DIR)
    ensure_dir(reports_dir)

    merged = _join_predictions_and_judgments(predictions_dir, judgments_path)
    merged.to_csv(reports_dir / "joined_full.csv", index=False)

    overall_rows = []
    eff_rows = []
    domain_dfs = []
    qtype_dfs = []
    quality_dfs = []
    hi_lo_dfs = []

    for model_name, df in merged.groupby("model"):
        recs = df.to_dict(orient="records")

        overall = overall_accuracy(recs)
        overall["model"] = model_name
        overall_rows.append(overall)

        eff = efficiency_summary(recs)
        eff["model"] = model_name
        eff_rows.append(eff)

        d = accuracy_by_field(recs, "domain")
        d.insert(0, "model", model_name)
        domain_dfs.append(d)

        q = accuracy_by_field(recs, "question_type")
        q.insert(0, "model", model_name)
        qtype_dfs.append(q)

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
    md_lines.append("## Overall accuracy")
    md_lines.append(overall_df.to_markdown(index=False))
    md_lines.append("")
    md_lines.append("## Efficiency summary (per-sample means)")
    md_lines.append(eff_df.to_markdown(index=False))
    md_lines.append("")
    md_lines.append("## High vs low quality images")
    md_lines.append(pd.concat(hi_lo_dfs, ignore_index=True).to_markdown(index=False))
    summary_md = reports_dir / "summary_table.md"
    summary_md.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"[aggregate] reports written under {reports_dir}")
    return reports_dir
