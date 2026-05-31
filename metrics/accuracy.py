"""Aggregate per-sample judgments into the paper's accuracy breakdowns."""
from collections import defaultdict
from typing import Iterable

import pandas as pd

from .. import config


def _safe_pct(num: int, denom: int) -> float:
    return 100.0 * num / denom if denom else 0.0


def overall_accuracy(judgments: Iterable[dict]) -> dict:
    n = c = 0
    for j in judgments:
        n += 1
        c += int(j["correct"])
    return {"n": n, "correct": c, "accuracy_pct": _safe_pct(c, n)}


def accuracy_by_field(joined_rows: Iterable[dict], field: str) -> pd.DataFrame:
    """joined_rows already merges judgment + sample metadata."""
    buckets: dict[str, list[int]] = defaultdict(list)
    for row in joined_rows:
        buckets[row.get(field, "unknown")].append(int(row["correct"]))
    out = []
    for key, vals in buckets.items():
        out.append({
            field: key,
            "n": len(vals),
            "correct": sum(vals),
            "accuracy_pct": _safe_pct(sum(vals), len(vals)),
        })
    return pd.DataFrame(out).sort_values(field).reset_index(drop=True)


def accuracy_by_quality(joined_rows: list[dict]) -> pd.DataFrame:
    rows = []
    for flag in config.QUALITY_FIELDS:
        subset = [r for r in joined_rows if r.get("quality_flags", {}).get(flag) == "yes"]
        corrects = [int(r["correct"]) for r in subset]
        rows.append({
            "quality_issue": flag,
            "n": len(subset),
            "correct": sum(corrects),
            "accuracy_pct": _safe_pct(sum(corrects), len(subset)),
        })
    return pd.DataFrame(rows)


def hi_vs_lo_quality(joined_rows: list[dict]) -> pd.DataFrame:
    hi = [r for r in joined_rows if not any(r.get("quality_flags", {}).get(f) == "yes" for f in config.QUALITY_FIELDS)]
    lo = [r for r in joined_rows if any(r.get("quality_flags", {}).get(f) == "yes" for f in config.QUALITY_FIELDS)]
    return pd.DataFrame([
        {
            "bucket": "high_quality",
            "n": len(hi),
            "correct": sum(int(r["correct"]) for r in hi),
            "accuracy_pct": _safe_pct(sum(int(r["correct"]) for r in hi), len(hi)),
        },
        {
            "bucket": "low_quality_any_issue",
            "n": len(lo),
            "correct": sum(int(r["correct"]) for r in lo),
            "accuracy_pct": _safe_pct(sum(int(r["correct"]) for r in lo), len(lo)),
        },
    ])


def efficiency_summary(prediction_rows: list[dict]) -> dict:
    """Mean of the per-sample efficiency metrics."""
    if not prediction_rows:
        return {}
    def m(k): return sum(r[k] for r in prediction_rows) / len(prediction_rows)
    return {
        "n": len(prediction_rows),
        "mean_ttft_s": m("ttft_s"),
        "mean_throughput_tok_s": m("throughput_tok_s"),
        "mean_peak_vram_gb": m("peak_vram_gb"),
        "mean_visual_tokens": m("visual_tokens"),
        "mean_output_tokens": m("output_tokens"),
        "mean_total_generation_s": m("total_generation_s"),
    }
