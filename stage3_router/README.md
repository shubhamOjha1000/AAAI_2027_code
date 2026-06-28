# Stage-3 Question-Conditioned Router — pilot

Trains a tiny router that decides, **per question**, whether to feed the decoder the
expensive gaze **fovea** (DETAIL) or just the cheap **global** thumbnail (GIST) — the
"how much" axis of the Gaze×Question adaptive-tiling architecture.

This is a **feasibility pilot** on the 300-sample, 5-condition WearVQA run
(SmolVLM2-256M). Runs on **Colab CPU**; the router is a tiny MLP over cached
question embeddings.

## Data (bundled in `data/`)
- `partition5_compare_verdicts.jsonl` — per-(sample, algo) correctness (judged).
- `partition5_compare_responses.jsonl` — per-(sample, algo) `n_partitions` (tokens).

Label: **DETAIL** iff global-only (B) is **wrong** and gaze-fovea (C) is **right**;
else **GIST**. (29 DETAIL / 271 GIST.)

## Pipeline (run in order)
```bash
python build_router_labels.py        # -> outputs/router_labels.jsonl
python extract_question_features.py  # -> outputs/features.npy  (sentence-transformers, CPU)
python splits.py                     # -> outputs/folds.json    (stratified+grouped 5-fold)
python train_router.py               # -> outputs/oof_predictions.jsonl  (CV, balanced sampler)
python eval_router.py                # -> outputs/metrics.json + outputs/pareto.png
```
Or just run the notebook: `train_router_colab.ipynb`.

## What it reports
- **Router head:** DETAIL precision / recall / F1 / PR-AUC (out-of-fold; accuracy is
  meaningless at a 9.7% positive prior).
- **End-to-end:** (accuracy, mean tokens) of the learned router vs the anchors —
  **always-Full / Fovea / Global**, the **oracle** `max(B,C)`, and the
  **question-type prior** — plus the router's tau-sweep Pareto curve.

## Targets (from the pilot data)
- Oracle `max(B,C)`: **28.0% @ ~73 tokens** (the ceiling; near always-Full's 28.7% @ 832).
- Type-prior: **24.0% @ ~139 tokens** (the floor a learned router must beat — it
  collapses to always-Fovea).

Success = the learned router lands **clearly above the type-prior, toward the oracle**.

## Imbalance handling (29:271)
- Keep every negative; **WeightedRandomSampler** oversamples positives to ~50/50 per
  batch (`BATCH_SIZE=32`).
- Early-stop on inner-val **PR-AUC**; tune the decision threshold **tau on inner-val**
  to maximise routed end-to-end accuracy (never on the test fold).
- Out-of-fold predictions pooled over 5 folds × 5 seeds for stable metrics.

## Caveats
Pilot only: 256M model, 300 samples, **29 positives**, strict text-only judge. A
learned router may not beat the type-prior at this scale — that is the signal to scale
to full WearVQA + real eye-tracked gaze, not that the idea is wrong (the oracle shows
the headroom is real).
