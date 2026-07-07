# FRM — Foveal Relevance Module (Stage 2b)

Learned, **question-free** relevance predictor on the global-thumbnail side of the
gaze×question adaptive-tiling VLM. At inference it takes the gaze token + all 64
global tokens and outputs a per-token relevance `r[64]` **without the question**;
`KEEP_g = top_n(r)` is the retrieved context, `MERGE_g = rest` goes to Stage 2c.
It is trained by **distillation from the frozen VLM**: the VLM *with* the question
produces per-token importance labels; FRM learns their **marginal** over questions
(the "anticipatory context" — what people usually need when they look there).

Reference: `../FRM_training_spec.pdf`.

## Setup (locked)
| choice | value |
|---|---|
| dataset | **WearVQA gaze_only** (proof-of-concept) — `MyDrive/wearvqa_gaze_only` |
| VLM | **SmolVLM2-2.2B-Instruct**, frozen (teacher + encoder) |
| compute | **Colab GPU** for everything (label-gen is the expensive part) |
| scope | standalone FRM box; Stage 2a=fixed 1-ring exclusion, 2c=mean-pool stand-in |

> ⚠️ **Anti-leakage caveat.** WearVQA gaze was annotated *at the answer cue*
> (question-derived). The spec forbids exactly this. So we run the
> **precondition** + **Exp 2a** first to measure how much far-context signal
> actually exists; expect it to be weak, and report that honestly. This is a
> proof-of-concept harness, not the final real-gaze (Ego-Exo4D/EGTEA/Aria) run.

## Files
| file | role |
|---|---|
| `config.py` | all paths / dims / hyperparameters (env-overridable) |
| `data.py` | WearVQA gaze_only loader (`question`, `response`, `gaze`) |
| `geometry.py` | gaze→cell, fovea region, cell distances (8×8 grid) |
| `vlm_teacher.py` | frozen SmolVLM2: `G[64,d]`, rollout labels, LOO, answer-preservation |
| `labels.py` | **Phase 1** — cache `G` + `imp_*` per sample (resumable) |
| `frm_model.py` | **student** cross-attn head + masked KL loss (the only trained part) |
| `train_frm.py` | **Phase 2** — OOF KL-distillation training |
| `deploy.py` | **Phase 3** — `r = FRM(gaze,G)`, `KEEP_g = top_n` |
| `baselines.py` | eccentricity (rival) / random (floor) / full-G (ceiling) |
| `metrics.py`, `preserve_eval.py` | Spearman, top-n recall, far-stratify, answer-preservation |
| `experiments/` | `precondition`, `exp2a`, `exp1`, `exp5`, `exp7` |
| `notebooks/` | one Colab notebook per experiment (+ label-gen) |

## Run order (Colab)
1. `notebooks/00_generate_labels.ipynb` — **Phase 1, GPU, one-time.** Caches
   `G` + teacher labels to `MyDrive/frm_out/labels`. Do this first.
2. `notebooks/exp2a_precondition_and_label_validation.ipynb` — precondition +
   Exp 2a. **Cheap, no training.** Tells you whether the data has signal at all.
3. `notebooks/exp1_frm_vs_eccentricity.ipynb` — **make-or-break.** FRM vs
   eccentricity on far context. Go/no-go for the whole module.
4. `notebooks/exp5_budget_sweep.ipynb` — accuracy-vs-tokens knee.
5. `notebooks/exp7_capacity_sweep.ipynb` — smallest architecture that fits.

## The 4 experiments
- **Exp 2a** — answer- vs question-grounding vs gaze-distance. Divergence ⇒ FRM
  needed. *(run first, no training)*
- **Exp 1** — FRM vs Eccentricity, answer-preservation @ n=12, stratified NEAR/FAR.
  FRM > Ecc on FAR ⇒ justified; FRM ≈ Ecc ⇒ drop it.
- **Exp 5** — KEEP_g budget sweep n∈{4,8,12,16,24}.
- **Exp 7** — capacity sweep (heads×depth), pick smallest non-underfitting.

## Local dev
`config.py`, `geometry.py`, `frm_model.py` import without a GPU. Override the
Drive paths for local checks:
```bash
FRM_OUT_DIR=/tmp/frm_out DATA_DIR=/tmp/none python -c "import geometry"
```
