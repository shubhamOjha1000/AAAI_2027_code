"""Step 7 — evaluate the router and place it on the accuracy-vs-tokens Pareto.

Two levels:
  (1) Router classification on the rare DETAIL class: precision / recall / F1 /
      PR-AUC from out-of-fold predictions (accuracy is meaningless at 9.7% prior).
  (2) End-to-end: route each sample (DETAIL->use fovea C, GIST->use global B) and
      report (VQA accuracy, mean visual tokens), compared to the anchors:
      always-Full / always-Fovea / always-Global, the ORACLE max(B,C), and the
      question-TYPE-PRIOR. Also sweeps tau to draw the router's Pareto curve.
"""
import json
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, average_precision_score

import config as C


def _load():
    rows = {f"{r['question_type']}||{r['id']}": r
            for r in (json.loads(l) for l in open(C.LABELS) if l.strip())}
    oof = [json.loads(l) for l in open(C.OOF) if l.strip()]
    keys = [o["key"] for o in oof]
    rows = [rows[k] for k in keys]
    prob = np.array([o["prob"] for o in oof])
    y = np.array([1 if o["label"] == "DETAIL" else 0 for o in oof])
    qt = np.array([r["question_type"] for r in rows])
    bc = np.array([r["B_correct"] for r in rows], dtype=bool)
    cc = np.array([r["C_correct"] for r in rows], dtype=bool)
    ac = np.array([r["A_correct"] for r in rows], dtype=bool)
    bt = np.array([r["B_tokens"] for r in rows], dtype=float)
    ct = np.array([r["C_tokens"] for r in rows], dtype=float)
    at = np.array([r["A_tokens"] for r in rows], dtype=float)
    tau = json.load(open(C.OOF.replace("oof_predictions.jsonl", "router_meta.json")))["operating_tau"]
    return keys, qt, y, prob, ac, bc, cc, at, bt, ct, tau


def route_point(detail_mask, bc, cc, bt, ct):
    correct = np.where(detail_mask, cc, bc)
    tokens = np.where(detail_mask, ct, bt)
    return 100 * correct.mean(), tokens.mean()


def main():
    keys, qt, y, prob, ac, bc, cc, at, bt, ct, tau = _load()
    out = {}

    # ---- (1) router classification on DETAIL ----
    pred = (prob > tau).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y, pred, labels=[1], average="binary", zero_division=0)
    out["router"] = {
        "operating_tau": round(tau, 3),
        "DETAIL_precision": round(float(p), 3),
        "DETAIL_recall": round(float(r), 3),
        "DETAIL_f1": round(float(f1), 3),
        "PR_AUC": round(float(average_precision_score(y, prob)) if y.sum() else 0.0, 3),
        "n_positive": int(y.sum()), "n": int(len(y)),
    }

    # ---- (2) end-to-end anchors ----
    accA, tokA = 100 * ac.mean(), at.mean()
    accC, tokC = 100 * cc.mean(), ct.mean()
    accB, tokB = 100 * bc.mean(), bt.mean()
    # oracle: fovea only where it uniquely rescues (B wrong & C right); else global
    oracle_detail = (~bc) & cc
    accO, tokO = route_point(oracle_detail, bc, cc, bt, ct)
    # type-prior: per type pick whichever of B/C has higher type accuracy
    type_detail = np.zeros(len(y), dtype=bool)
    for t in np.unique(qt):
        m = qt == t
        type_detail[m] = cc[m].mean() > bc[m].mean()
    accT, tokT = route_point(type_detail, bc, cc, bt, ct)
    # learned router at operating tau
    accR, tokR = route_point(pred.astype(bool), bc, cc, bt, ct)

    out["end_to_end"] = {
        "always_full_A": [round(accA, 1), round(tokA, 1)],
        "always_fovea_C": [round(accC, 1), round(tokC, 1)],
        "always_global_B": [round(accB, 1), round(tokB, 1)],
        "oracle_maxBC": [round(accO, 1), round(tokO, 1)],
        "type_prior": [round(accT, 1), round(tokT, 1)],
        "router_learned": [round(accR, 1), round(tokR, 1)],
    }

    # ---- router Pareto (sweep tau) ----
    pareto = []
    for t in C.TAU_GRID:
        a, tk = route_point(prob > t, bc, cc, bt, ct)
        pareto.append([round(t, 2), round(a, 1), round(tk, 1)])
    out["router_pareto"] = pareto

    json.dump(out, open(C.METRICS, "w"), indent=2)
    print(json.dumps(out, indent=2))

    # ---- plot ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 5))
        pc = np.array([[p[2], p[1]] for p in pareto])  # tokens, acc
        ax.plot(pc[:, 0], pc[:, 1], "-", color="#aaa", label="router (tau sweep)", zorder=1)
        pts = {"Full A": (tokA, accA, "#1f77b4"), "Fovea C": (tokC, accC, "#ff7f0e"),
               "Global B": (tokB, accB, "#d62728"), "Oracle": (tokO, accO, "#2ca02c"),
               "Type-prior": (tokT, accT, "#9467bd"), "Router": (tokR, accR, "#000")}
        for name, (x, yv, col) in pts.items():
            ax.scatter([x], [yv], c=col, s=90, zorder=3,
                       marker="*" if name == "Router" else "o")
            ax.annotate(name, (x, yv), textcoords="offset points", xytext=(6, 5), fontsize=9)
        ax.set_xlabel("mean visual tokens (lower = cheaper)")
        ax.set_ylabel("VQA accuracy (%)")
        ax.set_title("Stage-3 router: accuracy vs tokens")
        ax.grid(alpha=0.3); ax.legend(loc="lower right", fontsize=8)
        fig.tight_layout(); fig.savefig(C.PARETO_PNG, dpi=140)
        print(f"saved {C.PARETO_PNG}")
    except Exception as e:
        print("plot skipped:", e)


if __name__ == "__main__":
    main()
