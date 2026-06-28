"""Step 1 — materialize per-sample router labels + everything eval needs.

Reads the 5-condition verdicts (correctness) and responses (n_partitions) and
writes one row per sample with:
  id, question_type, scene_id, question,
  label  in {DETAIL, GIST}   (DETAIL iff global wrong AND fovea right),
  {A,B,C}_correct (bool), {A,B,C}_tokens (int)

Label rationale (see project notes): the router should send a question to the
high-res fovea only when the cheap global route would have been wrong but the
fovea rescues it. Everything else -> GIST.
"""
import json
import config as C


def _load(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    verdicts = _load(C.VERDICTS)
    responses = _load(C.RESPONSES)

    correct = {(r["question_type"], str(r["id"]), r["algo"]): bool(r["correct"]) for r in verdicts}
    npart = {(r["question_type"], str(r["id"]), r["algo"]): int(r["n_partitions"]) for r in responses}
    question = {(r["question_type"], str(r["id"])): r["question"] for r in verdicts}

    keys = sorted({(r["question_type"], str(r["id"])) for r in verdicts})
    rows = []
    n_detail = 0
    for qt, sid in keys:
        b = correct[(qt, sid, C.GLOBAL_ALGO)]
        c = correct[(qt, sid, C.FOVEA_ALGO)]
        a = correct[(qt, sid, C.FULL_ALGO)]
        label = "DETAIL" if (not b and c) else "GIST"
        n_detail += label == "DETAIL"
        rows.append({
            "id": sid,
            "question_type": qt,
            "scene_id": sid,                       # group key (each image = its own group)
            "question": question[(qt, sid)],
            "label": label,
            "A_correct": a, "B_correct": b, "C_correct": c,
            "A_tokens": npart[(qt, sid, C.FULL_ALGO)] * C.SEQ,
            "B_tokens": npart[(qt, sid, C.GLOBAL_ALGO)] * C.SEQ,
            "C_tokens": npart[(qt, sid, C.FOVEA_ALGO)] * C.SEQ,
        })

    with open(C.LABELS, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    print(f"wrote {len(rows)} rows -> {C.LABELS}")
    print(f"  DETAIL (positive): {n_detail}   GIST (negative): {len(rows) - n_detail}")
    print(f"  positive rate: {100 * n_detail / len(rows):.1f}%")


if __name__ == "__main__":
    main()
