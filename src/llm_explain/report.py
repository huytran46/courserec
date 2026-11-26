"""Per-group summary, hand-label sheet, and checker precision/recall against hand labels."""
from __future__ import annotations

import random
from collections import Counter, defaultdict

from .core import FLAGGED, LABELS

CSV_FIELDS = ["example_id", "group", "path_facts", "explanation", "claim_idx", "head", "relation", "tail",
              "quote", "human_true", "path_odd", "missed_claim", "notes"]
SCORED = [l for l in LABELS if l != "recommendation_framing"]


def summarize(rows):
    by = defaultdict(list)
    for r in rows:
        by[r["group"]].append(r)
    out = {}
    for g, rs in sorted(by.items()):
        ok = [r for r in rs if r["claims"] is not None]
        c = Counter(x["label"] for r in ok for x in r["claims"])
        total = sum(c.values())
        scored = total - c["recommendation_framing"]
        out[g] = {"examples": len(rs), "extraction_failures": len(rs) - len(ok), "claims_total": total,
                  "claims_scored": scored, "claims_per_explanation": total / len(ok) if ok else None,
                  "recommendation_framing": c["recommendation_framing"],
                  "count": {l: c[l] for l in SCORED},
                  "rate": {l: (c[l] / scored if scored else None) for l in SCORED},
                  "cjk_leak": sum(r["cjk_leak"] for r in rs),
                  "explanations_with_flag_wording": sum(r["flag_wording"] for r in rs),
                  "code_name_mismatches": sum(r["code_name_mismatches"] for r in rs)}
    return out


def pick_for_labeling(rows, n=40, seed=0):
    """Round-robin over groups (each shuffled with a fixed seed), so small groups give all they have."""
    rng = random.Random(seed)
    pools = defaultdict(list)
    for r in sorted(rows, key=lambda r: r["example_id"]):
        pools[r["group"]].append(r)
    for g in sorted(pools):
        rng.shuffle(pools[g])
    out = []
    while len(out) < n and any(pools.values()):
        for g in sorted(pools):
            if pools[g] and len(out) < n:
                out.append(pools[g].pop())
    return sorted(out, key=lambda r: r["example_id"])


def label_rows(rows):
    out = []
    for r in rows:
        for k, c in enumerate(r["claims"]) if r["claims"] is not None else [("", {})]:
            base = {f: "" for f in CSV_FIELDS}
            base.update(example_id=r["example_id"], group=r["group"], path_facts=r["path_facts"],
                        explanation=r["explanation"], claim_idx=k,
                        **{f: str(c.get(f, "")) for f in ("head", "relation", "tail", "quote")})
            out.append(base)
    return out


def score_labels(sheet, rows):
    """Positive = a claim that is NOT true on the path. Checker predicts positive when its label is in FLAGGED."""
    by_id = {r["example_id"]: r for r in rows}
    cells = defaultdict(Counter)
    for s in sheet:
        if str(s["claim_idx"]) == "":
            continue
        r = by_id[int(s["example_id"])]
        c = r["claims"][int(s["claim_idx"])]
        if any(str(c.get(f, "")) != s[f] for f in ("head", "relation", "tail")):
            raise ValueError(f"sheet row does not match results: example {s['example_id']} claim {s['claim_idx']}")
        v = s["human_true"].strip().lower()
        if v not in ("true", "false"):
            raise ValueError(f"human_true must be true/false: example {s['example_id']} claim {s['claim_idx']}: {v!r}")
        key = ("t" if (c["label"] in FLAGGED) == (v == "false") else "f") + ("p" if c["label"] in FLAGGED else "n")
        cells["all"][key] += 1
        cells[r["group"]][key] += 1
    out = {}
    for g, c in cells.items():
        tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
        out[g] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
                  "precision": tp / (tp + fp) if tp + fp else None, "recall": tp / (tp + fn) if tp + fn else None}
    return out
