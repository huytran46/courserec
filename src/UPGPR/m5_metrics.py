"""M5 metric contract (units: fractions in [0,1], NOT percent) and S0/S3 selection on a saved beam.

Reuses Spike B reconstruction (collect_pred_paths, best_paths_legacy, select_path) unchanged:
  * item rank = legacy (dot(user+enroll,item), legacy path prob) desc; ties broken by item id (stable, asc id first)
  * train items excluded (train-only exclusion, validation/test protocol of baseline); no fallback items
  * S0 = legacy best path per pair; S3 = concept path with highest legacy score (S0 if none), same item list.
Eligible users = all users with >=1 held-out positive in the split; users with no list score 0.
"""
from __future__ import annotations

import math
import os
import pickle
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spike_b_posthoc as sb  # noqa: E402

K = 10
FAMILIES = ["co_learner", "concept", "teacher", "school"]  # frozen before results (mooc.json schema)
CONCEPT_REL = "item_concept"
FAMILY_REL = {"concept": "item_concept", "teacher": "item_teacher", "school": "item_school"}


def rank_key(c):
    """c = (score, prob, path, beam_idx). Higher score, then higher prob, then lower item id."""
    return (c[0], c[1], -c[2][-1][2])


def freeze(best):
    return {u: sorted(cs, key=rank_key, reverse=True)[:K] for u, cs in best.items()}


def has_concept(path):
    return CONCEPT_REL in sb.relations(path)


def families(path, self_loop="self_loop"):
    """Set of families touched by the EFFECTIVE (non self-loop) edges. co_learner = an enroll edge whose
    tail is a user (item -> user), i.e. the co-enrollment hop."""
    fam = set()
    for rel, ntype, _ in path[1:]:
        if rel == self_loop:
            continue
        if rel == "enroll" and ntype == "user":
            fam.add("co_learner")
        for f, r in FAMILY_REL.items():
            if rel == r:
                fam.add(f)
    return fam


def family_of_path(path, self_loop="self_loop"):
    """Single family label of a path: priority concept > teacher > school > co_learner (mutually exclusive in
    4-node patterns; priority only matters for longer paths). None if no family (e.g. only direct enroll)."""
    f = families(path, self_loop)
    for name in ("concept", "teacher", "school", "co_learner"):
        if name in f:
            return name
    return None


def user_metrics(ids, positives, k=K):
    """ids: unique item ids in rank order (best first), len<=k. Returns dict of per-user metrics."""
    ids = list(ids)[:k]
    assert len(set(ids)) == len(ids), "duplicate item in list"
    npos = len(positives)
    hits = [1 if i in positives else 0 for i in ids]
    h = sum(hits)
    dcg = sum(1.0 / math.log2(r + 2) for r, x in enumerate(hits) if x)
    idcg = sum(1.0 / math.log2(r + 2) for r in range(min(k, npos)))
    mrr = next((1.0 / (r + 1) for r, x in enumerate(hits) if x), 0.0)
    return {"precision": h / k, "recall": h / npos if npos else 0.0, "ndcg": dcg / idcg if idcg else 0.0,
            "hr": 1.0 if h else 0.0, "mrr": mrr, "hits": h, "n_items": len(ids)}


def build(results, scores, labels, train_labels):
    pred = sb.collect_pred_paths(results, scores, labels)
    best = sb.best_paths_legacy(pred, train_labels)
    return pred, freeze(best)


def per_user_rows(pred, frozen, labels, selector, self_loop="self_loop"):
    """One row per eligible user (every uid in labels with >=1 positive)."""
    rows = {}
    for uid, pos in labels.items():
        if not pos:
            continue
        top = frozen.get(uid, [])
        ids = [c[2][-1][2] for c in top]
        m = user_metrics(ids, set(pos))
        sel = []
        for s0 in top:
            cands = pred[uid][s0[2][-1][2]]
            ch = sb.select_path("S3" if selector == "S3" else "S0", cands, s0, self_loop)
            assert any(ch is c for c in cands)
            sel.append(ch)
        nc = sum(has_concept(c[2]) for c in sel)
        fams = Counter(f for c in sel if (f := family_of_path(c[2], self_loop)))
        m.update({"uid": uid, "n_valid_expl": len(sel), "n_concept_expl": nc, "concept_at10": nc / K,
                  "expl_per_item": (len(sel) / len(ids)) if ids else 0.0,
                  "fam": dict(fams), "ids": ids, "sel_beam_idx": [c[3] for c in sel],
                  "n_candidates": len(pred.get(uid, {})),
                  "n_paths": sum(len(v) for v in pred.get(uid, {}).values())})
        rows[uid] = m
    return rows


def macro(rows, key):
    xs = [r[key] for r in rows.values()]
    return float(np.mean(xs)) if xs else float("nan")


def aggregate(rows, labels):
    a = {k: macro(rows, k) for k in ("precision", "recall", "ndcg", "hr", "mrr", "concept_at10", "expl_per_item")}
    n = len(rows)
    tot_expl = sum(r["n_valid_expl"] for r in rows.values())
    tot_conc = sum(r["n_concept_expl"] for r in rows.values())
    a["n_users"] = n
    a["valid_expl_per10"] = tot_expl / (K * n) if n else float("nan")
    a["users_zero_returned"] = sum(r["n_items"] == 0 for r in rows.values())
    a["users_lt10"] = sum(r["n_items"] < K for r in rows.values())
    a["concept_fraction_pooled_over_existing_expl"] = (tot_conc / tot_expl) if tot_expl else None
    a["n_existing_expl"] = tot_expl
    withe = [r for r in rows.values() if r["n_valid_expl"]]
    a["concept_fraction_macro_users_with_expl"] = (float(np.mean([r["n_concept_expl"] / r["n_valid_expl"] for r in withe]))
                                                  if withe else None)
    a["n_users_with_expl"] = len(withe)
    fam_tot = Counter()
    for r in rows.values():
        fam_tot.update(r["fam"])
    nf = sum(fam_tot.values())
    a["family_share_pooled"] = {f: (fam_tot[f] / nf if nf else None) for f in FAMILIES}
    ents, dist = [], []
    for r in withe:
        c = Counter(r["fam"]); m = sum(c.values())
        if m == 0:
            continue
        p = [v / m for v in c.values()]
        ents.append(-sum(x * math.log(x) for x in p) / math.log(len(FAMILIES)))
        dist.append(len(c))
    a["norm_entropy_macro_users_with_family"] = float(np.mean(ents)) if ents else None
    a["distinct_families_macro"] = float(np.mean(dist)) if dist else None
    pe = [v / nf for v in fam_tot.values()] if nf else []
    a["norm_entropy_pooled"] = (-sum(x * math.log(x) for x in pe) / math.log(len(FAMILIES))) if pe else None
    # candidate recall over held-out positives
    pos_total = sum(len(labels[u]) for u in rows)
    hit_total = sum(r["hits"] for r in rows.values())
    a["top10_hits_total"] = hit_total
    a["positives_total"] = pos_total
    return a


def availability(pred, labels, train_labels):
    """Concept availability in the beam (train items excluded): pair has >=1 concept path / pairs with a path."""
    pooled_pairs = pooled_conc = 0
    macro_u = []
    cand_hit = cand_pos = 0
    for uid, pos in labels.items():
        if not pos:
            continue
        tr = set(train_labels.get(uid, []))
        pairs = {pid: cs for pid, cs in pred.get(uid, {}).items() if pid not in tr}
        c = sum(any(has_concept(x[2]) for x in cs) for cs in pairs.values())
        pooled_pairs += len(pairs); pooled_conc += c
        if pairs:
            macro_u.append(c / len(pairs))
        cand_hit += sum(1 for p in pos if p in pairs); cand_pos += len(pos)
    return {"pooled": pooled_conc / pooled_pairs if pooled_pairs else None, "pooled_pairs": pooled_pairs,
            "pairs_with_concept": pooled_conc,
            "macro_users": float(np.mean(macro_u)) if macro_u else None, "n_users_with_pairs": len(macro_u),
            "candidate_recall_pooled": cand_hit / cand_pos if cand_pos else None, "candidate_hits": cand_hit,
            "positives": cand_pos}


def evaluate_beam(beam_path, tmp_dir, kg_args, split, assert_invariants=True):
    """Full M5 evaluation of one saved beam on split in {validation,test}. Returns dict + per-user rows."""
    emb = pickle.load(open(os.path.join(tmp_dir, "transe_embed.pkl"), "rb"))
    scores = np.dot(emb["user"] + emb[kg_args["interaction"]][0], emb["item"].T)
    train = pickle.load(open(os.path.join(tmp_dir, "train_label.pkl"), "rb"))
    labels = pickle.load(open(os.path.join(tmp_dir, f"{split}_label.pkl"), "rb"))
    results = pickle.load(open(beam_path, "rb"))
    sl = kg_args["self_loop"]
    pred, frozen = build(results, scores, labels, train)
    out, rows = {}, {}
    for sel in ("S0", "S3"):
        rows[sel] = per_user_rows(pred, frozen, labels, sel, sl)
        out[sel] = aggregate(rows[sel], labels)
    if assert_invariants:
        for u in rows["S0"]:
            assert rows["S0"][u]["ids"] == rows["S3"][u]["ids"], "S3 changed item list/order"
            for k in ("precision", "recall", "ndcg", "hr", "mrr"):
                assert rows["S0"][u][k] == rows["S3"][u][k]
            assert rows["S3"][u]["n_concept_expl"] >= rows["S0"][u]["n_concept_expl"]
    out["availability"] = availability(pred, labels, train)
    out["gain_S3_minus_S0_concept_at10"] = out["S3"]["concept_at10"] - out["S0"]["concept_at10"]
    out["eligible_users"] = sorted(rows["S0"])
    return out, rows
