"""M5 concept funnel on the exact M5 top-10 cohorts, using the Spike A matched cache.

Stages per (user,item) pair of an arm's top-10 list (train items already excluded):
  full graph (eval horizon 3) -> after env pruning (max_acts 250) -> M5 beam -> S0 selected / S3 selected.
Full/pruned stages are policy independent (A cache, same KG/embeddings/eval3/pruning). Beam/S0/S3 come from the
M5 raw beam of the arm. Pairs absent from the cache are reported as `uncovered` (unknown), never as `no path`.
Concept = any path with an `item_concept` relation (same as m5_metrics.has_concept).
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m5_metrics as mm  # noqa: E402
import spike_b_posthoc as sb  # noqa: E402

CACHE_COLS = ["uid", "cid", "status", "full_paths", "full_concept", "pruned_paths", "pruned_concept"]


def load_cache(a_dir):
    """Union of topk_pairs.csv and positive_pairs.csv of the A matched cache: (uid,cid) -> row tuple.
    Rows present in both files must agree (policy independent stages)."""
    import pandas as pd
    out = {}
    for name in ("topk_pairs.csv", "positive_pairs.csv"):
        d = pd.read_csv(os.path.join(a_dir, name), usecols=CACHE_COLS)
        assert not d.duplicated(["uid", "cid"]).any(), f"duplicate pairs in {name}"
        for r in d.itertuples(index=False):
            key = (int(r.uid), int(r.cid))
            val = (r.status, int(r.full_paths), int(r.full_concept), int(r.pruned_paths), int(r.pruned_concept))
            if key in out:
                assert out[key] == val, f"cache disagreement for {key}"
            out[key] = val
    return out


def arm_pairs(pred, frozen, self_loop="self_loop"):
    """(u,item) -> dict for each pair of the arm's frozen top-10 list."""
    out = {}
    for u, top in frozen.items():
        for s0 in top:
            pid = s0[2][-1][2]
            cands = pred[u][pid]
            ch0 = sb.select_path("S0", cands, s0, self_loop)
            ch3 = sb.select_path("S3", cands, s0, self_loop)
            out[(u, pid)] = {"beam_c": any(mm.has_concept(c[2]) for c in cands),
                             "s0_c": mm.has_concept(ch0[2]), "s3_c": mm.has_concept(ch3[2]),
                             "n_beam": len(cands)}
    return out


def bucket(row, cache_row):
    """Mutually exclusive S0 outcome bucket of one pair. uncovered is distinct from no_concept_in_graph."""
    if cache_row is None:
        return "uncovered"
    status, full_paths, full_c, pruned_paths, pruned_c = cache_row
    if status != "reachable" or full_paths == 0:
        return "no_path_in_graph"
    if full_c == 0:
        return "no_concept_path_in_graph"
    if pruned_c == 0:
        return "pruning_loss"
    if not row["beam_c"]:
        return "beam_miss"
    if not row["s0_c"]:
        return "selector_miss_s0"
    return "selected_s0"


BUCKETS = ["uncovered", "no_path_in_graph", "no_concept_path_in_graph", "pruning_loss", "beam_miss",
           "selector_miss_s0", "selected_s0"]


def _msd(xs):
    import numpy as np
    xs = list(xs)
    return {"per_seed": xs, "mean": float(np.mean(xs)), "sd_sample": float(np.std(xs, ddof=1)) if len(xs) > 1 else None}


def summarize(F):
    """F = funnel.json content. Slot-level decomposition (denominator 10 x eligible users, same for both arms;
    uncovered pairs included because beam/S0/S3 are known for every pair) and covered-cohort stage shares.

    A = S3 concept@10 = share of top-10 slots whose pair has >=1 concept path in the beam (availability ceiling);
    R = S0 concept@10 (realised without the selector). Identities (exact):
      total = A_shaped - R_control = (A_shaped - A_control) + (A_control - R_control)
      R_shaped - R_control = (A_shaped - A_control) - [(A_shaped-R_shaped) - (A_control-R_control)]."""
    slots = F["slots"]
    dec = {k: [] for k in ("A_control", "R_control", "A_shaped", "R_shaped", "total", "availability_reward",
                           "selector_control", "selector_shaped", "interaction", "reward_under_s0")}
    stages = {}
    for s, r in F["seeds"].items():
        a = {arm: (r[arm]["all"]["s3_c_all"] / slots, r[arm]["all"]["s0_c_all"] / slots) for arm in ("control", "shaped")}
        (Ac, Rc), (As, Rs) = a["control"], a["shaped"]
        for k, v in (("A_control", Ac), ("R_control", Rc), ("A_shaped", As), ("R_shaped", Rs), ("total", As - Rc),
                     ("availability_reward", As - Ac), ("selector_control", Ac - Rc), ("selector_shaped", As - Rs),
                     ("interaction", (As - Rs) - (Ac - Rc)), ("reward_under_s0", Rs - Rc)):
            dec[k].append(v)
        for arm in ("control", "shaped"):
            for sc in ("all", "common", "only"):
                f = r[arm][sc]
                c = f["covered"]
                st = stages.setdefault(f"{arm}_{sc}", {k: [] for k in ("pairs", "covered", "covered_share", "full", "pruned", "beam",
                                                                       "s0", "s3", "no_path", "no_concept_in_graph",
                                                                       "pruning_loss", "beam_miss", "selector_miss", "selected")})
                b = f["buckets"]
                for k, v in (("pairs", f["pairs"]), ("covered", c), ("covered_share", c / f["pairs"]),
                             ("full", f["full_c"] / c), ("pruned", f["pruned_c"] / c), ("beam", f["beam_c"] / c),
                             ("s0", f["s0_c"] / c), ("s3", f["s3_c"] / c), ("no_path", b["no_path_in_graph"] / c),
                             ("no_concept_in_graph", b["no_concept_path_in_graph"] / c), ("pruning_loss", b["pruning_loss"] / c),
                             ("beam_miss", b["beam_miss"] / c), ("selector_miss", b["selector_miss_s0"] / c),
                             ("selected", b["selected_s0"] / c)):
                    st[k].append(v)
    return {"decomposition_slots": {k: _msd(v) for k, v in dec.items()},
            "stages_of_covered": {k: {m: _msd(v) for m, v in d.items()} for k, d in stages.items()}}


def funnel(pairs, cache):
    """Counts over `pairs` (dict from arm_pairs). Covered-only stage counts plus violation counts."""
    n = len(pairs)
    f = {"pairs": n, "covered": 0, "full_c": 0, "pruned_c": 0, "beam_c": 0, "s0_c": 0, "s3_c": 0,
         "beam_c_all": 0, "s0_c_all": 0, "s3_c_all": 0, "buckets": {b: 0 for b in BUCKETS},
         "violations": {"s0_not_in_beam": 0, "s3_ne_beam": 0, "beam_without_pruned": 0, "beam_without_full": 0}}
    for k, r in pairs.items():
        cr = cache.get(k)
        f["beam_c_all"] += r["beam_c"]; f["s0_c_all"] += r["s0_c"]; f["s3_c_all"] += r["s3_c"]
        f["buckets"][bucket(r, cr)] += 1
        if r["s0_c"] and not r["beam_c"]:
            f["violations"]["s0_not_in_beam"] += 1
        if r["s3_c"] != r["beam_c"]:
            f["violations"]["s3_ne_beam"] += 1
        if cr is None:
            continue
        f["covered"] += 1
        f["full_c"] += cr[2] > 0; f["pruned_c"] += cr[4] > 0
        f["beam_c"] += r["beam_c"]; f["s0_c"] += r["s0_c"]; f["s3_c"] += r["s3_c"]
        if r["beam_c"] and cr[4] == 0:
            f["violations"]["beam_without_pruned"] += 1
        if r["beam_c"] and cr[2] == 0:
            f["violations"]["beam_without_full"] += 1
    return f
