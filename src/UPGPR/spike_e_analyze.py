"""Spike E analysis: stage-specific metrics, same-pair comparison, summary.json.

python src/UPGPR/spike_e_analyze.py --config <effective_config.json> --root tmp/spike_e_mooc_t4e3

Reads eval_{init,control,shaped}/ (beam.pkl, spike_b/summary.json, ...) and train logs.
Writes <root>/analysis/{same_pair_comparison.csv, branch_stats.json}, <root>/summary.json.
Pair-weighted = pooled over pairs; user-mean = mean over users of per-user fractions (users with >=1 pair in cohort).
"""
from __future__ import absolute_import, division, print_function

import argparse
import csv
import json
import os
import pickle
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spike_b_posthoc as sb  # noqa: E402

THRESH_PP = 2.0


def has_concept(paths):
    return any("item_concept" in [e[0] for e in p[2][1:]] for p in paths)


def load_branch(cfg, root, name):
    ta = cfg["TEST_AGENT"]
    embeds = pickle.load(open(f"{ta['tmp_dir']}/transe_embed.pkl", "rb"))
    scores = np.dot(embeds["user"] + embeds[cfg["KG_ARGS"]["interaction"]][0], embeds["item"].T)
    train = pickle.load(open(f"{ta['tmp_dir']}/train_label.pkl", "rb"))
    test = pickle.load(open(f"{ta['tmp_dir']}/test_label.pkl", "rb"))
    res = pickle.load(open(f"{root}/eval_{name}/beam.pkl", "rb"))
    pp = sb.collect_pred_paths(res, scores, test)
    best = sb.best_paths_legacy(pp, train)
    frozen = sb.freeze_ranking(best)
    # candidates = non-train items in beam (same filter as the evaluator)
    tsets = {u: set(v) for u, v in train.items()}
    cands = {u: {pid: v for pid, v in d.items() if pid not in tsets[u]} for u, d in pp.items() if train.get(u)}
    topk = {u: [c[2][-1][2] for c in top] for u, top in frozen.items()}
    return {"pp": cands, "topk": topk, "train": train, "test": test}


def branch_stats(b):
    test, cands = b["test"], b["pp"]
    n_pos = hit = 0
    per_user = []
    for u, pos in test.items():
        pos_new = [p for p in pos if p not in set(b["train"].get(u, []))]
        h = sum(1 for p in pos_new if p in cands.get(u, {}))
        n_pos += len(pos_new); hit += h
        if pos_new:
            per_user.append(h / len(pos_new))
    ncand = [len(cands.get(u, {})) for u in test]
    # concept availability over ALL candidate pairs in the beam
    tot = con = 0
    for u, d in cands.items():
        for pid, v in d.items():
            tot += 1; con += has_concept(v)
    return {"candidate_recall_pair_weighted": hit / n_pos if n_pos else None,
            "candidate_recall_user_mean": float(np.mean(per_user)) if per_user else None,
            "test_positive_pairs_denominator": n_pos, "users_denominator": len(per_user),
            "mean_candidates_per_user": float(np.mean(ncand)),
            "all_candidate_pairs": tot, "concept_available_all_candidate_pairs": con / tot if tot else None}


def cat(c_has, s_has):
    return {(False, True): "control_no_concept_shaped_has", (True, False): "control_has_shaped_lost",
            (True, True): "both_have", (False, False): "both_lack"}[(c_has, s_has)]


def same_pair(ctl, shp, root):
    rows = []
    cohorts = {}
    # all common candidate pairs
    common, top_i, top_u = {}, {}, {}
    users = sorted(set(ctl["pp"]) & set(shp["pp"]))
    for u in users:
        cc, sc = ctl["pp"][u], shp["pp"][u]
        common[u] = [p for p in cc if p in sc]
        tc, ts = set(ctl["topk"].get(u, [])), set(shp["topk"].get(u, []))
        top_i[u] = sorted(tc & ts)
        top_u[u] = sorted(tc | ts)
    cohorts = {"common_candidate_pairs": common, "topK_intersection": top_i, "topK_union": top_u}
    out = {}
    for cname, cohort in cohorts.items():
        counts = {k: 0 for k in ("control_no_concept_shaped_has", "control_has_shaped_lost", "both_have", "both_lack",
                                 "item_beam_missing_in_control", "item_beam_missing_in_shaped", "item_beam_missing_in_both")}
        per_user = {}
        for u, pids in cohort.items():
            for pid in pids:
                in_c, in_s = pid in ctl["pp"][u], pid in shp["pp"][u]
                if in_c and in_s:
                    k = cat(has_concept(ctl["pp"][u][pid]), has_concept(shp["pp"][u][pid]))
                elif in_s:
                    k = "item_beam_missing_in_control"
                elif in_c:
                    k = "item_beam_missing_in_shaped"
                else:
                    k = "item_beam_missing_in_both"
                counts[k] += 1
                per_user.setdefault(u, {}).setdefault(k, 0)
                per_user[u][k] += 1
        n = sum(counts.values())
        nu = len(per_user)
        for k, v in counts.items():
            um = float(np.mean([pu.get(k, 0) / sum(pu.values()) for pu in per_user.values()])) if nu else None
            rows.append({"cohort": cname, "category": k, "n_pairs": v, "pair_weighted_frac": v / n if n else None,
                         "user_mean_frac": um, "cohort_pairs_denominator": n, "cohort_users_denominator": nu})
        out[cname] = {"counts": counts, "n_pairs": n, "n_users": nu}
    # candidate-set / top-K overlap
    jac_c, jac_t, ov_t = [], [], []
    for u in users:
        a, b = set(ctl["pp"][u]), set(shp["pp"][u])
        if a | b:
            jac_c.append(len(a & b) / len(a | b))
        ta, tb = set(ctl["topk"].get(u, [])), set(shp["topk"].get(u, []))
        if ta | tb:
            jac_t.append(len(ta & tb) / len(ta | tb))
            ov_t.append(len(ta & tb) / max(1, min(len(ta), len(tb))))
    out["overlap"] = {"candidate_set_jaccard_user_mean": float(np.mean(jac_c)),
                      "topK_jaccard_user_mean": float(np.mean(jac_t)), "topK_overlap_frac_user_mean": float(np.mean(ov_t)),
                      "users": len(users)}
    os.makedirs(f"{root}/analysis", exist_ok=True)
    with open(f"{root}/same_pair_comparison.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    return out


def spike_b_block(root, name):
    s = json.load(open(f"{root}/eval_{name}/spike_b/summary.json"))
    r = json.load(open(f"{root}/eval_{name}/spike_b/baseline_reconstruction.json"))
    ex = s["explanations"]
    blk = {"metrics_all_users": s["recommendation_metrics_unchanged_all_selectors"],
           "n_users_lt10_predictions": r["n_users_lt10"],
           "vs_logs_max_abs_diff": max(v["abs_diff"] for v in r["vs_logs"].values()) if r["vs_logs"] else None,
           "selector_invariants": r["selector_invariants"], "n_pairs_topK": ex["S0"]["n_pairs"]}
    for sel in ("S0", "S3"):
        e = ex[sel]
        blk[sel] = {"concept_presence_selected": e["relation_presence"]["concept"],
                    "teacher_presence": e["relation_presence"]["teacher"], "school_presence": e["relation_presence"]["school"],
                    "effective_len_mean": e["effective_len_mean"], "self_loop_rate": e["self_loop_rate"],
                    "pattern_entropy": e["pattern_entropy"], "pattern_distribution": e["pattern_distribution"]}
    o = ex["S3"]["oracle"]
    blk["S3"]["oracle"] = o
    blk["concept_beam_availability_own_topK_pairs"] = o["ceiling_rate"]  # ceiling = pairs with >=1 concept path in own beam
    blk["S3_selector_gain_pp"] = 100 * (ex["S3"]["relation_presence"]["concept"] - ex["S0"]["relation_presence"]["concept"])
    for k in ("runtime", "bootstrap", "paired_bootstrap"):
        if k in s:
            blk[k] = s[k]
    return blk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--root", required=True)
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    root = a.root
    B = {n: spike_b_block(root, n) for n in ("init", "control", "shaped")}
    ctl, shp, ini = (load_branch(cfg, root, n) for n in ("control", "shaped", "init"))
    st = {"control": branch_stats(ctl), "shaped": branch_stats(shp), "init": branch_stats(ini)}
    sp = same_pair(ctl, shp, root)
    logs = {n: json.load(open(f"{root}/{n}/train_log.json")) for n in ("control", "shaped")}
    meta = {n: json.load(open(f"{root}/{n}/train_meta.json")) for n in ("control", "shaped")}
    rt = {}
    for n in ("init", "control", "shaped"):
        try:
            m = re.search(r"beam seconds ([\d.]+)", open(f"{root}/eval_{n}/beam.log").read())
            rt[n] = {"beam_seconds": float(m.group(1)) if m else None, **json.load(open(f"{root}/eval_{n}/eval_runtime.json"))}
        except Exception as e:  # pragma: no cover
            rt[n] = {"error": str(e)}
    for n in ("control", "shaped"):
        rt[n]["train_seconds"] = sum(l["wall_time_s"] for l in logs[n])
    c, s = B["control"], B["shaped"]
    pp = lambda x: 100 * x
    d_s0 = pp(s["S0"]["concept_presence_selected"] - c["S0"]["concept_presence_selected"])
    d_s3 = pp(s["S3"]["concept_presence_selected"] - c["S3"]["concept_presence_selected"])
    d_s0_vs_cs3 = pp(s["S0"]["concept_presence_selected"] - c["S3"]["concept_presence_selected"])
    d_ceil = pp(s["concept_beam_availability_own_topK_pairs"] - c["concept_beam_availability_own_topK_pairs"])
    ndcg_rel = (s["metrics_all_users"]["ndcg"] - c["metrics_all_users"]["ndcg"]) / c["metrics_all_users"]["ndcg"] * 100
    plumbing = {
        "lambda0_matches_baseline_reward": "tests/test_spike_e_reward.py (random walks vs commit 5334bf9, driver==train_agent weights)",
        "bonus_timing_gate_tests": "tests/test_spike_e_reward.py",
        "bonus_appears_in_training_shaped": [l["frac_terminal_B1"] for l in logs["shaped"]],
        "early_bonus_violations": {n: sum(l["early_bonus_violations"] for l in logs[n]) for n in logs},
        "nonfinite_loss": {n: sum(l["nonfinite_loss_count"] for l in logs[n]) for n in logs},
        "same_init_hash": meta["control"]["init_sha256"] == meta["shaped"]["init_sha256"],
        "metric_invariants_ok": all(v["selector_invariants"][k]["max_metric_abs_diff"] <= 1e-12 for v in B.values() for k in v["selector_invariants"]),
        "eval_vs_logs_max_abs_diff": {n: B[n]["vs_logs_max_abs_diff"] for n in B},
    }
    summary = {
        "scope": "one seed (23), 5 epochs per branch, train4/eval3; feasibility only. Not learner usefulness, not reproduction.",
        "lambda": {"control": 0.0, "shaped": 5.0},
        "branches": B, "candidate_stats": st, "same_pair": sp, "training": logs, "runtime_seconds": rt,
        "deltas_shaped_minus_control": {
            "S0_concept_presence_pp": d_s0, "S3_concept_presence_pp": d_s3,
            "shaped_S0_minus_control_S3_pp": d_s0_vs_cs3, "own_beam_concept_availability_pp": d_ceil,
            "ndcg_relative_pct": ndcg_rel,
            "note": "own-topK differences mix item-set change and path change; see same_pair for pure path availability"},
        "criteria": {
            "threshold_pp": THRESH_PP,
            "S0_beats_control_S0": d_s0 >= THRESH_PP, "S3_beats_control_S3": d_s3 >= THRESH_PP,
            "shaped_S0_beats_control_S3": d_s0_vs_cs3 >= THRESH_PP, "ceiling_up": d_ceil >= THRESH_PP,
            "ndcg_drop_over_10pct": ndcg_rel < -10.0},
        "plumbing": plumbing,
    }
    json.dump(summary, open(f"{root}/summary.json", "w"), indent=2, default=float)
    print(json.dumps({"deltas": summary["deltas_shaped_minus_control"], "criteria": summary["criteria"], "plumbing": {k: plumbing[k] for k in ("early_bonus_violations", "nonfinite_loss", "same_init_hash", "metric_invariants_ok")}}, indent=1))
    print(json.dumps(sp, indent=1))
    print(json.dumps(st, indent=1))


if __name__ == "__main__":
    main()
