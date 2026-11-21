"""Spike B: same-item post-hoc path selection on a saved UPGPR beam.

Fixed question: for the exact same user / item / topK as the baseline, does a
different explanation path (chosen only from the beam of that same pair)
satisfy a target relation property more often?

Three stages are kept strictly separate:
  1. candidate generation  -> the saved beam (never touched here)
  2. item ranking          -> reconstructed once from the legacy evaluator logic, then FROZEN
  3. path-only selection   -> selectors S0..SL pick a path per frozen (uid, item) pair;
                              items are never re-sorted afterwards.

Legacy branch (sum_prob=false) is reproduced exactly:
  * saved probs are already shifted by +1 (batch_beam_search: probs + actmask); we never add 1.
  * legacy path score = reduce(mul, saved probs)  (evaluator's `path_prob`).
  * best path per pair = sorted(by path_prob, reverse)[0]  (stable -> first in beam on ties).
  * items sorted by (dot(user+enroll,item), path_prob) reverse, stable; top10.

Pre-registered primary targets (fixed before any real beam was read):
  S1 teacher  : path contains relation `item_teacher`
  S2 school   : path contains relation `item_school`
  S3 concept  : path contains relation `item_concept`
  SL shortest : path has the minimal effective length (non self-loop hops) in its own beam
Binary pairwise targets, no combined objective.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import pickle
import re
import subprocess
import sys
import time
from collections import Counter
from functools import reduce

import numpy as np

TOPK = 10
BOOTSTRAP_SEED = 20240131
BOOTSTRAP_N = 2000
TOL = 1e-12
SELECTORS = ["S0", "S1", "S2", "S3", "SL"]
TARGET_REL = {"S1": "item_teacher", "S2": "item_school", "S3": "item_concept"}
REL_NAMES = {"teacher": "item_teacher", "school": "item_school", "concept": "item_concept"}

# candidate = (score, prob, path, beam_idx)


# ----------------------------------------------------------------- path helpers
def relations(path):
    return [n[0] for n in path[1:]]


def effective_len(path, self_loop="self_loop"):
    return sum(1 for r in relations(path) if r != self_loop)


def has_self_loop(path, self_loop="self_loop"):
    return any(r == self_loop for r in relations(path))


def pattern_key(path):
    """Same key format as legacy `path_patterns`: '<rel>_<type>-->' per node.
    Start node is (self_loop, 'user', uid) in real beams; relations() skips it."""
    return "".join(f"{n[0]}_{n[1]}-->" for n in path)


# ----------------------------------------------------------------- legacy reconstruction
def collect_pred_paths(results, scores, test_labels):
    """Legacy step 1: per user, per item, candidate list in beam order."""
    pred_paths = {uid: {} for uid in test_labels}
    for beam_idx, (path, probs) in enumerate(zip(results["paths"], results["probs"])):
        if path[-1][1] != "item":
            continue
        uid = path[0][2]
        if uid not in pred_paths:
            continue
        pid = path[-1][2]
        pred_paths[uid].setdefault(pid, [])
        path_score = scores[uid][pid]
        path_prob = reduce(lambda x, y: x * y, probs)
        pred_paths[uid][pid].append((path_score, path_prob, path, beam_idx))
    return pred_paths


def best_paths_legacy(pred_paths, train_labels):
    """Legacy step 3a: drop train items, best-prob path per pair. Users w/o train labels skipped."""
    best = {}
    for uid in pred_paths:
        train_pids = set(train_labels.get(uid, []))
        if len(train_pids) == 0:
            continue
        best[uid] = []
        for pid in pred_paths[uid]:
            if pid in train_pids:
                continue
            s = sorted(pred_paths[uid][pid], key=lambda x: x[1], reverse=True)
            best[uid].append(s[0])
    return best


def freeze_ranking(best):
    """Legacy step 3b (sort_by='score'): returns uid -> list of best candidates, top10, best first."""
    frozen = {}
    for uid, cands in best.items():
        s = sorted(cands, key=lambda x: (x[0], x[1]), reverse=True)
        frozen[uid] = s[:TOPK]
    return frozen


def to_evaluator_format(ids_desc):
    """Evaluator expects ascending (smallest first); evaluate() reverses it back."""
    return ids_desc[::-1]


def audit_probs(results):
    allp = [p for ps in results["probs"] for p in ps]
    if not allp:
        return {"n": 0, "shifted": None}
    a = np.asarray(allp, dtype=np.float64)
    shifted = bool(a.min() >= 1.0 - 1e-6 and a.max() <= 2.0 + 1e-6)
    return {"n": int(a.size), "min": float(a.min()), "max": float(a.max()),
            "shifted_plus1": shifted,
            "note": "saved probs used as-is; legacy score = product of saved probs; +1 never re-applied"}


# ----------------------------------------------------------------- recommendation metrics (port of evaluate(), compute_all=True)
def rec_metrics(topk_matches, test_labels, min_courses=TOPK):
    """topk_matches: uid -> ids ascending (evaluator format). Returns all-user + valid-only, in percent."""
    keys = ("ndcg", "recall", "hit", "precision")
    valid = {k: [] for k in keys}
    allu = {k: [] for k in keys}
    per_user = {}
    n_invalid = 0
    for uid, rel_set in test_labels.items():
        preds = topk_matches.get(uid, [])
        is_invalid = uid not in topk_matches or len(preds) < min_courses
        if is_invalid:
            n_invalid += 1
        pred_list = preds[::-1]
        if len(pred_list) == 0:
            vals = dict(ndcg=0.0, recall=0.0, hit=0.0, precision=0.0)
        else:
            dcg, hit_num = 0.0, 0.0
            for i, p in enumerate(pred_list):
                if p in rel_set:
                    dcg += 1.0 / (math.log(i + 2) / math.log(2))
                    hit_num += 1
            idcg = 0.0
            for i in range(min(len(rel_set), len(pred_list))):
                idcg += 1.0 / (math.log(i + 2) / math.log(2))
            vals = dict(ndcg=dcg / idcg, recall=hit_num / len(rel_set),
                        hit=1.0 if hit_num > 0 else 0.0, precision=hit_num / len(pred_list))
            if not is_invalid:
                for k in keys:
                    valid[k].append(vals[k])
        for k in keys:
            allu[k].append(vals[k])
        per_user[uid] = (vals, not is_invalid)
    mean = lambda xs: float(np.mean(xs) * 100) if len(xs) else float("nan")
    return {
        "all_users": {k: mean(allu[k]) for k in keys},
        "valid_only": {k: mean(valid[k]) for k in keys},
        "n_users": len(test_labels), "n_valid": len(valid["ndcg"]), "n_invalid": n_invalid,
        "per_user": per_user,
    }


# ----------------------------------------------------------------- selectors
def own_property(selector, path, cands, self_loop="self_loop"):
    if selector in TARGET_REL:
        return TARGET_REL[selector] in relations(path)
    if selector == "SL":
        return effective_len(path, self_loop) == min(effective_len(c[2], self_loop) for c in cands)
    return None


def select_path(selector, cands, s0, self_loop="self_loop"):
    """cands: all beam candidates of the pair (beam order). s0: legacy-best candidate.
    Returns chosen candidate; always an element of cands. Deterministic."""
    if selector == "S0":
        return s0
    if own_property(selector, s0[2], cands, self_loop):
        return s0  # keep S0 if target already met
    if selector == "SL":
        # s0 not minimal: min effective length, then max legacy prob, then earliest beam idx
        return sorted(cands, key=lambda c: (effective_len(c[2], self_loop), -c[1], c[3]))[0]
    pref = [c for c in cands if own_property(selector, c[2], cands, self_loop)]
    if not pref:
        return s0
    return sorted(pref, key=lambda c: (-c[1], c[3]))[0]


# ----------------------------------------------------------------- aggregate stats
def entropy_bits(counter):
    n = sum(counter.values())
    if n == 0:
        return {"entropy_bits": None, "normalized": None, "n_patterns": 0}
    ps = [c / n for c in counter.values()]
    h = -sum(p * math.log2(p) for p in ps)
    k = len(ps)
    return {"entropy_bits": h, "normalized": (h / math.log2(k)) if k > 1 else 0.0, "n_patterns": k}


def list_diversity(paths):
    """Only for 4-node paths (user, historical item, shared entity, target item). Else None."""
    if not paths or any(len(p) != 4 for p in paths):
        return None
    hist = len({p[1][2] for p in paths})
    shared = len({(p[2][1], p[2][2]) for p in paths})
    n = len(paths)
    return {"hist_div": hist / n, "shared_div": shared / n}


def bootstrap_ci(diffs, seed=BOOTSTRAP_SEED, n=BOOTSTRAP_N):
    d = np.asarray(diffs, dtype=np.float64)
    if d.size == 0:
        return None
    rng = np.random.default_rng(seed)
    means = np.array([d[rng.integers(0, d.size, d.size)].mean() for _ in range(n)])
    return {"mean": float(d.mean()), "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
            "n_users": int(d.size), "seed": seed, "n_boot": n,
            "note": "user-level bootstrap over test users; NOT variance of training seeds"}


def run_selectors(frozen, pred_paths, self_loop="self_loop", sink=None):
    """Select per frozen pair for every selector. Asserts invariants. Returns (records, runtime)."""
    records, runtime = [], {}
    for sel in SELECTORS:
        t0 = time.perf_counter()
        for uid, top in frozen.items():
            for rank, s0 in enumerate(top, start=1):
                pid = s0[2][-1][2]
                cands = pred_paths[uid][pid]
                ch = select_path(sel, cands, s0, self_loop)
                assert any(ch is c for c in cands), "selected path not in original same-pair beam"
                assert ch[2][-1][2] == pid and ch[2][0][2] == uid
                records.append({
                    "uid": uid, "cid": pid, "rank": rank, "selector": sel,
                    "path": ch[2], "beam_idx": ch[3],
                    "old_prop": own_property(sel, s0[2], cands, self_loop),
                    "new_prop": own_property(sel, ch[2], cands, self_loop),
                    "legacy_score": float(ch[1]), "s0_legacy_score": float(s0[1]),
                    "changed_path": ch is not s0,
                    "has_pref": (any(own_property(sel, c[2], cands, self_loop) for c in cands)
                                 if sel != "S0" else None),
                    "n_cands": len(cands),
                })
        runtime[sel] = time.perf_counter() - t0
    return records, runtime


def summarize(records, frozen, self_loop="self_loop"):
    out = {}
    by_sel = {s: [] for s in SELECTORS}
    for r in records:
        by_sel[r["selector"]].append(r)
    for sel, rs in by_sel.items():
        n = len(rs)
        if n == 0:
            out[sel] = {"n_pairs": 0}
            continue
        pat = Counter(pattern_key(r["path"]) for r in rs)
        rel_pres = {k: sum(v in relations(r["path"]) for r in rs) / n for k, v in REL_NAMES.items()}
        effs = [effective_len(r["path"], self_loop) for r in rs]
        d = {
            "n_pairs": n,
            "relation_presence": rel_pres,
            "effective_len_mean": float(np.mean(effs)),
            "self_loop_rate": sum(has_self_loop(r["path"], self_loop) for r in rs) / n,
            "pattern_distribution": {k: v / n for k, v in sorted(pat.items(), key=lambda kv: -kv[1])},
            "pattern_entropy": {**entropy_bits(pat), "base": 2,
                                "type_definition": "pattern = per-node '<relation>_<node_type>' sequence incl. self_loop (legacy path_patterns key)"},
            "scope": "topK final explanations of frozen top10 pairs (not all candidate pairs)",
        }
        if sel != "S0":
            hp = [r for r in rs if r["has_pref"]]
            d["oracle"] = {
                "own_target": TARGET_REL.get(sel, "min_effective_len"),
                "pairs_with_preferred_alt": len(hp) / n,
                "s0_already_has": sum(bool(r["old_prop"]) for r in rs) / n,
                "improvable": sum((not r["old_prop"]) and r["has_pref"] for r in rs) / n,
                "selector_changed_path": sum(r["changed_path"] for r in rs) / n,
                "selector_changed_property": sum(r["old_prop"] != r["new_prop"] for r in rs) / n,
                "achieved_target_rate": sum(bool(r["new_prop"]) for r in rs) / n,
                "ceiling_rate": len(hp) / n,
            }
            d["oracle"]["at_ceiling"] = abs(d["oracle"]["achieved_target_rate"] - d["oracle"]["ceiling_rate"]) <= TOL
        div = []
        byuser = {}
        for r in rs:
            byuser.setdefault(r["uid"], []).append(r)
        for rr in byuser.values():
            x = list_diversity([r["path"] for r in sorted(rr, key=lambda r: r["rank"])])
            if x:
                div.append(x)
        d["diversity"] = ({"hist_div_mean": float(np.mean([x["hist_div"] for x in div])),
                           "shared_div_mean": float(np.mean([x["shared_div"] for x in div])),
                           "n_lists": len(div),
                           "definition": "distinct historical items / distinct (type,id) shared entities per user list / list size; only lists whose paths are all 4-node"}
                          if div else None)
        out[sel] = d
    return out


def paired_bootstrap(records):
    """Per-user mean of own-target indicator, selector minus S0 (S0 evaluated on same property)."""
    res = {}
    for sel in SELECTORS[1:]:
        per = {}
        for r in records:
            if r["selector"] != sel:
                continue
            per.setdefault(r["uid"], []).append(int(bool(r["new_prop"])) - int(bool(r["old_prop"])))
        res[sel] = bootstrap_ci([float(np.mean(v)) for _, v in sorted(per.items())])
        # paired pair-level changes
        rs = [r for r in records if r["selector"] == sel]
        res[sel] = {"user_bootstrap_diff": res[sel],
                    "pairs_improved": sum((not r["old_prop"]) and r["new_prop"] for r in rs),
                    "pairs_worsened": sum(r["old_prop"] and (not r["new_prop"]) for r in rs),
                    "pairs_unchanged_prop": sum(r["old_prop"] == r["new_prop"] for r in rs)}
    return res


def beam_wide(pred_paths, self_loop="self_loop"):
    """Secondary only. All item-endpoint beam paths / all candidate pairs (NOT topK explanations)."""
    n_paths = 0
    rel = Counter()
    pair_any = Counter()
    n_pairs = 0
    for uid, d in pred_paths.items():
        for pid, cands in d.items():
            n_pairs += 1
            for k, v in REL_NAMES.items():
                if any(v in relations(c[2]) for c in cands):
                    pair_any[k] += 1
            for c in cands:
                n_paths += 1
                for k, v in REL_NAMES.items():
                    rel[k] += v in relations(c[2])
    return {"scope": "ALL candidate pairs / beam paths (not topK explanation metrics)",
            "n_paths": n_paths, "n_pairs": n_pairs,
            "path_share_with_relation": {k: rel[k] / max(n_paths, 1) for k in REL_NAMES},
            "pair_share_any_path_with_relation": {k: pair_any[k] / max(n_pairs, 1) for k in REL_NAMES}}


# ----------------------------------------------------------------- audits
def sha256(path, limit=None):
    if not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def audit_ids(results, user_embeds, item_embeds, kg_args):
    """Check node ids within embedding tables and hop types consistent with kg_relation."""
    bad_type, bad_range, n = 0, 0, 0
    kr = kg_args["kg_relation"]
    sl = kg_args["self_loop"]
    for path in results["paths"]:
        n += 1
        for (r0, t0, i0), (r1, t1, i1) in zip(path[:-1], path[1:]):
            exp = t0 if r1 == sl else kr.get(t0, {}).get(r1)
            if exp != t1:
                bad_type += 1
                break
        u, it = path[0][2], path[-1][2]
        if not (0 <= u < len(user_embeds)) or (path[-1][1] == "item" and not (0 <= it < len(item_embeds))):
            bad_range += 1
    return {"n_paths": n, "paths_with_type_inconsistency": bad_type, "paths_with_id_out_of_range": bad_range,
            "ok": bad_type == 0 and bad_range == 0}


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


# ----------------------------------------------------------------- orchestration
def jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(type(o))


def run(config, path_file, out_dir, epochs=None, smoke=None, logs_json=None, argv=None):
    cfg = json.load(open(config))
    kg_args = cfg["KG_ARGS"]
    ta = cfg["TEST_AGENT"]
    tmp_dir = ta["tmp_dir"]
    if not os.path.isfile(path_file):
        print(f"BLOCKED: beam file not found: {path_file}. Not training a substitute.")
        return 3
    os.makedirs(out_dir, exist_ok=True)
    m = re.search(r"epoch_(\d+)", os.path.basename(path_file))
    ep = epochs if epochs is not None else (int(m.group(1)) if m else None)
    full_ep = cfg["TRAIN_AGENT"]["epochs"]
    is_smoke = smoke if smoke is not None else (ep is None or ep < full_ep)

    embeds = pickle.load(open(f"{tmp_dir}/transe_embed.pkl", "rb"))
    user_e, item_e = embeds["user"], embeds["item"]
    enroll_e = embeds[kg_args["interaction"]][0]
    scores = np.dot(user_e + enroll_e, item_e.T)
    train_labels = pickle.load(open(f"{tmp_dir}/train_label.pkl", "rb"))
    test_labels = pickle.load(open(f"{tmp_dir}/test_label.pkl", "rb"))
    results = pickle.load(open(path_file, "rb"))
    sl = kg_args["self_loop"]

    pred_paths = collect_pred_paths(results, scores, test_labels)
    best = best_paths_legacy(pred_paths, train_labels)
    frozen = freeze_ranking(best)
    frozen_ids = {u: [c[2][-1][2] for c in top] for u, top in frozen.items()}
    base_eval = {u: to_evaluator_format(ids) for u, ids in frozen_ids.items()}
    base_m = rec_metrics(base_eval, test_labels)

    recon = {"probs_audit": audit_probs(results), "id_audit": audit_ids(results, user_e, item_e, kg_args),
             "n_beam_paths": len(results["paths"]), "n_test_users": len(test_labels),
             "n_users_with_frozen_list": len(frozen_ids),
             "n_users_lt10": sum(len(v) < TOPK for v in frozen_ids.values()),
             "n_users_no_train_labels_skipped": sum(1 for u in test_labels if u not in frozen_ids),
             "metrics_reconstructed": {k: base_m[k] for k in ("all_users", "valid_only", "n_users", "n_valid", "n_invalid")},
             "sort": "key=(score, path_prob) reverse, stable; legacy sum_prob=false"}
    if logs_json and os.path.isfile(logs_json):
        logged = json.load(open(logs_json))
        recon["vs_logs"] = {k: {"logged": logged[k], "reconstructed": base_m["all_users"][k],
                                "abs_diff": abs(logged[k] - base_m["all_users"][k])} for k in logged if k in base_m["all_users"]}
    else:
        recon["vs_logs"] = None

    records, runtime = run_selectors(frozen, pred_paths, sl)

    # invariants: exact ordered ids and metrics identical to baseline for each selector
    inv = {}
    for sel in SELECTORS:
        got = {}
        for r in records:
            if r["selector"] == sel:
                got.setdefault(r["uid"], []).append((r["rank"], r["cid"]))
        for u, ids in frozen_ids.items():
            assert [c for _, c in sorted(got[u])] == ids, f"order changed {sel} uid={u}"
        mm = rec_metrics({u: [c for _, c in sorted(v)][::-1] for u, v in got.items()}, test_labels)
        diff = max(abs(mm[c][k] - base_m[c][k]) for c in ("all_users", "valid_only") for k in base_m[c])
        assert diff <= TOL, f"metrics differ for {sel}: {diff}"
        inv[sel] = {"order_identical": True, "max_metric_abs_diff": diff}
    recon["selector_invariants"] = inv

    summ = summarize(records, frozen, sl)
    summary = {"smoke": is_smoke, "epochs": ep, "train_agent_epochs_config": full_ep,
               "claim_scope": ("SMOKE: feasibility only, no baseline/paper-reproduction claim" if is_smoke
                               else "full-epoch config run (still verify vs paper separately)"),
               "preregistered_targets": {**TARGET_REL, "SL": "min effective length in own beam"},
               "recommendation_metrics_unchanged_all_selectors": base_m["all_users"],
               "recommendation_metrics_valid_only": base_m["valid_only"],
               "explanations": summ, "paired": paired_bootstrap(records),
               "beam_wide_secondary": beam_wide(pred_paths, sl)}

    json.dump(recon, open(f"{out_dir}/baseline_reconstruction.json", "w"), indent=2, default=jsonable)
    json.dump(summary, open(f"{out_dir}/summary.json", "w"), indent=2, default=jsonable)
    with open(f"{out_dir}/selected_paths.jsonl", "w") as f:
        for r in records:
            row = {"uid": r["uid"], "cid": r["cid"], "rank": r["rank"], "selector": r["selector"],
                   "path": r["path"], "old_property": r["old_prop"], "new_property": r["new_prop"],
                   "legacy_score": r["legacy_score"], "s0_legacy_score": r["s0_legacy_score"],
                   "changed_path": r["changed_path"]}
            f.write(json.dumps(row, default=jsonable) + "\n")
    write_per_user(f"{out_dir}/per_user.csv", records, base_m, frozen_ids)
    write_report(f"{out_dir}/report.md", recon, summary)
    manifest = {
        "commit": git_commit(), "seed_bootstrap": BOOTSTRAP_SEED, "bootstrap_n": BOOTSTRAP_N,
        "train_seed_config": cfg["TRAIN_AGENT"].get("seed"), "epochs": ep, "smoke": is_smoke,
        "commands": [" ".join(argv or sys.argv)], "selector_runtime_sec": runtime,
        "config_path": config, "effective_test_agent_config": ta,
        "hashes": {"config": sha256(config), "beam": sha256(path_file),
                   "embeddings": sha256(f"{tmp_dir}/transe_embed.pkl"),
                   "train_label": sha256(f"{tmp_dir}/train_label.pkl"),
                   "test_label": sha256(f"{tmp_dir}/test_label.pkl"),
                   "kg": sha256(f"{tmp_dir}/kg.pkl"),
                   "checkpoint": sha256(os.path.join(os.path.dirname(path_file), f"tmp_policy_model_epoch_{ep}.ckpt")),
                   "id_map_files": {n: sha256(os.path.join(ta["data_dir"], fn))
                                    for n, fn in kg_args.get("entity_files", {}).items()}},
    }
    json.dump(manifest, open(f"{out_dir}/manifest.json", "w"), indent=2, default=jsonable)
    print(f"wrote outputs to {out_dir}")
    return 0


def write_per_user(path, records, base_m, frozen_ids):
    by = {}
    for r in records:
        by.setdefault((r["uid"], r["selector"]), []).append(r)
    cols = ["uid", "n_final", "valid10", "ndcg", "recall", "hit", "precision"]
    for s in SELECTORS:
        cols += [f"{s}_new_prop_rate", f"{s}_changed_paths", f"{s}_eff_len_mean"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for uid, ids in frozen_ids.items():
            vals, valid = base_m["per_user"][uid]
            row = [uid, len(ids), int(valid), vals["ndcg"], vals["recall"], vals["hit"], vals["precision"]]
            for s in SELECTORS:
                rs = by[(uid, s)]
                row += [float(np.mean([bool(r["new_prop"]) for r in rs])) if s != "S0" else "",
                        sum(r["changed_path"] for r in rs),
                        float(np.mean([effective_len(r["path"]) for r in rs]))]
            w.writerow(row)


def write_report(path, recon, summary):
    L = ["# Spike B: same-item post-hoc path selection", "",
         f"**Scope: {summary['claim_scope']}** (epochs={summary['epochs']}, config train epochs={summary['train_agent_epochs_config']})", "",
         "Path-only selection on the frozen baseline top10 of each user. Not item reranking. Items are never re-sorted.", "",
         "## Baseline reconstruction",
         f"- probs audit: {recon['probs_audit']}", f"- id audit: {recon['id_audit']}",
         f"- users: {recon['n_test_users']}, with list: {recon['n_users_with_frozen_list']}, <10 preds: {recon['n_users_lt10']}",
         f"- metrics (all users, %): {recon['metrics_reconstructed']['all_users']}",
         f"- metrics (valid-only, %): {recon['metrics_reconstructed']['valid_only']}",
         f"- vs logs: {recon['vs_logs']}", "",
         "## Ranking invariants", f"{recon['selector_invariants']}", "",
         "## Explanation metrics on topK final explanations", "",
         "| selector | n pairs | teacher | school | concept | eff len | self-loop | pattern H (bits) | H norm |", "|---|---|---|---|---|---|---|---|---|"]
    for s, d in summary["explanations"].items():
        rp = d["relation_presence"]
        e = d["pattern_entropy"]
        L.append(f"| {s} | {d['n_pairs']} | {rp['teacher']:.4f} | {rp['school']:.4f} | {rp['concept']:.4f} | "
                 f"{d['effective_len_mean']:.4f} | {d['self_loop_rate']:.4f} | {e['entropy_bits']:.4f} | {e['normalized']:.4f} |")
    L += ["", "Pattern entropy: base 2, pattern = per-node `<relation>_<node_type>` sequence (legacy key).", "",
          "## Oracle availability (fixed topK pairs)", "",
          "| selector | has pref alt | S0 already | improvable | changed path | changed prop | achieved | ceiling | at ceiling |",
          "|---|---|---|---|---|---|---|---|---|"]
    for s, d in summary["explanations"].items():
        o = d.get("oracle")
        if o:
            L.append(f"| {s} | {o['pairs_with_preferred_alt']:.4f} | {o['s0_already_has']:.4f} | {o['improvable']:.4f} | "
                     f"{o['selector_changed_path']:.4f} | {o['selector_changed_property']:.4f} | "
                     f"{o['achieved_target_rate']:.4f} | {o['ceiling_rate']:.4f} | {o['at_ceiling']} |")
    L += ["", "## Paired changes + user bootstrap (selector minus S0, own target)", "",
          f"Seed {BOOTSTRAP_SEED}, {BOOTSTRAP_N} resamples. User-level uncertainty only, NOT training-seed variance.", "",
          json.dumps(summary["paired"], indent=2, default=jsonable), "",
          "## Beam-wide (secondary, NOT topK explanation metrics)", json.dumps(summary["beam_wide_secondary"], indent=2), "",
          "## Interpretation limits",
          "- If a property does not change because no alternative exists, that is an availability bottleneck of the beam. Reward shaping could find paths outside the beam, so this does not show shaping is useless.",
          "- High post-hoc headroom means a strong baseline, not proof that the reward has no value.", ""]
    open(path, "w").write("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/UPGPR/mooc.json")
    ap.add_argument("--path_file", required=True, help="policy_paths_epoch_N.pkl")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--smoke", action="store_true", default=None)
    ap.add_argument("--logs_json", default=None, help="legacy run_0.json to compare")
    a = ap.parse_args()
    sys.exit(run(a.config, a.path_file, a.out_dir, a.epochs, a.smoke, a.logs_json, sys.argv))


if __name__ == "__main__":
    main()
