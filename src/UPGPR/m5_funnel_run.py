"""M5 funnel packaging CLI. No training, no beam search; reads saved M5 beams, per-user CSVs and the Spike A cache.

  funnel         pair-level funnel per seed/arm over all / common / only-one-arm cohorts  -> tables/funnel.json
  summarize      slot-level arithmetic decomposition + covered-cohort stage shares        -> tables/funnel_summary.json
  slots          slot accounting from per-user CSVs (missing slots vs short-list users)   -> tables/slot_reconciliation.json
  path_validity  Spike A path_accepted (full + pruned) on every beam path of every top-10 pair
                                                                                          -> tables/funnel_path_validity.json

Arms: control = lambda 0, shaped = lambda* (root/summary.json), seeds 23/24/25, layout from m5_run.arm_dir.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m5_funnel as fn  # noqa: E402
import m5_metrics as mm  # noqa: E402
import m5_run  # noqa: E402

SEEDS = (23, 24, 25)
K = 10


def arms(root):
    s = json.load(open(os.path.join(root, "summary.json")))
    return {"control": 0.0, "shaped": float(s["lambda_star"])}


def adir(root, seed, lam):
    return m5_run.arm_dir(root, seed, lam)


def load_inputs(root, tmp_dir):
    import numpy as np
    cfg = json.load(open(os.path.join(root, "effective_config.json")))
    kg = cfg["KG_ARGS"]
    emb = pickle.load(open(os.path.join(tmp_dir, "transe_embed.pkl"), "rb"))
    scores = np.dot(emb["user"] + emb[kg["interaction"]][0], emb["item"].T)
    train = pickle.load(open(os.path.join(tmp_dir, "train_label.pkl"), "rb"))
    test = pickle.load(open(os.path.join(tmp_dir, "test_label.pkl"), "rb"))
    return cfg, kg, scores, train, test


def cmd_funnel(a):
    cfg, kg, scores, train, test = load_inputs(a.root, a.tmp_dir)
    cache = fn.load_cache(a.cache_dir)
    arm_l = arms(a.root)
    users = sum(1 for p in test.values() if p)
    slots = K * users
    out = {"cache_dir": a.cache_dir, "cache_pairs": len(cache), "users": users, "slots": slots, "seeds": {}}
    for s in SEEDS:
        P, ev = {}, {}
        t0 = time.time()
        for arm, lam in arm_l.items():
            d = adir(a.root, s, lam)
            res = pickle.load(open(os.path.join(d, "beam_test", "beam.pkl"), "rb"))
            pred, frozen = mm.build(res, scores, test, train)
            P[arm] = fn.arm_pairs(pred, frozen, kg["self_loop"])
            ev[arm] = json.load(open(os.path.join(d, "eval_test.json")))
            del res, pred, frozen
        com = set(P["control"]) & set(P["shaped"])
        r = {"n_common": len(com)}
        for arm, allp in P.items():
            r[arm] = {"all": fn.funnel(allp, cache),
                      "common": fn.funnel({k: allp[k] for k in com}, cache),
                      "only": fn.funnel({k: v for k, v in allp.items() if k not in com}, cache)}
            tie = {"s0_slots": r[arm]["all"]["s0_c_all"] / slots, "s0_eval": ev[arm]["S0"]["concept_at10"],
                   "s3_slots": r[arm]["all"]["s3_c_all"] / slots, "s3_eval": ev[arm]["S3"]["concept_at10"]}
            assert abs(tie["s0_slots"] - tie["s0_eval"]) < 1e-9 and abs(tie["s3_slots"] - tie["s3_eval"]) < 1e-9, tie
            r[arm]["tie"] = tie
        r["time_s"] = round(time.time() - t0)
        out["seeds"][str(s)] = r
        print("seed", s, "done", r["time_s"], "s", flush=True)
    os.makedirs(os.path.join(a.root, "tables"), exist_ok=True)
    json.dump(out, open(os.path.join(a.root, "tables", "funnel.json"), "w"), indent=1)


def cmd_summarize(a):
    F = json.load(open(os.path.join(a.root, "tables", "funnel.json")))
    out = fn.summarize(F)
    json.dump(out, open(os.path.join(a.root, "tables", "funnel_summary.json"), "w"), indent=1)
    d = out["decomposition_slots"]
    for k, v in d.items():
        print(f"{k:22s} mean {100 * v['mean']:7.2f} pp  per_seed {[round(100 * x, 2) for x in v['per_seed']]}")


def cmd_slots(a):
    arm_l = arms(a.root)
    out = {"slots_per_seed": None, "per_seed": {}}
    for s in SEEDS:
        for arm, lam in arm_l.items():
            d = adir(a.root, s, lam)
            res = {}
            for sel in ("S0", "S3"):
                rows = list(csv.DictReader(open(os.path.join(d, f"per_user_test_{sel}.csv"))))
                n = len(rows)
                n_items = [int(r["n_items"]) for r in rows]
                nv = [int(r["n_valid_expl"]) for r in rows]
                nc = [int(r["n_concept_expl"]) for r in rows]
                assert nv == n_items, "explanation count differs from returned item count"
                slots = K * n
                returned = sum(n_items)
                short = [x for x in n_items if x < K]
                res[sel] = {"users": n, "slots": slots, "returned_slots": returned, "missing_slots": slots - returned,
                            "users_lt10": len(short), "users_zero": sum(x == 0 for x in n_items),
                            "missing_slots_in_short_users": sum(K - x for x in short),
                            "concept_slots": sum(nc), "returned_non_concept_slots": returned - sum(nc),
                            "no_concept_slots": slots - sum(nc),
                            "share_concept": sum(nc) / slots, "share_missing": (slots - returned) / slots,
                            "share_returned_non_concept": (returned - sum(nc)) / slots}
                assert res[sel]["missing_slots"] == res[sel]["missing_slots_in_short_users"]
                assert abs(res[sel]["share_concept"] + res[sel]["share_missing"] + res[sel]["share_returned_non_concept"] - 1) < 1e-12
            assert res["S0"]["returned_slots"] == res["S3"]["returned_slots"], "S3 changed list length"
            out["per_seed"][f"{s}_{arm}"] = res
            out["slots_per_seed"] = res["S3"]["slots"]
            r3 = res["S3"]
            print(f"seed {s} {arm:8s} slots {r3['slots']} returned {r3['returned_slots']} missing {r3['missing_slots']} "
                  f"users_lt10 {r3['users_lt10']} (zero {r3['users_zero']}) S3 concept {r3['concept_slots']} "
                  f"returned_non_concept {r3['returned_non_concept_slots']} no_concept {r3['no_concept_slots']} "
                  f"({100 * r3['share_missing']:.2f}% missing + {100 * r3['share_returned_non_concept']:.2f}% non-concept)")
    # cross-check with funnel.json pair counts when present
    fp = os.path.join(a.root, "tables", "funnel.json")
    if os.path.isfile(fp):
        F = json.load(open(fp))
        for s in SEEDS:
            for arm in arm_l:
                assert F["seeds"][str(s)][arm]["all"]["pairs"] == out["per_seed"][f"{s}_{arm}"]["S3"]["returned_slots"], (s, arm)
        out["funnel_pairs_equal_returned_slots"] = True
        print("funnel.json pair counts == per-user returned slots: OK")
    json.dump(out, open(os.path.join(a.root, "tables", "slot_reconciliation.json"), "w"), indent=1)


def cmd_path_validity(a):
    sys.path[:0] = [a.spike_a_src]
    from easydict import EasyDict as edict
    from spike_a import artifacts as aa, core, matched
    cfg, kg_d, scores, train, test = load_inputs(a.root, a.tmp_dir)
    kga = edict(cfg["KG_ARGS"])
    kgo = aa.load_pickle(os.path.join(a.tmp_dir, "kg.pkl"))
    embs = aa.load_pickle(os.path.join(a.tmp_dir, "transe_embed.pkl"))
    g = core.Graph.from_kg(kgo, kga)
    max_acts = cfg["TEST_AGENT"]["max_acts"]
    env_p = core.make_env(kgo, embs, kga, max_acts)
    full_a, pr_a = core.FullActions(g), core.PrunedActions(g, env_p, max_acts)
    nl = lambda path: tuple(matched.nonloop(path[1:], kga.self_loop))  # noqa: E731
    out = {}
    for s in SEEDS:
        for arm, lam in arms(a.root).items():
            t0 = time.time()
            res = pickle.load(open(os.path.join(adir(a.root, s, lam), "beam_test", "beam.pkl"), "rb"))
            pred, frozen = mm.build(res, scores, test, train)
            del res
            seen = set()
            st = {"paths": 0, "concept_paths": 0, "bad_full": 0, "bad_pruned": 0, "sel_paths": 0, "sel_bad_pruned": 0,
                  "sel_not_in_cands": 0, "pairs": 0}
            for u, top in frozen.items():
                pr_a.reset()
                for s0 in top:
                    pid = s0[2][-1][2]
                    cands = pred[u][pid]
                    st["pairs"] += 1
                    cset = {nl(c[2]) for c in cands}
                    sels = [fn.sb.select_path(x, cands, s0, kga.self_loop) for x in ("S0", "S3")]
                    st["sel_not_in_cands"] += sum(nl(ch[2]) not in cset for ch in sels)
                    for c in cands:
                        p = nl(c[2])
                        if (u, p) in seen:
                            continue
                        seen.add((u, p))
                        pt = [(kga.self_loop, "user", u)] + list(p)
                        st["paths"] += 1
                        st["concept_paths"] += mm.has_concept(c[2])
                        st["bad_full"] += not matched.path_accepted(pt, g, full_a)[0]
                        st["bad_pruned"] += not matched.path_accepted(pt, g, pr_a)[0]
                    for ch in sels:
                        pt = [(kga.self_loop, "user", u)] + list(nl(ch[2]))
                        st["sel_paths"] += 1
                        st["sel_bad_pruned"] += not matched.path_accepted(pt, g, pr_a)[0]
            st["secs"] = round(time.time() - t0)
            out[f"{s}_{arm}"] = st
            print(s, arm, st, flush=True)
            json.dump(out, open(os.path.join(a.root, "tables", "funnel_path_validity.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["funnel", "summarize", "slots", "path_validity"])
    ap.add_argument("--root", required=True, help="m5 output root (Drive m5/)")
    ap.add_argument("--tmp_dir", help="dir with kg.pkl, *_label.pkl, transe_embed.pkl (mooc_train4_eval3)")
    ap.add_argument("--cache_dir", help="Spike A matched cache dir (funnel)")
    ap.add_argument("--spike_a_src", help="dir containing the spike_a package (path_validity)")
    a = ap.parse_args()
    {"funnel": cmd_funnel, "summarize": cmd_summarize, "slots": cmd_slots, "path_validity": cmd_path_validity}[a.cmd](a)


if __name__ == "__main__":
    main()
