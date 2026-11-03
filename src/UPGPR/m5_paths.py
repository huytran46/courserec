"""M5 read-only analysis: which pattern types do the displayed explanations (selector S0, S3 for reference) have?

Reads the saved M5 raw beams (test split) and recomputes the same frozen top-10 lists and S0/S3 selections as
m5_metrics. Nothing is trained or re-ranked; outputs go to NEW files under <root>/tables.

Full pattern of a displayed path = sequence of (relation, entity type reached) over the non-self-loop edges after the start
user, e.g. enroll>item, item_concept>concept, item_concept>item.

Group rule (decided from the entity types of the INTERMEDIATE nodes, i.e. all nodes after the start user except the final
item, ignoring item-type nodes because every path must pass through the enrolled item first):
  user -> co_learner, concept -> concept, category -> category, teacher -> teacher, school -> school
  exactly one such entity type            -> that group
  two or more different entity types      -> hybrid (reported separately, never merged)
  none, or the path does not end at item  -> other
The MOOC KG has entity types user, item, school, teacher, concept only: category does not exist, its count is 0 by schema.

  counts   --root R --tmp_dir T     -> tables/s0_path_types_counts.json
  report   --root R                 -> tables/s0_path_types.json (+ markdown printed)
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

SEEDS = (23, 24, 25)
K = 10
ENTITY_GROUP = {"user": "co_learner", "concept": "concept", "category": "category", "teacher": "teacher",
                "school": "school"}
GROUPS = ["co_learner", "concept", "category", "teacher", "school", "hybrid", "other"]


def pattern(path, self_loop="self_loop"):
    """path = [(relation, node_type, node_id), ...] with path[0] the start. Returns tuple of (relation, node_type)."""
    return tuple((r, t) for r, t, _ in path[1:] if r != self_loop)


def pattern_str(pat):
    return "user " + " ".join(f"-{r}-> {t}" for r, t in pat) if pat else "user (no edge)"


def group_of(pat):
    if not pat or pat[-1][1] != "item":
        return "other"
    ents = {t for _, t in pat[:-1] if t != "item"}
    if not ents:
        return "other"
    if len(ents) > 1:
        return "hybrid"
    e = next(iter(ents))
    return ENTITY_GROUP.get(e, "other")


def count_selection(pred, frozen, selector, self_loop="self_loop", concept_rel="item_concept"):
    """Counts over the displayed (selected) path of every returned slot. Returns dict."""
    import spike_b_posthoc as sb
    groups, pats = Counter(), Counter()
    n = n_concept_any = 0
    hybrid_with_concept = 0
    for u, top in frozen.items():
        for s0 in top:
            cands = pred[u][s0[2][-1][2]]
            ch = sb.select_path(selector, cands, s0, self_loop)
            pat = pattern(ch[2], self_loop)
            g = group_of(pat)
            groups[g] += 1
            pats[pattern_str(pat)] += 1
            has_c = any(r == concept_rel for r, _ in pat)
            n_concept_any += has_c
            hybrid_with_concept += has_c and g == "hybrid"
            n += 1
    return {"slots_with_explanation": n, "groups": {g: groups.get(g, 0) for g in GROUPS},
            "concept_any": n_concept_any, "hybrid_with_concept": hybrid_with_concept,
            "patterns": dict(pats.most_common())}


def msd(xs):
    import numpy as np
    xs = list(xs)
    return {"per_seed": xs, "mean": float(np.mean(xs)), "sd_sample": float(np.std(xs, ddof=1)) if len(xs) > 1 else None}


def build_report(C, slots, top_n=10):
    """C = counts json. Shares over all slots (slots/seed, missing counted separately) and over explained slots."""
    out = {"slots_per_seed": slots, "arms": {}, "consistency": {}, "top_patterns": {}}
    for arm in ("control", "shaped"):
        for sel in ("S0", "S3"):
            per = {s: C["seeds"][str(s)][arm][sel] for s in SEEDS}
            row = {"slots_with_explanation": msd([per[s]["slots_with_explanation"] for s in SEEDS]),
                   "missing_slots": msd([slots - per[s]["slots_with_explanation"] for s in SEEDS]), "groups": {}}
            for g in GROUPS:
                cnt = [per[s]["groups"][g] for s in SEEDS]
                row["groups"][g] = {"count": msd(cnt),
                                    "share_of_all_slots": msd([c / slots for c in cnt]),
                                    "share_of_explained": msd([c / per[s]["slots_with_explanation"]
                                                               for c, s in zip(cnt, SEEDS)])}
            out["arms"][f"{arm}_{sel}"] = row
            tot = Counter()
            for s in SEEDS:
                tot.update(per[s]["patterns"])
            lst = []
            for p, c in tot.most_common(top_n):
                sh = [per[s]["patterns"].get(p, 0) / per[s]["slots_with_explanation"] for s in SEEDS]
                lst.append({"pattern": p, "count_pooled": c,
                            "share_of_explained_pooled": c / sum(per[s]["slots_with_explanation"] for s in SEEDS),
                            "share_of_explained_per_seed": sh, "mean": msd(sh)["mean"], "sd_sample": msd(sh)["sd_sample"]})
            out["top_patterns"][f"{arm}_{sel}"] = {"patterns": lst, "n_distinct_patterns_pooled": len(tot)}
            out["consistency"][f"{arm}_{sel}"] = {
                "concept_group_share_of_all_slots": [per[s]["groups"]["concept"] / slots for s in SEEDS],
                "any_item_concept_share_of_all_slots": [per[s]["concept_any"] / slots for s in SEEDS],
                "eval_concept_at10": [C["seeds"][str(s)][arm]["eval_concept_at10"][sel] for s in SEEDS],
                "hybrid_with_concept_slots": [per[s]["hybrid_with_concept"] for s in SEEDS]}
            c = out["consistency"][f"{arm}_{sel}"]
            c["max_abs_diff_any_concept_vs_eval"] = max(abs(a - b) for a, b in zip(c["any_item_concept_share_of_all_slots"], c["eval_concept_at10"]))
            c["max_abs_diff_concept_group_vs_eval"] = max(abs(a - b) for a, b in zip(c["concept_group_share_of_all_slots"], c["eval_concept_at10"]))
    return out


def cmd_counts(a):
    import numpy as np
    import m5_funnel_run as fr
    import m5_metrics as mm
    cfg, kg, scores, train, test = fr.load_inputs(a.root, a.tmp_dir)
    out = {"kg_relations": kg["kg_relation"], "kg_entities": kg["entities"], "self_loop": kg["self_loop"],
           "users": sum(1 for p in test.values() if p), "seeds": {}}
    out["slots"] = K * out["users"]
    for s in SEEDS:
        out["seeds"][str(s)] = {}
        for arm, lam in fr.arms(a.root).items():
            d = fr.adir(a.root, s, lam)
            res = pickle.load(open(os.path.join(d, "beam_test", "beam.pkl"), "rb"))
            pred, frozen = mm.build(res, scores, test, train)
            ev = json.load(open(os.path.join(d, "eval_test.json")))
            r = {sel: count_selection(pred, frozen, sel, kg["self_loop"]) for sel in ("S0", "S3")}
            r["eval_concept_at10"] = {sel: ev[sel]["concept_at10"] for sel in ("S0", "S3")}
            out["seeds"][str(s)][arm] = r
            print(s, arm, {sel: r[sel]["groups"] for sel in ("S0", "S3")}, flush=True)
            del res, pred, frozen
    os.makedirs(os.path.join(a.root, "tables"), exist_ok=True)
    json.dump(out, open(os.path.join(a.root, "tables", "s0_path_types_counts.json"), "w"), indent=1)


def cmd_report(a):
    C = json.load(open(os.path.join(a.root, "tables", "s0_path_types_counts.json")))
    R = build_report(C, C["slots"])
    R["kg_entities"] = C["kg_entities"]
    R["kg_relations"] = C["kg_relations"]
    json.dump(R, open(os.path.join(a.root, "tables", "s0_path_types.json"), "w"), indent=1)
    for k, v in R["arms"].items():
        print(k, "explained", [int(x) for x in v["slots_with_explanation"]["per_seed"]], "missing",
              [int(x) for x in v["missing_slots"]["per_seed"]])
        for g in GROUPS:
            x = v["groups"][g]
            print(f"  {g:10s} count {[int(c) for c in x['count']['per_seed']]}  all {100 * x['share_of_all_slots']['mean']:6.2f}+-"
                  f"{100 * (x['share_of_all_slots']['sd_sample'] or 0):4.2f}  explained {100 * x['share_of_explained']['mean']:6.2f}+-"
                  f"{100 * (x['share_of_explained']['sd_sample'] or 0):4.2f}  per_seed_expl "
                  f"{[round(100 * y, 2) for y in x['share_of_explained']['per_seed']]}")
    for k, v in R["consistency"].items():
        print("consistency", k, "concept-group", [round(100 * x, 3) for x in v["concept_group_share_of_all_slots"]], "any-concept",
              [round(100 * x, 3) for x in v["any_item_concept_share_of_all_slots"]], "eval", [round(100 * x, 3) for x in v["eval_concept_at10"]],
              "maxdiff", v["max_abs_diff_any_concept_vs_eval"], v["max_abs_diff_concept_group_vs_eval"])
    for k, v in R["top_patterns"].items():
        print("TOP", k, "distinct", v["n_distinct_patterns_pooled"])
        for p in v["patterns"]:
            print(f"   {100 * p['share_of_explained_pooled']:6.2f}%  {p['pattern']}  per_seed {[round(100 * x, 2) for x in p['share_of_explained_per_seed']]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["counts", "report"])
    ap.add_argument("--root", required=True)
    ap.add_argument("--tmp_dir")
    a = ap.parse_args()
    {"counts": cmd_counts, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
