"""Spike A, step 2: reachability matched to Spike B (full-epoch 3/3).

Two separate cohorts, same KG and the embeddings Spike B's environment loaded:
  positive : test (user, course) pairs not in train (held-out positives)
  topk     : Spike B selector S0 pairs (frozen baseline top-10 per user, de-duplicated)
Per pair: full-graph, pruned (BatchKGEnvironment._get_actions, max_acts from config) and beam
availability (paths actually present in the saved beam), at horizon 3. No training, no writes outside
--out-dir.
"""
import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.dirname(_HERE)
_REPO = os.path.dirname(_SRC)
for _p in (os.path.join(_SRC, "UPGPR"), _SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
from easydict import EasyDict as edict  # noqa: E402

from spike_a import artifacts, core, matched  # noqa: E402
from spike_a.bruteforce import brute_force  # noqa: E402
from spike_a.cli import (CODE_FILES, INF_ACTS, _git, _quant, _share, _write_json, label,  # noqa: E402
                         pair_stats, verify_path)

TOOL_VERSION = "spike-a-matched/1"
H = 3
TARGETS = ("concept", "teacher", "school")
FAMS = matched.FAMILIES
KEYMAP = {"kg.pkl": "kg", "train_label.pkl": "train_label", "test_label.pkl": "test_label",
          "validation_label.pkl": "validation_label", "transe_embed.pkl": "embeddings"}


def parse(argv):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="Spike B effective_config.json")
    p.add_argument("--base-config", default="config/UPGPR/mooc.json")
    p.add_argument("--artifacts-dir", required=True, help="B run dir (kg.pkl, labels, transe_embed.pkl)")
    p.add_argument("--beam-paths", required=True)
    p.add_argument("--b-dir", required=True, help="spike_b_full_e50 dir (manifest, selected_paths.jsonl, ...)")
    p.add_argument("--a-dir", help="old Spike A output dir (manifest.json, pairs.csv)")
    p.add_argument("--a-artifacts-dir", help="old Spike A artifacts dir, to recompute its hashes")
    p.add_argument("--embed-ckpt-dir", help="TransE checkpoint dir (default: <artifacts>/train_transe_model)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--env-source-commit", required=True, help="commit of the code that ran the B pipeline")
    p.add_argument("--tool-commit", help="git commit of this tool (NOT the upstream commit)")
    p.add_argument("--ref-hash", action="append", default=[], metavar="NAME=SHA256",
                   help="reference hash from the task statement (kg, train_label, test_label, embeddings)")
    p.add_argument("--expect-positive-pairs", type=int, default=12860)
    p.add_argument("--expect-topk-pairs", type=int, default=64461)
    p.add_argument("--expect-topk-users", type=int, default=6507)
    p.add_argument("--expect-max-acts", type=int, default=250)
    p.add_argument("--expect-horizon", type=int, default=3)
    p.add_argument("--max-expansions-per-user", type=int, default=50_000_000)
    p.add_argument("--bruteforce-users", type=int, default=3)
    p.add_argument("--bruteforce-max-calls", type=int, default=300_000)
    p.add_argument("--witnesses", type=int, default=3)
    p.add_argument("--resume", action="store_true", help="continue from out-dir/checkpoint.jsonl")
    p.add_argument("--max-users", type=int, help="DEBUG ONLY: process the first N users; output is labelled partial")
    return p.parse_args(argv)


def sha(path):
    return artifacts.sha256(path) if path and os.path.exists(path) else None


def _jload(p):
    with open(p) as f:
        return json.load(f)


# ------------------------------------------------------------------ provenance
def embedding_ckpt_match(embeds, ckpt_dir, kg_args):
    """Which TransE checkpoint epochs reproduce transe_embed.pkl exactly? (name says nothing)"""
    try:
        import torch
    except ImportError:
        return {"status": "unavailable_no_torch", "matching_epochs": None}
    if not os.path.isdir(ckpt_dir):
        return {"status": "unavailable_no_ckpt_dir", "matching_epochs": None}
    rels = list(kg_args.item_relation.keys()) + [kg_args.interaction]
    match, checked = [], []
    for fn in sorted(os.listdir(ckpt_dir)):
        if not (fn.startswith("transe_model_sd_epoch_") and fn.endswith(".ckpt")):
            continue
        ep = int(fn.split("_")[-1].split(".")[0])
        sd = torch.load(os.path.join(ckpt_dir, fn), map_location="cpu")
        ok = all(np.array_equal(sd[f"{e}.weight"].numpy()[:-1], embeds[e]) for e in kg_args.entities)
        ok = ok and all(np.array_equal(sd[r].numpy()[0], embeds[r][0]) for r in rels)
        checked.append(ep)
        if ok:
            match.append(ep)
    return {"status": "checked", "checkpoints_checked": len(checked), "matching_epochs": sorted(match)}


def tool_identity(d=_HERE):
    # hidden files (e.g. macOS '._*.py' resource forks from tar) are not part of the tool
    files = {f: sha(os.path.join(d, f)) for f in sorted(os.listdir(d))
             if f.endswith(".py") and not f.startswith(".")}
    bundle = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return {"files_sha256": files, "bundle_sha256": bundle}


def build_provenance(a, B, cfg_path):
    """Table artifact | hash A | hash B | task reference | recomputed (B dir) | recomputed (A dir)."""
    mb = _jload(os.path.join(a.b_dir, "manifest.json"))
    ma = _jload(os.path.join(a.a_dir, "manifest.json")) if a.a_dir and os.path.exists(os.path.join(a.a_dir, "manifest.json")) else None
    refs = dict(x.split("=", 1) for x in a.ref_hash)
    spec = {}
    for fname, key in KEYMAP.items():
        spec[key] = {
            "A_manifest": ma["artifacts"][fname]["sha256"] if ma and fname in ma.get("artifacts", {}) else None,
            "B_manifest": mb["hashes"].get(key),
            "task_reference": refs.get(key),
            "recomputed_B": sha(os.path.join(a.artifacts_dir, fname)),
            "recomputed_A": sha(os.path.join(a.a_artifacts_dir, fname)) if a.a_artifacts_dir else None,
        }
    spec["beam"] = {"A_manifest": ma.get("beam", {}).get("sha256") if ma else None,
                    "B_manifest": mb["hashes"].get("beam"), "task_reference": refs.get("beam"),
                    "recomputed_B": sha(a.beam_paths), "recomputed_A": None}
    spec["effective_config"] = {"A_manifest": None, "B_manifest": mb["hashes"].get("config"),
                                "task_reference": refs.get("effective_config"),
                                "recomputed_B": sha(cfg_path), "recomputed_A": None}
    return matched.provenance_table(spec), mb, ma


# ------------------------------------------------------------------ main
def main(argv=None):
    a = parse(argv)
    t_start = time.time()
    out = a.out_dir
    if os.path.exists(os.path.join(out, "manifest.json")):
        print(f"refusing to overwrite previous run in {out}")
        return 4
    os.makedirs(out, exist_ok=True)
    diag = {"errors": [], "warnings": [], "notes": []}
    manifest = {
        "tool": TOOL_VERSION, "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(["spike_a.matched_cli", *(argv if argv is not None else sys.argv[1:])]),
        "horizon": H,
        "tool_identity": {"git_commit": a.tool_commit, **tool_identity()},
    }
    manifest["environment"] = {"pipeline_source_commit": a.env_source_commit,
                               "checkout_head_where_tool_ran": _git("rev-parse", "HEAD")[1]}

    # ---- required inputs
    ad = a.artifacts_dir
    need = {"effective_config": a.config, "kg.pkl": f"{ad}/kg.pkl", "train_label.pkl": f"{ad}/train_label.pkl",
            "test_label.pkl": f"{ad}/test_label.pkl", "transe_embed.pkl": f"{ad}/transe_embed.pkl",
            "beam": a.beam_paths, "B manifest.json": f"{a.b_dir}/manifest.json",
            "selected_paths.jsonl": f"{a.b_dir}/selected_paths.jsonl",
            "baseline_reconstruction.json": f"{a.b_dir}/baseline_reconstruction.json"}
    missing = [p for p in need.values() if not os.path.exists(p)]
    if missing:
        manifest.update(status="BLOCKED", missing=missing)
        _write_json(os.path.join(out, "manifest.json"), manifest)
        open(os.path.join(out, "report.md"), "w").write("# Spike A matched-B: BLOCKED\n\nMissing:\n\n"
                                                        + "\n".join(f"- `{m}`" for m in missing) + "\n")
        print("BLOCKED", *missing, sep="\n  ")
        return 2

    # ---- provenance table (recomputed from the files, not copied from reports)
    rows, mb, ma = build_provenance(a, ad, a.config)
    prov = {"table": rows, "a_manifest_present": ma is not None,
            "a_inputs_recomputed": bool(a.a_artifacts_dir),
            "note": ("old Spike A input files recomputed from --a-artifacts-dir" if a.a_artifacts_dir
                     else "old Spike A inputs not re-hashed: only its manifest is compared")}
    st = {r["artifact"]: r for r in rows}
    stop = []
    for k in ("kg", "train_label", "test_label", "embeddings", "beam", "effective_config"):
        r = st[k]
        vals = {x for x in (r["B_manifest"], r["task_reference"], r["recomputed_B"]) if x}
        if len(vals) > 1 or r["recomputed_B"] is None:
            stop.append(f"{k}: B-side hashes disagree or file missing {r}")
    a_vs_b = {k: st[k]["status"] for k in st}
    prov["reuse_decision"] = {
        "A_full_graph_results_reusable_as_cross_check": all(
            st[k]["A_manifest"] and st[k]["A_manifest"] == st[k]["recomputed_B"]
            for k in ("kg", "train_label", "test_label")),
        "A_pruned_results_reusable": bool(st["embeddings"]["A_manifest"]
                                           and st["embeddings"]["A_manifest"] == st["embeddings"]["recomputed_B"]),
        "A_vs_B_status": a_vs_b}
    manifest["provenance"] = prov
    if stop:
        diag["errors"] += [{"code": "provenance_mismatch_B", "detail": s} for s in stop]

    cfg = _jload(a.config)
    base = _jload(a.base_config) if os.path.exists(a.base_config) else None
    ta = cfg["TEST_AGENT"]
    manifest["effective_config"] = {"path": os.path.abspath(a.config), "sha256": sha(a.config),
                                    "test_agent": ta, "train_embeds_epochs": cfg["TRAIN_EMBEDS"]["epochs"],
                                    "train_agent_epochs": cfg["TRAIN_AGENT"]["epochs"],
                                    "train_agent_max_path_len": cfg["TRAIN_AGENT"]["max_path_len"]}
    if ta["max_path_len"] != a.expect_horizon:
        diag["errors"].append({"code": "eval_horizon_not_3", "detail": ta["max_path_len"]})
    if ta["max_acts"] != a.expect_max_acts:
        diag["errors"].append({"code": "max_acts_unexpected", "detail": ta["max_acts"]})
    if base is not None:
        changed = [k for k in ("PREPROCESS", "TRAIN_EMBEDS", "TRAIN_AGENT", "TEST_AGENT")
                   for kk in set(cfg[k]) | set(base[k]) if cfg[k].get(kk) != base[k].get(kk) and kk != "tmp_dir"
                   and kk != "data_dir"]
        manifest["effective_config"]["differs_from_base_config_beyond_tmp_dir"] = sorted(set(changed))
        if cfg["KG_ARGS"] != base["KG_ARGS"]:
            diag["errors"].append({"code": "kg_args_differ_from_base_config", "detail": ""})
    max_acts = ta["max_acts"]
    kg_args = edict(cfg["KG_ARGS"])

    # ---- environment commit and code drift
    rc, _ = _git("cat-file", "-e", a.env_source_commit + "^{commit}")
    env = {"commit_known_in_checkout": rc == 0}
    if rc == 0:
        _, dif = _git("diff", "--name-only", a.env_source_commit, "HEAD", "--", *CODE_FILES)
        env["code_files_changed_vs_checkout_head"] = dif.splitlines()
        if dif:
            diag["errors"].append({"code": "source_code_differs_from_checkout", "detail": dif.replace("\n", ", ")})
    else:
        diag["errors"].append({"code": "env_source_commit_unknown_in_checkout", "detail": a.env_source_commit})
    manifest["environment"].update(env)

    # ---- load and audit
    kg = artifacts.load_pickle(f"{ad}/kg.pkl")
    train = artifacts.load_pickle(f"{ad}/train_label.pkl")
    test = artifacts.load_pickle(f"{ad}/test_label.pkl")
    vp = f"{ad}/validation_label.pkl"
    val = artifacts.load_pickle(vp) if os.path.exists(vp) else None
    embeds = artifacts.load_pickle(f"{ad}/transe_embed.pkl")
    rep = artifacts.audit(kg.G, kg_args, train, test, val, embeds, data_dir=None)
    rep["warnings"].append({"code": "id_map_files_not_audited",
                            "detail": "no processed_files here; vocab sizes checked only against KG/embedding rows"})
    if json.dumps(kg.kg_args, sort_keys=True, default=str) != json.dumps(cfg["KG_ARGS"], sort_keys=True):
        rep["errors"].append({"code": "kg_args_differ_from_config", "detail": ""})
    diag["errors"] += rep["errors"]
    diag["warnings"] += rep["warnings"]
    manifest["audit"] = rep
    ckdir = a.embed_ckpt_dir or f"{ad}/train_transe_model"
    ck = embedding_ckpt_match(embeds, ckdir, kg_args)
    manifest["embedding_epoch_evidence"] = {
        "config_epochs": cfg["TRAIN_EMBEDS"]["epochs"], **ck,
        "log_last_line": _tail(f"{ckdir}/train_log.txt")}
    if ck["status"] == "checked" and cfg["TRAIN_EMBEDS"]["epochs"] not in (ck["matching_epochs"] or []):
        diag["errors"].append({"code": "embeddings_not_equal_to_config_epoch_checkpoint",
                               "detail": f"matching epochs: {ck['matching_epochs']}"})
    manifest["split_sizes"] = {"train_pairs": sum(len(set(v)) for v in train.values()),
                               "test_pairs": sum(len(set(v)) for v in test.values()),
                               "kg_nodes": rep["info"].get("node_counts")}
    if diag["errors"]:
        manifest["status"] = "AUDIT_FAILED"
        _write_json(os.path.join(out, "provenance_comparison.json"), prov)
        _write_json(os.path.join(out, "diagnostics.json"), diag)
        _write_json(os.path.join(out, "manifest.json"), manifest)
        open(os.path.join(out, "report.md"), "w").write(
            "# Spike A matched-B: AUDIT FAILED (no numerical interpretation)\n\n"
            + "\n".join(f"- **{e['code']}**: {e['detail']}" for e in diag["errors"]) + "\n")
        print("AUDIT FAILED:", *[e["code"] for e in diag["errors"]])
        return 3

    # ---- cohorts
    G = kg.G
    g = core.Graph.from_kg(kg, kg_args)
    n_items = len(G["item"])
    pos = defaultdict(list)
    pos_excl = Counter()
    for u in sorted(test):
        trn = set(train.get(u, ()))
        for c in sorted(set(test[u])):
            if u not in G["user"] or c not in G["item"]:
                pos_excl["unsupported_node_missing_from_kg"] += 1
            elif c in trn:
                pos_excl["positive_already_in_train"] += 1
            else:
                pos[u].append(c)
    cohort, cinfo = matched.load_s0_cohort(f"{a.b_dir}/selected_paths.jsonl")
    ccheck = matched.check_cohort(cohort, train, n_items)
    br = _jload(f"{a.b_dir}/baseline_reconstruction.json")
    cdiag = {"loader": cinfo, "checks": ccheck, "pairs_in_test_positives": matched.pairs_in_test_positives(cohort, test),
             "baseline_reconstruction": {"n_users_lt10": br.get("n_users_lt10"),
                                         "n_users_with_frozen_list": br.get("n_users_with_frozen_list"),
                                         "n_beam_paths": br.get("n_beam_paths")}}
    if ccheck["problems"]:
        diag["errors"].append({"code": "topk_cohort_invalid", "detail": ccheck["problems"][:5]})
    for key, got, exp in (("positive_pairs", sum(len(v) for v in pos.values()), a.expect_positive_pairs),
                          ("topk_pairs", len(cohort), a.expect_topk_pairs),
                          ("topk_users", ccheck["users"], a.expect_topk_users),
                          ("topk_users_lt10", ccheck["users_lt10"], br.get("n_users_lt10"))):
        cdiag[key] = {"got": got, "expected": exp, "match": got == exp}
        if got != exp:
            diag["warnings"].append({"code": f"cohort_count_differs:{key}", "detail": f"got {got} expected {exp}"})
    manifest["cohorts"] = {"positive": {"users": len(pos), "pairs": cdiag["positive_pairs"]["got"],
                                         "exclusions": dict(pos_excl)},
                           "topk_S0": cdiag}
    if ccheck["problems"]:
        manifest["status"] = "AUDIT_FAILED"
        _write_json(os.path.join(out, "diagnostics.json"), diag)
        _write_json(os.path.join(out, "manifest.json"), manifest)
        return 3
    top_by_user = defaultdict(dict)
    for c in cohort:
        top_by_user[c["uid"]][c["cid"]] = c
    all_pairs = {(u, c) for u, cs in pos.items() for c in cs} | {(c["uid"], c["cid"]) for c in cohort}

    # ---- beam index
    beam = artifacts.load_pickle(a.beam_paths)
    bidx = matched.beam_index(beam, all_pairs, kg_args.self_loop)
    bidx_u = defaultdict(dict)
    for (u, c), e in bidx.items():
        bidx_u[u][c] = e
    manifest["beam"] = {"file": os.path.abspath(a.beam_paths), "sha256": sha(a.beam_paths),
                        "n_paths": len(beam["paths"]), "cohort_pairs_with_beam_paths": len(bidx)}
    del beam

    env_p = core.make_env(kg, embeds, kg_args, max_acts)
    env_f = core.make_env(kg, embeds, kg_args, INF_ACTS)
    full_a = core.FullActions(g)
    pr_a = core.PrunedActions(g, env_p, max_acts)

    # ---- per-user enumeration (checkpointed)
    ck_path = os.path.join(out, "checkpoint.jsonl")
    done = {}
    if a.resume and os.path.exists(ck_path):
        with open(ck_path) as f:
            for line in f:
                pk = json.loads(line)
                done[pk["uid"]] = pk
    users = sorted(set(pos) | set(top_by_user))
    if a.max_users:
        users = users[:a.max_users]
        diag["warnings"].append({"code": "PARTIAL_debug_max_users", "detail": a.max_users})
    t0 = time.time()
    ckf = open(ck_path, "a")
    pat_cat = {}
    for i, u in enumerate(users):
        if u in done:
            continue
        if i % 250 == 0:
            print(f"[matched] user {i}/{len(users)} ({time.time() - t0:.0f}s)", flush=True)
        targets = set(pos.get(u, [])) | set(top_by_user.get(u, {}))
        rf = core.enumerate_user(g, u, targets, H, full_a, a.max_expansions_per_user, witnesses_per_pattern=1)
        rp = core.enumerate_user(g, u, targets, H, pr_a, a.max_expansions_per_user, witnesses_per_pattern=1)
        pk = {"uid": u, "pairs": []}
        for c in sorted(targets):
            pair = {"uid": u, "cid": c, "positive": c in pos.get(u, []),
                    "rank": top_by_user.get(u, {}).get(c, {}).get("rank")}
            for mode, res in (("full", rf), ("pruned", rp)):
                fam, pats = Counter(), {}
                for (pat, _k), n in res.counts[c].items():
                    fam[core.family(pat)] += n
                    pats[label(pat)] = pats.get(label(pat), 0) + n
                    pat_cat[label(pat)] = core.pattern_str(pat)
                pair[mode] = {"paths": sum(fam.values()), "fam": dict(fam), "patterns": pats,
                              "censored": res.censored}
                ws = {}
                for (t, pat), plist in res.witnesses.items():
                    if t == c:
                        ws.setdefault(core.family(pat), [list(x) for x in plist[0]])
                pair[mode]["witness"] = ws
            be = bidx_u.get(u, {}).get(c)
            if be:
                bf_fam, bp = Counter(), {}
                bad_f = bad_p = 0
                for nl, cnt in be["paths"].items():
                    f_ = matched.path_family(nl, g)
                    bf_fam[f_] += 1
                    pt = [(g.self_loop, "user", u)] + list(nl)
                    okf, _ = matched.path_accepted(pt, g, full_a)
                    okp, _ = matched.path_accepted(pt, g, pr_a)
                    bad_f += not okf
                    bad_p += not okp
                    bp[f_] = [list(x) for x in pt] if f_ not in bp else bp[f_]
                pair["beam"] = {"seqs": be["seqs"], "paths": len(be["paths"]), "fam": dict(bf_fam),
                                "not_accepted_full": bad_f, "not_accepted_pruned": bad_p, "witness": bp}
            else:
                pair["beam"] = {"seqs": 0, "paths": 0, "fam": {}, "not_accepted_full": 0, "not_accepted_pruned": 0,
                                "witness": {}}
            s0 = top_by_user.get(u, {}).get(c)
            if s0:
                pt = [tuple(x) for x in s0["path"]]
                nl = tuple(matched.nonloop(pt[1:], kg_args.self_loop))
                okf, _ = matched.path_accepted(pt, g, full_a)
                okp, why = matched.path_accepted(pt, g, pr_a)
                pair["s0"] = {"fam": matched.path_family(nl, g), "pattern": label(_pattern_of(nl, g)),
                              "path": [list(x) for x in pt], "accepted_full": okf, "accepted_pruned": okp,
                              "in_beam": nl in (be["paths"] if be else {}), "pruned_reason": None if okp else why}
            pk["pairs"].append(pair)
        ckf.write(json.dumps(pk) + "\n")
        ckf.flush()
        done[u] = pk
    ckf.close()
    timing = {"enumeration_seconds_this_invocation": round(time.time() - t0, 1),
              "env_get_actions_calls": pr_a.env_calls, "fast_path_action_sets": pr_a.fast_calls}

    pairs = [p for u in users for p in done[u]["pairs"]]
    pos_recs = [p for p in pairs if p["positive"]]
    top_recs = sorted((p for p in pairs if p["rank"] is not None), key=lambda p: (p["uid"], p["rank"]))
    # pattern catalog for schema equivalence (rebuild patterns from labels is lossy, so use strings)
    schema = _schema_check(g, done, users)

    # ---- brute-force cross-check on cheap real users (both modes)
    bf = {"users_requested": a.bruteforce_users, "pairs_checked": 0, "mismatches": 0, "aborted": [], "details": []}
    if a.bruteforce_users:
        def cost(u):
            return sum(sum(len(v) for v in G["item"][c].values()) for c in G["user"][u].get(kg_args.interaction, ()))
        for u in sorted(users, key=cost)[: a.bruteforce_users]:
            targets = set(pos.get(u, [])) | set(top_by_user.get(u, {}))
            by = {p["cid"]: p for p in done[u]["pairs"]}
            for mode, e in (("full", env_f), ("pruned", env_p)):
                r = brute_force(e, kg_args, u, targets, H, max_calls=a.bruteforce_max_calls)
                if r["aborted"]:
                    bf["aborted"].append({"uid": u, "mode": mode})
                    continue
                for c in sorted(targets):
                    ok = (r["paths"].get(c, 0) == by[c][mode]["paths"])
                    bf["pairs_checked"] += 1
                    bf["mismatches"] += (not ok)
                    bf["details"].append({"uid": u, "cid": c, "mode": mode, "engine": by[c][mode]["paths"],
                                          "bruteforce": r["paths"].get(c, 0), "repeated": r["repeated"], "match": ok})
    manifest["bruteforce"] = bf

    # ---- B reference counts straight from selected_paths.jsonl
    bref = _b_reference(f"{a.b_dir}/selected_paths.jsonl")

    # ---- summaries
    summary = {"cohort_notes": "positive and topk are separate denominators; never pooled",
               "positive": _cohort_summary(pos_recs, "positive"),
               "topk": _cohort_summary(top_recs, "topk")}
    fun = {}
    for t in TARGETS + ("user",):
        recs = [_funnel_rec(p) for p in top_recs]
        fun[t] = matched.funnel(recs, t)
    summary["topk"]["funnel"] = fun
    summary["topk"]["b_reference_check"] = _check_vs_b(fun, bref)
    summary["positive"]["funnel_beam_available"] = {
        t: matched.funnel([_funnel_rec(p, s0=False) for p in pos_recs], t) for t in TARGETS + ("user",)}
    summary["beam_consistency"] = {
        "beam_paths_not_accepted_by_full": sum(p["beam"]["not_accepted_full"] for p in pairs),
        "beam_paths_not_accepted_by_pruned": sum(p["beam"]["not_accepted_pruned"] for p in pairs),
        "s0_paths_not_accepted_by_full": sum(1 for p in top_recs if not p["s0"]["accepted_full"]),
        "s0_paths_not_accepted_by_pruned": sum(1 for p in top_recs if not p["s0"]["accepted_pruned"]),
        "s0_paths_not_in_beam_pair": sum(1 for p in top_recs if not p["s0"]["in_beam"])}
    summary["schema_equivalence"] = schema
    summary["pattern_catalog"] = pat_cat
    a_reg = _regress_vs_a(a, pos_recs)
    summary["regression_vs_old_spike_a_full_graph"] = a_reg
    summary["censored_pairs"] = {"full": sum(p["full"]["censored"] for p in pairs),
                                 "pruned": sum(p["pruned"]["censored"] for p in pairs)}
    _write_json(os.path.join(out, "summary.json"), summary)
    _write_json(os.path.join(out, "provenance_comparison.json"), prov)

    # ---- csv
    _write_csv(os.path.join(out, "positive_pairs.csv"), pos_recs, topk=False)
    _write_csv(os.path.join(out, "topk_pairs.csv"), top_recs, topk=True)

    # ---- witnesses
    wit, ok_all = _witnesses(pairs, top_recs, g, G, kg_args, train, env_p, a.witnesses)
    _write_json(os.path.join(out, "witnesses.json"), {"all_edges_and_actions_verified": ok_all, "witnesses": wit})

    partial = bool(a.max_users)
    any_cens = summary["censored_pairs"]["full"] or summary["censored_pairs"]["pruned"]
    manifest["timing"] = timing
    manifest["exactness"] = {"exact_full_cohorts": not partial, "any_censored": bool(any_cens)}
    manifest["diagnostics_summary"] = {"errors": diag["errors"], "warnings": diag["warnings"]}
    bad = (summary["beam_consistency"]["beam_paths_not_accepted_by_pruned"] or bf["mismatches"]
           or not summary["topk"]["b_reference_check"]["all_match"] or not ok_all)
    manifest["status"] = ("PARTIAL" if partial else ("COMPLETE_WITH_DISCREPANCIES" if bad else "COMPLETE"))
    manifest["total_seconds"] = round(time.time() - t_start, 1)
    _write_json(os.path.join(out, "manifest.json"), manifest)
    if diag["warnings"] or bad:
        _write_json(os.path.join(out, "diagnostics.json"), diag)
    _report(out, manifest, summary, prov, bf, bref)
    print("done:", manifest["status"], "->", out)
    return 0


# ------------------------------------------------------------------ helpers
def _tail(path, n=200):
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - n))
            return f.read().decode(errors="replace").strip().splitlines()[-1]
    except OSError:
        return None


def _funnel_rec(p, s0=True):
    return {"full": {k for k, v in p["full"]["fam"].items() if v},
            "pruned": {k for k, v in p["pruned"]["fam"].items() if v},
            "beam": {k for k, v in p["beam"]["fam"].items() if v},
            "s0": {p["s0"]["fam"]} if (s0 and "s0" in p) else set()}


def _pattern_of(nl, g):
    prev, hops = "user", []
    for rel, nt, _ in nl:
        hops.append(g.hop(prev, rel, nt))
        prev = nt
    return tuple(hops)


def _schema_check(g, done, users):
    """Rebuild hop tuples from the pattern labels seen in either mode and test equivalence."""
    labels = {lab for u in users for p in done[u]["pairs"] for mode in ("full", "pruned")
              for lab in p[mode]["patterns"]}
    parsed = set()
    for lab in labels:
        hops = []
        for part in lab.split(">"):
            rel, sign = part[:-1], part[-1]
            head, tail = g.canon[rel]
            hops.append((rel, sign, tail if sign == "+" else head))
        parsed.add(tuple(hops))
    eq = matched.schema_equivalence(parsed)
    eq["n_patterns"] = len(parsed)
    return eq


def _b_reference(path):
    """Exact expected counts recomputed from Spike B's own selected_paths.jsonl."""
    rel = {"concept": "item_concept", "teacher": "item_teacher", "school": "item_school"}
    s0 = Counter()
    sel_gain = Counter()
    sel_ceiling = Counter()
    changed = Counter()
    n = 0
    sel_name = {"S1": "teacher", "S2": "school", "S3": "concept"}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            relations = {x[0] for x in r["path"][1:]}
            if r["selector"] == "S0":
                n += 1
                for t, rn in rel.items():
                    s0[t] += rn in relations
                s0["user"] += not (relations & set(rel.values()))
            elif r["selector"] in sel_name:
                t = sel_name[r["selector"]]
                sel_ceiling[t] += bool(r["new_property"])
                changed[t] += bool(r["changed_path"])
    return {"n_s0_pairs": n, "s0_has": dict(s0), "beam_ceiling_from_selector_new_property": dict(sel_ceiling),
            "selector_changed_pairs": dict(changed)}


def _check_vs_b(fun, bref):
    out = {"all_match": True, "checks": []}
    for t in TARGETS:
        exp_beam = bref["beam_ceiling_from_selector_new_property"].get(t)
        exp_s0 = bref["s0_has"].get(t)
        exp_gain = bref["selector_changed_pairs"].get(t)
        got_beam, got_s0, got_gain = fun[t]["beam"], fun[t]["selected"], fun[t]["selector_miss"]
        for name, got, exp in ((f"{t}_beam_ceiling", got_beam, exp_beam), (f"{t}_s0_has", got_s0, exp_s0),
                               (f"{t}_selector_gain", got_gain, exp_gain)):
            if exp is None:  # selector rows absent in B's jsonl: no reference, not a mismatch
                out["checks"].append({"name": name, "ours": got, "spike_b_from_jsonl": None, "match": None})
                continue
            ok = got == exp
            out["all_match"] &= ok
            out["checks"].append({"name": name, "ours": got, "spike_b_from_jsonl": exp, "match": ok})
    return out


def _regress_vs_a(a, pos_recs):
    if not a.a_dir or not os.path.exists(os.path.join(a.a_dir, "pairs.csv")):
        return {"status": "old A pairs.csv not available"}
    old = {}
    with open(os.path.join(a.a_dir, "pairs.csv")) as f:
        for r in csv.DictReader(f):
            old[(int(r["uid"]), int(r["cid"]))] = (int(r["full_paths"]), int(r["pruned_paths"] or -1))
    diff_full = diff_pruned = common = 0
    for p in pos_recs:
        k = (p["uid"], p["cid"])
        if k in old:
            common += 1
            diff_full += old[k][0] != p["full"]["paths"]
            diff_pruned += old[k][1] != p["pruned"]["paths"]
    return {"pairs_compared": common, "old_pairs": len(old), "full_count_differences": diff_full,
            "pruned_count_differences_expected_nonzero_due_to_different_embeddings": diff_pruned}


def _cohort_summary(recs, name):
    n = len(recs)
    out = {"eligible_pairs": n, "users": len({p["uid"] for p in recs})}
    for mode in ("full", "pruned", "beam"):
        reach = [p for p in recs if (p[mode]["paths"] if mode != "beam" else p["beam"]["paths"]) > 0]
        m = {"reachable_pairs": len(reach), "share_of_eligible": _share(len(reach), n)}
        for t in FAMS:
            k = sum(1 for p in recs if p[mode]["fam"].get(t))
            m[f"pairs_with_{t}"] = k
            m[f"share_eligible_with_{t}"] = _share(k, n)
            m[f"share_reachable_with_{t}"] = _share(k, len(reach))
        for label_, fn in (("ge2_families", lambda p: sum(1 for v in p[mode]["fam"].values() if v) >= 2),
                           ("ge2_distinct_paths", lambda p: p[mode]["paths"] >= 2)):
            k = sum(1 for p in recs if fn(p))
            m[f"pairs_{label_}"] = k
            m[f"share_eligible_{label_}"] = _share(k, n)
            m[f"share_reachable_{label_}"] = _share(k, len(reach))
        counts = [p[mode]["paths"] for p in recs]
        m["paths_per_eligible_pair"] = _quant(counts)
        m["paths_per_reachable_pair"] = _quant([c for c in counts if c > 0])
        out[mode] = m
    lost_all = sum(1 for p in recs if p["full"]["paths"] > 0 and p["pruned"]["paths"] == 0)
    lost_fam = sum(1 for p in recs if any(p["full"]["fam"].get(t) and not p["pruned"]["fam"].get(t) for t in FAMS))
    out["pruning_effect"] = {"pairs_losing_all_reachability": lost_all, "pairs_losing_at_least_one_family": lost_fam,
                             "family_losses": {t: sum(1 for p in recs if p["full"]["fam"].get(t)
                                                      and not p["pruned"]["fam"].get(t)) for t in FAMS},
                             "fraction_of_paths_retained_total": _share(sum(p["pruned"]["paths"] for p in recs),
                                                                        sum(p["full"]["paths"] for p in recs))}
    return out


def _write_csv(path, recs, topk):
    cols = ["uid", "cid"] + (["rank", "is_test_positive"] if topk else []) + ["status"]
    for mode in ("full", "pruned", "beam"):
        cols += [f"{mode}_paths", f"{mode}_n_families", f"{mode}_alt_families", f"{mode}_ge2_paths", f"{mode}_censored"]
        cols += [f"{mode}_{t}" for t in FAMS]
    cols += ["pruning_lost_all", "pruning_lost_family"]
    if topk:
        cols += ["s0_family", "s0_accepted_full", "s0_accepted_pruned", "s0_in_beam"]
        for t in TARGETS + ("user",):
            cols += [f"{t}_pruning_loss", f"{t}_beam_miss", f"{t}_selector_miss"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for p in recs:
            r = {"uid": p["uid"], "cid": p["cid"]}
            if topk:
                r["rank"], r["is_test_positive"] = p["rank"], p["positive"]
            r["status"] = ("reachable" if p["full"]["paths"] else
                           ("indeterminate_censored" if p["full"]["censored"] else "unreachable"))
            for mode in ("full", "pruned", "beam"):
                d = p[mode]
                nf = sum(1 for v in d["fam"].values() if v)
                tot = d["paths"]
                r.update({f"{mode}_paths": tot, f"{mode}_n_families": nf, f"{mode}_alt_families": nf >= 2,
                          f"{mode}_ge2_paths": tot >= 2, f"{mode}_censored": d.get("censored", False)})
                for t in FAMS:
                    r[f"{mode}_{t}"] = d["fam"].get(t, 0)
            r["pruning_lost_all"] = p["full"]["paths"] > 0 and p["pruned"]["paths"] == 0
            r["pruning_lost_family"] = any(p["full"]["fam"].get(t) and not p["pruned"]["fam"].get(t) for t in FAMS)
            if topk:
                fr = _funnel_rec(p)
                r.update({"s0_family": p["s0"]["fam"], "s0_accepted_full": p["s0"]["accepted_full"],
                          "s0_accepted_pruned": p["s0"]["accepted_pruned"], "s0_in_beam": p["s0"]["in_beam"]})
                for t in TARGETS + ("user",):
                    r[f"{t}_pruning_loss"] = t in fr["full"] and t not in fr["pruned"]
                    r[f"{t}_beam_miss"] = t in fr["pruned"] and t not in fr["beam"]
                    r[f"{t}_selector_miss"] = t in fr["beam"] and t not in fr["s0"]
            w.writerow(r)


def _witnesses(pairs, top_recs, g, G, kg_args, train, env_p, k):
    cats = {"pruning_loss": lambda fr, t: t in fr["full"] and t not in fr["pruned"],
            "beam_miss": lambda fr, t: t in fr["pruned"] and t not in fr["beam"],
            "selector_miss": lambda fr, t: t in fr["beam"] and t not in fr["s0"],
            "available_in_pruned": lambda fr, t: t in fr["pruned"]}
    chosen = defaultdict(list)
    for p in top_recs:
        fr = _funnel_rec(p)
        for t in TARGETS:
            for cat, fn in cats.items():
                if len(chosen[(t, cat)]) < k and fn(fr, t):
                    chosen[(t, cat)].append(p)
    out, ok_all = [], True
    for (t, cat), ps in sorted(chosen.items()):
        for p in ps:
            entry = {"target_family": t, "category": cat, "uid": p["uid"], "cid": p["cid"], "rank": p["rank"],
                     "paths": {}}
            for mode in ("full", "pruned", "beam"):
                w = p[mode]["witness"].get(t)
                if not w:
                    continue
                path = [tuple(x) for x in w]
                probs = verify_path(path, p["uid"], p["cid"], G, kg_args, train,
                                    env=env_p if mode in ("pruned", "beam") else None)
                ok_all &= not probs
                entry["paths"][mode] = {"path": w, "problems": probs}
            if p.get("s0"):
                entry["s0_selected_path"] = p["s0"]["path"]
            out.append(entry)
    return out, ok_all


def _fmt(x):
    return "n/a" if x is None else f"{100 * x:.2f}%"


def _report(out, m, s, prov, bf, bref):
    L = ["# Spike A (matched to Spike B full 3/3): headroom and where path availability is lost\n",
         f"**Status: {m['status']}**\n"]
    ti = m["tool_identity"]
    L += ["## Provenance\n",
          f"- tool commit `{ti['git_commit']}`, tool bundle sha256 `{ti['bundle_sha256'][:16]}` (not the upstream commit)",
          f"- B pipeline source commit `{m['environment']['pipeline_source_commit']}`; code files changed vs checkout: "
          f"{m['environment'].get('code_files_changed_vs_checkout_head')}",
          f"- embedding epoch evidence: config {m['embedding_epoch_evidence']['config_epochs']}, "
          f"matching checkpoint epochs {m['embedding_epoch_evidence'].get('matching_epochs')}",
          "", "| artifact | hash A (manifest) | hash B (manifest) | task ref | recomputed B | recomputed A | status |",
          "|---|---|---|---|---|---|---|"]
    for r in prov["table"]:
        sh = lambda v: (v[:12] if v else "-")
        L.append(f"| {r['artifact']} | {sh(r['A_manifest'])} | {sh(r['B_manifest'])} | {sh(r['task_reference'])} | "
                 f"{sh(r['recomputed_B'])} | {sh(r['recomputed_A'])} | {r['status']} |")
    L += [f"\nReuse: {prov['reuse_decision']}\n", "## Cohorts (separate denominators)\n"]
    for name in ("positive", "topk"):
        c = s[name]
        L += [f"### {name}: {c['eligible_pairs']} pairs, {c['users']} users\n"]
        for mode in ("full", "pruned", "beam"):
            x = c[mode]
            L.append(f"- {mode}: reachable {x['reachable_pairs']} ({_fmt(x['share_of_eligible'])}); eligible-share with "
                     + ", ".join(f"{t} {_fmt(x[f'share_eligible_with_{t}'])}" for t in FAMS)
                     + f"; >=2 families {_fmt(x['share_eligible_ge2_families'])}; >=2 paths {_fmt(x['share_eligible_ge2_distinct_paths'])}"
                     + f"; paths median/p90/max (reachable) {_qs(x['paths_per_reachable_pair'])}")
        L.append(f"- pruning effect: {c['pruning_effect']}\n")
    L += ["## Top-K funnel (same denominator = all topk pairs)\n",
          "| target | full | pruned | beam | S0 selected | pruning loss | beam miss | selector miss |", "|---|---|---|---|---|---|---|---|"]
    for t, f in s["topk"]["funnel"].items():
        L.append(f"| {t} | {f['full']} | {f['pruned']} | {f['beam']} | {f['selected']} | {f['pruning_loss']} | "
                 f"{f['beam_miss']} | {f['selector_miss']} |")
    L.append(f"\n(n = {s['topk']['funnel']['concept']['n']}; proportions in summary.json)\n")
    L += ["Check against Spike B (exact counts from its own jsonl):\n"]
    for c in s["topk"]["b_reference_check"]["checks"]:
        L.append(f"- {c['name']}: ours {c['ours']} vs B {c['spike_b_from_jsonl']} -> {'OK' if c['match'] else 'MISMATCH'}")
    L += ["", f"Consistency: {s['beam_consistency']}", f"Schema equivalence (family vs relation vs terminal): "
          f"{s['schema_equivalence']['equivalent']} {s['schema_equivalence']['family_to_relation']}",
          f"Brute force: {bf['pairs_checked']} pair-modes, {bf['mismatches']} mismatches, aborted {len(bf['aborted'])}",
          f"Regression vs old Spike A: {s['regression_vs_old_spike_a_full_graph']}",
          "", "## Scope\n",
          "Availability only: where paths exist in the graph, survive environment action pruning, appear in the saved "
          "beam and are chosen by S0. A beam miss is a location of lost availability; policy, branching budget and "
          "search may all contribute. No claim on reward shaping, causal attribution or learner usefulness."]
    open(os.path.join(out, "report.md"), "w").write("\n".join(L) + "\n")


def _qs(q):
    return "n/a" if not q else f"{q['median']:.0f}/{q['p90']:.0f}/{q['max']}"


if __name__ == "__main__":
    sys.exit(main())
