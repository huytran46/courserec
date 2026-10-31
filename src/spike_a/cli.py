"""Spike A CLI: alternative-path reachability for test (learner, course) pairs.

    python src/spike_a/cli.py --config config/UPGPR/mooc.json --artifacts-dir <run dir> \
        --source-commit <sha> --out-dir <new dir>

No training, no writes outside --out-dir. Requires the repo venv (torch is imported by kg_env
only when pruned analysis runs).
"""
import argparse
import csv
import json
import os
import random
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from math import comb

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.dirname(_HERE)
_REPO = os.path.dirname(_SRC)
for _p in (os.path.join(_SRC, "UPGPR"), _SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
from easydict import EasyDict as edict  # noqa: E402

from spike_a import artifacts, core  # noqa: E402
from spike_a.bruteforce import brute_force  # noqa: E402

TOOL_VERSION = "spike-a/1"
REQUIRED = ["kg.pkl", "train_label.pkl", "test_label.pkl"]
OPTIONAL = ["validation_label.pkl", "transe_embed.pkl"]
CODE_FILES = ["src/UPGPR/kg_env.py", "src/UPGPR/knowledge_graph.py", "src/UPGPR/make_dataset.py",
              "src/UPGPR/data_utils.py", "src/UPGPR/preprocess_mooc.py", "src/UPGPR/utils.py"]
INF_ACTS = 10 ** 9


def parse(argv):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--out-dir", required=True, help="must not contain a previous run's manifest.json")
    p.add_argument("--artifacts-dir", help="dir with kg.pkl etc. (default: TEST_AGENT.tmp_dir)")
    p.add_argument("--data-dir", help="processed_files dir for ID-map audit (default: TEST_AGENT.data_dir)")
    p.add_argument("--source-commit", help="commit that produced the artifacts")
    p.add_argument("--allow-unverified", action="store_true",
                   help="proceed without --source-commit; outputs are labelled UNVERIFIED_PROVENANCE")
    p.add_argument("--horizons", type=int, nargs="+", default=[3])
    p.add_argument("--enable-h4", action="store_true", help="also horizon 4 (more expensive)")
    p.add_argument("--enable-h5", action="store_true", help="also horizon 5 (much more expensive)")
    p.add_argument("--max-acts", type=int, help="override TEST_AGENT.max_acts (recorded as deviation)")
    p.add_argument("--no-pruned", action="store_true")
    p.add_argument("--max-expansions-per-user", type=int, default=50_000_000)
    p.add_argument("--sample-users", type=int, help="analyse a random sample of users (not exact cohort)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bruteforce-pairs", type=int, default=5)
    p.add_argument("--bruteforce-max-calls", type=int, default=300_000)
    p.add_argument("--witnesses", type=int, default=3, help="pairs per family to keep witnesses for")
    p.add_argument("--beam-paths", help="optional policy_paths_epoch_N.pkl for separate beam analysis")
    p.add_argument("--reference-node-count", type=int,
                   help="node count of a reference run (e.g. smoke run) to explain discrepancies against")
    p.add_argument("--accept-node-count-discrepancy", help="reason text; downgrades the discrepancy error")
    a = p.parse_args(argv)
    if 4 in a.horizons and not a.enable_h4:
        p.error("horizon 4 needs --enable-h4 (expensive)")
    if 5 in a.horizons and not a.enable_h5:
        p.error("horizon 5 needs --enable-h5 (very expensive)")
    if 3 not in a.horizons:
        p.error("horizon 3 is the primary analysis and must be included")
    a.horizons = sorted(set(a.horizons))
    return a


# --------------------------------------------------------------------------- helpers
def _git(*args):
    try:
        r = subprocess.run(["git", *args], cwd=_REPO, capture_output=True, text=True, timeout=60)
        return r.returncode, r.stdout.strip()
    except Exception as e:  # pragma: no cover
        return 1, str(e)


def _file_info(path):
    st = os.stat(path)
    return {"path": os.path.abspath(path), "sha256": artifacts.sha256(path), "bytes": st.st_size,
            "mtime_utc": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat()}


def _write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=False, default=str)


def label(pattern):
    return ">".join(f"{r}{s}" for r, s, _ in pattern)


def pair_stats(counts, horizon):
    """counts: Counter[(pattern, k)] -> summary of one (learner, course) pair."""
    per_pat = Counter()
    by_len = Counter()
    for (pat, k), n in counts.items():
        per_pat[pat] += n
        by_len[k] += n
    fams = {core.family(p) for p in per_pat}
    return {
        "paths": sum(per_pat.values()),
        "seqs": core.action_sequences(counts, horizon),
        "per_pat": per_pat,
        "by_len": by_len,
        "n_patterns": len(per_pat),
        "families": sorted(fams),
        "n_families": len(fams),
        "types": sorted({t for f in fams for t in f.split("+")} - {"item"}),
    }


def _quant(vals):
    if not vals:
        return None
    a = np.asarray(vals)
    q = np.percentile(a, [25, 50, 75, 90, 99])
    return {"min": int(a.min()), "p25": float(q[0]), "median": float(q[1]), "p75": float(q[2]),
            "p90": float(q[3]), "p99": float(q[4]), "max": int(a.max()), "mean": float(a.mean())}


def _share(num, den):
    return (num / den) if den else None


# --------------------------------------------------------------------------- witnesses
def load_raw(kg_args, data_dir):
    """Raw processed files as an independent edge source (None if absent)."""
    try:
        raw = {"enroll": set()}
        with open(os.path.join(data_dir, "train.txt")) as f:
            for line in f:
                u, c = line.split()
                raw["enroll"].add((int(u), int(c)))
        for rel, (fname, _) in kg_args.item_relation.items():
            s = set()
            with open(os.path.join(data_dir, fname)) as f:
                for cid, line in enumerate(f.read().splitlines()):
                    for x in line.split():
                        s.add((cid, int(x)))
            raw[rel] = s
        return raw
    except OSError:
        return None


def verify_path(path, uid, target, G, kg_args, train, env=None, raw=None):
    """Return list of problems (empty == every edge verified)."""
    probs = []
    sl = kg_args.self_loop
    if path[0] != (sl, "user", uid):
        probs.append("bad start")
    seen = {("user", uid)}
    for i in range(1, len(path)):
        rel, nt, nid = path[i]
        _, pt, pid = path[i - 1]
        if rel == sl:
            probs.append("self-loop in non-self-loop witness")
            continue
        if kg_args.kg_relation[pt].get(rel) != nt:
            probs.append(f"type mismatch at hop {i}")
            continue
        if nid not in G[pt][pid][rel] or pid not in G[nt][nid][rel]:
            probs.append(f"edge missing in KG at hop {i}")
        if (nt, nid) in seen:
            probs.append(f"repeated node at hop {i}")
        seen.add((nt, nid))
        if env is not None and (rel, nid) not in env._get_actions(path[:i], False):
            probs.append(f"hop {i} not in env._get_actions (pruned out)")
        if raw is not None:
            if rel == kg_args.interaction:
                u, c = (pid, nid) if pt == "user" else (nid, pid)
                if (u, c) not in raw["enroll"]:
                    probs.append(f"raw train.txt lacks enrol at hop {i}")
            elif rel in raw:
                c, x = (pid, nid) if pt == "item" else (nid, pid)
                if (c, x) not in raw[rel]:
                    probs.append(f"raw file lacks {rel} edge at hop {i}")
    if path[-1][1:] != ("item", target):
        probs.append("does not end at target")
    if target in train.get(uid, ()):
        probs.append("target is a train course")
    return probs


# --------------------------------------------------------------------------- summary
def summarize(rows_by_mode, H, n_users_eligible, pattern_nodes):
    out = {}
    for mode, rows in rows_by_mode.items():
        if rows is None:
            out[mode] = None
            continue
        elig = len(rows)
        reach = [r for r in rows if r["stats"]["paths"] > 0]
        censored = [r for r in rows if r["censored"]]
        users_reach = {r["uid"] for r in reach}
        users_all = {r["uid"] for r in rows}
        s = {
            "denominators": {"eligible_pairs": elig, "reachable_pairs": len(reach),
                             "users_with_eligible_pair": len(users_all),
                             "users_with_reachable_pair": len(users_reach)},
            "censored_pairs": len(censored),
            "exact": len(censored) == 0,
            "share_reachable_of_eligible": _share(len(reach), elig),
        }
        for t in ("teacher", "school", "concept", "user"):
            n = sum(1 for r in reach if t in r["stats"]["types"])
            s[f"share_reachable_with_{t}"] = _share(n, len(reach))
            s[f"share_eligible_with_{t}"] = _share(n, elig)
        for name, fn in (("ge2_patterns", lambda r: r["stats"]["n_patterns"] >= 2),
                         ("ge2_families", lambda r: r["stats"]["n_families"] >= 2),
                         ("ge2_paths", lambda r: r["stats"]["paths"] >= 2)):
            n = sum(1 for r in reach if fn(r))
            s[f"share_reachable_with_{name}"] = _share(n, len(reach))
            s[f"share_eligible_with_{name}"] = _share(n, elig)
            s[f"n_reachable_with_{name}"] = n
        by_user = defaultdict(list)
        for r in reach:
            by_user[r["uid"]].append(r)
        s["users_with_any_ge2_families_pair"] = sum(
            1 for rs in by_user.values() if any(x["stats"]["n_families"] >= 2 for x in rs))
        s["share_users_reachable_with_any_ge2_families_pair"] = _share(
            s["users_with_any_ge2_families_pair"], len(users_reach))
        s["paths_per_reachable_pair"] = _quant([r["stats"]["paths"] for r in reach])
        s["n_patterns_distribution"] = dict(sorted(Counter(r["stats"]["n_patterns"] for r in reach).items()))
        s["n_families_distribution"] = dict(sorted(Counter(r["stats"]["n_families"] for r in reach).items()))
        agg_len = Counter()
        loops_seq = Counter()
        pat_paths, pat_pairs = Counter(), Counter()
        for r in reach:
            for k, n in r["stats"]["by_len"].items():
                agg_len[k] += n
                loops_seq[H - k] += n * comb(H, k)
            for p, n in r["stats"]["per_pat"].items():
                pat_paths[p] += n
                pat_pairs[p] += 1
        s["effective_length_distribution_paths"] = dict(sorted(agg_len.items()))
        s["min_effective_length_per_pair"] = dict(sorted(
            Counter(min(r["stats"]["by_len"]) for r in reach).items()))
        tp = sum(r["stats"]["paths"] for r in reach)
        ts = sum(r["stats"]["seqs"] for r in reach)
        s["totals"] = {"dedup_non_selfloop_paths": tp, "action_sequences": ts,
                       "sequence_inflation_factor": _share(ts, tp)}
        s["action_sequences_by_selfloop_count"] = dict(sorted(loops_seq.items()))
        s["pattern_catalog"] = [
            {"label": label(p), "pattern": core.pattern_str(p), "family": core.family(p),
             "total_paths": pat_paths[p], "pairs_with_pattern": pat_pairs[p]}
            for p in sorted(pat_paths, key=lambda p: -pat_paths[p])]
        flags = []
        for k in ("share_reachable_with_ge2_families", "share_reachable_with_ge2_patterns",
                  "share_reachable_with_teacher", "share_reachable_with_school",
                  "share_reachable_with_concept"):
            if s[k] is not None and s[k] < 0.05:
                flags.append(f"{k}={s[k]:.4f} < 5%: review signal only, not evidence against relation shaping")
        s["review_flags_below_5pct"] = flags
        out[mode] = s
    return out


def compare_full_pruned(rows_full, rows_pruned):
    if rows_pruned is None:
        return None
    pf = {(r["uid"], r["cid"]): r for r in rows_full}
    n = lost = fam_lost = 0
    retained = []
    trans = Counter()
    for r in rows_pruned:
        f = pf[(r["uid"], r["cid"])]
        n += 1
        fp, pp = f["stats"]["paths"], r["stats"]["paths"]
        if fp > 0 and pp == 0:
            lost += 1
        if f["stats"]["n_families"] > r["stats"]["n_families"]:
            fam_lost += 1
        if fp > 0:
            retained.append(pp / fp)
        trans[(f"full_ge2fam={f['stats']['n_families'] >= 2}", f"pruned_ge2fam={r['stats']['n_families'] >= 2}")] += 1
    return {"pairs": n, "reachable_in_full_but_unreachable_pruned": lost,
            "pairs_losing_at_least_one_family": fam_lost,
            "mean_fraction_of_paths_retained": float(np.mean(retained)) if retained else None,
            "ge2_families_transitions": {f"{a} -> {b}": c for (a, b), c in sorted(trans.items())}}


# --------------------------------------------------------------------------- main
def main(argv=None):
    a = parse(argv)
    t_start = time.time()
    out = a.out_dir
    if os.path.exists(os.path.join(out, "manifest.json")):
        print(f"refusing to overwrite previous run in {out}")
        return 4
    os.makedirs(out, exist_ok=True)

    cfg_text = open(a.config, "rb").read()
    cfg = json.loads(cfg_text)
    kg_args = edict(cfg["KG_ARGS"])
    test_cfg, train_cfg = cfg["TEST_AGENT"], cfg["TRAIN_AGENT"]
    art_dir = a.artifacts_dir or test_cfg["tmp_dir"]
    data_dir = a.data_dir or test_cfg["data_dir"]
    max_acts = a.max_acts or test_cfg["max_acts"]
    import hashlib
    manifest = {
        "tool": TOOL_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv if argv is None else ["spike_a.cli", *argv]),
        "git": {"head": _git("rev-parse", "HEAD")[1], "source_commit_claimed": a.source_commit,
                "dirty_src": bool(_git("status", "--porcelain", "--", "src")[1])},
        "config": {"path": os.path.abspath(a.config), "sha256": hashlib.sha256(cfg_text).hexdigest(),
                   "effective": {"max_acts": max_acts, "max_acts_overridden": a.max_acts is not None,
                                 "train_max_acts": train_cfg["max_acts"], "test_max_path_len": test_cfg["max_path_len"],
                                 "use_pattern": test_cfg["use_pattern"], "reward": test_cfg["reward"],
                                 "seed": test_cfg["seed"], "state_history": test_cfg["state_history"],
                                 "artifacts_dir": os.path.abspath(art_dir), "data_dir": os.path.abspath(data_dir),
                                 "use_user_relations": cfg["PREPROCESS"]["use_user_relations"],
                                 "use_entity_relations": cfg["PREPROCESS"]["use_entity_relations"]}},
        "code_hashes": {f: artifacts.sha256(os.path.join(_REPO, f)) for f in CODE_FILES
                        if os.path.exists(os.path.join(_REPO, f))},
        "definitions": DEFINITIONS,
        "horizons_requested": a.horizons,
    }

    # ---- inputs
    paths = {n: os.path.join(art_dir, n) for n in REQUIRED + OPTIONAL}
    missing = [paths[n] for n in REQUIRED if not os.path.exists(paths[n])]
    if missing:
        manifest.update(status="BLOCKED", missing=missing,
                        note="required artifacts not found; supply the run directory that holds them")
        _write_json(os.path.join(out, "manifest.json"), manifest)
        open(os.path.join(out, "report.md"), "w").write(
            "# Spike A - BLOCKED\n\nMissing required files:\n\n" + "\n".join(f"- `{m}`" for m in missing) + "\n")
        print("BLOCKED, missing:", *missing, sep="\n  ")
        return 2
    manifest["artifacts"] = {n: _file_info(p) for n, p in paths.items() if os.path.exists(p)}
    kg = artifacts.load_pickle(paths["kg.pkl"])
    train = artifacts.load_pickle(paths["train_label.pkl"])
    test = artifacts.load_pickle(paths["test_label.pkl"])
    val = artifacts.load_pickle(paths["validation_label.pkl"]) if os.path.exists(paths["validation_label.pkl"]) else None
    embeds = None
    pruned_missing = []
    if a.no_pruned:
        pruned_status = "SKIPPED_by_flag"
    elif os.path.exists(paths["transe_embed.pkl"]):
        embeds = artifacts.load_pickle(paths["transe_embed.pkl"])
        pruned_status = "ENABLED"
    else:
        pruned_status = "BLOCKED_missing_embeddings"
        pruned_missing = [paths["transe_embed.pkl"] + " (transe_embed.pkl from the same run as kg.pkl)"]

    # ---- audit
    rep = artifacts.audit(kg.G, kg_args, train, test, val, embeds,
                          data_dir=data_dir if os.path.isdir(data_dir) else None)
    if not os.path.isdir(data_dir):
        rep["warnings"].append({"code": "data_dir_missing", "detail": f"{data_dir}: ID-map vocab files not audited"})
    if json.dumps(kg.kg_args, sort_keys=True, default=str) != json.dumps(cfg["KG_ARGS"], sort_keys=True):
        rep["errors"].append({"code": "kg_args_differ_from_config",
                              "detail": "KG pickle was built with different KG_ARGS than --config"})
    if test_cfg["max_acts"] != train_cfg["max_acts"]:
        rep["warnings"].append({"code": "train_test_max_acts_differ",
                                "detail": f"train={train_cfg['max_acts']} test={test_cfg['max_acts']}; using {max_acts}"})
    if cfg["PREPROCESS"]["use_user_relations"] or cfg["PREPROCESS"]["use_entity_relations"]:
        rep["warnings"].append({"code": "extra_relations_enabled", "detail": "user/entity relations in KG"})
    if embeds is not None and os.path.getmtime(paths["transe_embed.pkl"]) < os.path.getmtime(paths["kg.pkl"]):
        rep["warnings"].append({"code": "embed_older_than_kg", "detail": "mtime only; Drive copies may reset mtimes"})
    if a.reference_node_count is not None and rep["info"]["total_nodes"] != a.reference_node_count:
        d = {"code": "node_count_differs_from_reference",
             "detail": f"kg={rep['info']['total_nodes']} reference={a.reference_node_count} "
                       f"per_type={rep['info']['node_counts']}. Node counts depend only on preprocess_mooc "
                       f"(thresholds, MOOCCube version), not on split or training."}
        if a.accept_node_count_discrepancy:
            d["detail"] += f" ACCEPTED: {a.accept_node_count_discrepancy}"
            rep["warnings"].append(d)
        else:
            rep["errors"].append(d)
    # provenance
    prov = {"source_commit": a.source_commit, "verified": False}
    if a.source_commit:
        rc, _ = _git("cat-file", "-e", a.source_commit + "^{commit}")
        if rc != 0:
            rep["errors"].append({"code": "source_commit_unknown", "detail": a.source_commit})
        else:
            _, diff = _git("diff", "--name-only", a.source_commit, "HEAD", "--", *CODE_FILES)
            prov["code_files_changed_since_source_commit"] = diff.splitlines()
            if diff:
                rep["errors"].append({"code": "source_code_differs_from_head", "detail": diff.replace("\n", ", ")})
            else:
                prov["verified"] = True
    else:
        e = {"code": "source_commit_missing",
             "detail": "pass --source-commit (commit that produced these artifacts); filenames are not provenance"}
        (rep["warnings"] if a.allow_unverified else rep["errors"]).append(e)
    manifest["provenance"] = prov
    manifest["audit"] = rep
    manifest["pruned"] = {"status": pruned_status, "missing": pruned_missing, "max_acts": max_acts}
    n_test_pairs = sum(len(set(v)) for v in test.values())
    manifest["split_sizes"] = {"train_pairs": sum(len(set(v)) for v in train.values()),
                               "test_pairs": n_test_pairs,
                               "validation_pairs": None if val is None else sum(len(set(v)) for v in val.values()),
                               "train_users": len(train), "test_users": len(test),
                               "kg_nodes": rep["info"].get("node_counts")}
    if rep["errors"]:
        manifest["status"] = "AUDIT_FAILED"
        _write_json(os.path.join(out, "manifest.json"), manifest)
        _write_json(os.path.join(out, "diagnostics.json"), rep)
        open(os.path.join(out, "report.md"), "w").write(
            "# Spike A - AUDIT FAILED (no numerical interpretation)\n\n"
            + "\n".join(f"- **{e['code']}**: {e['detail']}" for e in rep["errors"])
            + "\n\nSee diagnostics.json. Node counts: " + json.dumps(rep["info"].get("node_counts")) + "\n")
        print("AUDIT FAILED:", *[e["code"] for e in rep["errors"]])
        return 3

    # ---- cohort
    G = kg.G
    g = core.Graph.from_kg(kg, kg_args)
    excluded = []
    eligible = defaultdict(list)
    for u in sorted(test):
        trn = set(train.get(u, ()))
        for c in sorted(set(test[u])):
            if u not in G["user"] or c not in G["item"]:
                excluded.append((u, c, "unsupported_node_missing_from_kg"))
            elif c in trn or c in G["user"][u].get(kg_args.interaction, ()):
                excluded.append((u, c, "positive_already_in_train"))
            else:
                eligible[u].append(c)
    users = sorted(eligible)
    sampled = a.sample_users is not None and a.sample_users < len(users)
    if sampled:
        users = sorted(random.Random(a.seed).sample(users, a.sample_users))
    manifest["cohort"] = {"sampled": sampled, "seed": a.seed if sampled else None,
                          "users": len(users), "users_with_eligible_pair_before_sampling": len(eligible),
                          "pairs": sum(len(eligible[u]) for u in users),
                          "exclusions": dict(Counter(r for _, _, r in excluded)), "exact_full_cohort": not sampled}
    with open(os.path.join(out, "excluded_pairs.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["uid", "cid", "reason"])
        w.writerows(excluded)

    env_full = env_pruned = None
    try:
        env_full = core.make_env(kg, embeds, kg_args, INF_ACTS)  # never scores, so embeddings not needed
        if embeds is not None:
            env_pruned = core.make_env(kg, embeds, kg_args, max_acts)
    except ImportError as e:
        manifest["pruned"].update(status="BLOCKED_import_error", missing=[str(e)])
        env_full = env_pruned = None
        embeds = None

    results = {}  # H -> {"full": rows, "pruned": rows|None}
    timing = {}
    witness_pool = []
    for H in a.horizons:
        full_a = core.FullActions(g)
        pr_a = core.PrunedActions(g, env_pruned, max_acts) if env_pruned is not None else None
        rows = {"full": [], "pruned": [] if pr_a else None}
        tf = tp = 0.0
        exp_f = exp_p = 0
        per_family_kept = Counter()
        multi_kept = 0
        for i, u in enumerate(users):
            if i % 250 == 0:
                print(f"[H={H}] user {i}/{len(users)}", flush=True)
            t0 = time.time()
            rf = core.enumerate_user(g, u, eligible[u], H, full_a, a.max_expansions_per_user)
            tf += time.time() - t0
            exp_f += rf.expansions
            rp = None
            if pr_a:
                t0 = time.time()
                rp = core.enumerate_user(g, u, eligible[u], H, pr_a, a.max_expansions_per_user)
                tp += time.time() - t0
                exp_p += rp.expansions
            for c in eligible[u]:
                for mode, res in (("full", rf), ("pruned", rp)):
                    if res is None:
                        continue
                    st = pair_stats(res.counts[c], H)
                    rows[mode].append({"uid": u, "cid": c, "stats": st, "censored": res.censored,
                                       "counts": res.counts[c]})
                    if H == 3 and mode == "full" and st["paths"] > 0:
                        fams = st["families"]
                        take = False
                        for fam in fams:
                            if per_family_kept[fam] < a.witnesses:
                                per_family_kept[fam] += 1
                                take = True
                        if len(fams) >= 2 and multi_kept < a.witnesses + 2:
                            multi_kept += 1
                            take = True
                        if take:
                            seen_f = set()
                            for (t, pat), ws in rf.witnesses.items():
                                if t == c and core.family(pat) not in seen_f:
                                    seen_f.add(core.family(pat))
                                    witness_pool.append(("full", u, c, pat, ws[0]))
                    if H == 3 and mode == "pruned" and rp is not None and (u, c) in {(w[1], w[2]) for w in witness_pool[-12:]}:
                        seen_f = set()
                        for (t, pat), ws in rp.witnesses.items():
                            if t == c and core.family(pat) not in seen_f:
                                seen_f.add(core.family(pat))
                                witness_pool.append(("pruned", u, c, pat, ws[0]))
        results[H] = rows
        timing[str(H)] = {"full_seconds": round(tf, 2), "pruned_seconds": round(tp, 2) if pr_a else None,
                          "full_expansions": exp_f, "pruned_expansions": exp_p if pr_a else None,
                          "env_get_actions_calls": pr_a.env_calls if pr_a else None,
                          "fast_path_action_sets": pr_a.fast_calls if pr_a else None}
    manifest["timing"] = timing

    # ---- brute-force cross-check on real pairs (primary horizon)
    bf = {"horizon": 3, "requested_pairs": a.bruteforce_pairs, "checked_pairs": 0, "mismatches": 0,
          "aborted": [], "details": []}
    if a.bruteforce_pairs:
        def cost(u):
            return sum(sum(len(v) for v in G["item"][c].values()) for c in G["user"][u].get(kg_args.interaction, ()))
        cand = sorted(users, key=cost)[: a.bruteforce_pairs]
        by_pair = {m: {(r["uid"], r["cid"]): r for r in rs} for m, rs in results[3].items() if rs is not None}
        for u in cand:
            for mode, env in (("full", env_full), ("pruned", env_pruned)):
                if env is None:
                    continue
                res = brute_force(env, kg_args, u, eligible[u], 3, max_calls=a.bruteforce_max_calls)
                if res["aborted"]:
                    bf["aborted"].append({"uid": u, "mode": mode})
                    continue
                for c in eligible[u]:
                    eng = by_pair[mode][(u, c)]["stats"]
                    ok = (res["paths"].get(c, 0) == eng["paths"] and res["seqs"].get(c, 0) == eng["seqs"])
                    bf["details"].append({"uid": u, "cid": c, "mode": mode, "engine_paths": eng["paths"],
                                          "bruteforce_paths": res["paths"].get(c, 0), "engine_seqs": eng["seqs"],
                                          "bruteforce_seqs": res["seqs"].get(c, 0), "repeated_node_paths": res["repeated"],
                                          "match": ok})
                    bf["mismatches"] += (not ok)
            bf["checked_pairs"] += len([d for d in bf["details"] if d["uid"] == u and d["mode"] == "full"])
    manifest["bruteforce"] = bf

    # ---- witnesses
    raw = load_raw(kg_args, data_dir) if os.path.isdir(data_dir) else None
    wit, all_ok = [], True
    for mode, u, c, pat, path in witness_pool:
        probs = verify_path(path, u, c, G, kg_args, train, env=env_pruned if mode == "pruned" else None, raw=raw)
        all_ok &= not probs
        wit.append({"mode": mode, "uid": u, "cid": c, "pattern": core.pattern_str(pat), "family": core.family(pat),
                    "path": [list(x) for x in path], "problems": probs})
    _write_json(os.path.join(out, "witnesses.json"),
                {"all_edges_verified": all_ok, "raw_files_cross_checked": raw is not None,
                 "pruned_edges_checked_against_env": env_pruned is not None, "witnesses": wit})

    # ---- outputs per horizon
    summary = {"status_note": "counts are exact only where censored_pairs == 0 and cohort.exact_full_cohort",
               "cohort": manifest["cohort"], "horizons": {}}
    for H, rows in results.items():
        pats = sorted({p for m in ("full", "pruned") if rows[m] for r in rows[m] for p in r["stats"]["per_pat"]})
        fname = "pairs.csv" if H == 3 else f"pairs_h{H}.csv"
        with open(os.path.join(out, fname), "w", newline="") as f:
            cols = ["uid", "cid", "horizon", "status"]
            for m in ("full", "pruned"):
                cols += [f"{m}_{x}" for x in ("paths", "seqs", "n_patterns", "n_families", "families", "censored",
                                              "alt_patterns", "alt_families", "ge2_paths")]
                cols += [f"{m}_len{k}" for k in range(1, H + 1)]
                cols += [f"{m}[{label(p)}]" for p in pats]
            cols += ["lost_paths_by_pruning"]
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            pr = {(r["uid"], r["cid"]): r for r in rows["pruned"]} if rows["pruned"] else {}
            for r in rows["full"]:
                st = r["stats"]
                row = {"uid": r["uid"], "cid": r["cid"], "horizon": H,
                       "status": ("reachable" if st["paths"] else
                                  ("indeterminate_censored" if r["censored"] else "unreachable"))}
                for m, rr in (("full", r), ("pruned", pr.get((r["uid"], r["cid"])))):
                    if rr is None:
                        continue
                    s = rr["stats"]
                    row.update({f"{m}_paths": s["paths"], f"{m}_seqs": s["seqs"], f"{m}_n_patterns": s["n_patterns"],
                                f"{m}_n_families": s["n_families"], f"{m}_families": "|".join(s["families"]),
                                f"{m}_censored": rr["censored"], f"{m}_alt_patterns": s["n_patterns"] >= 2,
                                f"{m}_alt_families": s["n_families"] >= 2, f"{m}_ge2_paths": s["paths"] >= 2})
                    for k in range(1, H + 1):
                        row[f"{m}_len{k}"] = s["by_len"].get(k, 0)
                    for p in pats:
                        row[f"{m}[{label(p)}]"] = s["per_pat"].get(p, 0)
                if (r["uid"], r["cid"]) in pr:
                    row["lost_paths_by_pruning"] = st["paths"] - pr[(r["uid"], r["cid"])]["stats"]["paths"]
                w.writerow(row)
        summary["horizons"][str(H)] = {
            "denominators": {"eligible_pairs": len(rows["full"]),
                             "reachable_pairs_full": sum(1 for r in rows["full"] if r["stats"]["paths"] > 0),
                             "users_with_eligible_pair": len({r["uid"] for r in rows["full"]})},
            **summarize(rows, H, len(users), None),
            "full_vs_pruned_same_pairs": compare_full_pruned(rows["full"], rows["pruned"]),
            "note": "totals at different horizons are not comparable as same-condition counts",
        }
    _write_json(os.path.join(out, "summary.json"), summary)

    # ---- optional beam analysis (separate)
    if a.beam_paths:
        manifest["beam"] = beam_analysis(a.beam_paths, results[3], kg_args, out)

    unv = not prov["verified"]
    any_cens = any(r["censored"] for rs in results.values() for m in rs.values() if m for r in m)
    manifest["status"] = "COMPLETE_UNVERIFIED_PROVENANCE" if unv else "COMPLETE"
    manifest["exactness"] = {"exact_full_cohort": not sampled, "any_censored": any_cens}
    manifest["total_seconds"] = round(time.time() - t_start, 2)
    _write_json(os.path.join(out, "manifest.json"), manifest)
    write_report(out, manifest, summary)
    print("done:", manifest["status"], "->", out)
    return 0


def beam_analysis(beam_path, rows3, kg_args, out):
    """Separate from graph reachability: what the trained beam actually surfaced for each eligible pair."""
    res = artifacts.load_pickle(beam_path)
    sl = kg_args.self_loop
    got = defaultdict(lambda: Counter())  # (u,c) -> Counter[pattern-of-labels]
    nseq = Counter()
    for path in res["paths"]:
        if path[-1][1] != "item":
            continue
        u, c = path[0][2], path[-1][2]
        nonloop = [x for x in path[1:] if x[0] != sl]
        sig = tuple((r, t) for r, t, _ in nonloop)
        got[(u, c)][sig] += 1
        nseq[(u, c)] += 1
    pr = {(r["uid"], r["cid"]): r for r in (rows3.get("pruned") or [])}
    fl = {(r["uid"], r["cid"]): r for r in rows3["full"]}
    n = len(fl)
    hit = sum(1 for k in fl if nseq.get(k, 0) > 0)
    multi = sum(1 for k in fl if len(got.get(k, {})) >= 2)
    notin = 0
    ref = pr or fl
    for k, sigs in got.items():
        if k not in ref:
            continue
        have = {tuple((r, t) for r, _, t in p) for p in ref[k]["stats"]["per_pat"]}
        notin += sum(1 for s in sigs if s not in have)
    return {"file": beam_path, "sha256": artifacts.sha256(beam_path), "eligible_pairs": n,
            "pairs_with_beam_path_to_positive": hit, "share": _share(hit, n),
            "pairs_with_ge2_beam_patterns": multi, "share_eligible_ge2_beam_patterns": _share(multi, n),
            "beam_patterns_not_in_" + ("pruned" if pr else "full") + "_enumeration": notin,
            "interpretation": "beam availability is NOT a graph reachability estimate; nonzero 'not_in' "
                              "means beam and enumeration disagree (version/embedding mismatch)"}


DEFINITIONS = {
    "path": "list of (relation, node_type, node_id); first tuple is the start (self_loop,'user',uid), not a hop",
    "effective_length": "number of non-self-loop edges",
    "horizon": "number of agent steps (max_path_len); T3 is primary",
    "dedup_non_self_loop_paths": "distinct sequences of non-self-loop edges; the main count",
    "action_sequences": "env action sequences incl. self-loops = sum_k n_k * C(horizon,k); equals dedup count only "
                        "when every path has effective length == horizon",
    "visited_filter": "ordinary traversal never revisits a (type,id) on the path; only self-loop repeats a node",
    "pattern": "per hop: relation, direction (+ canonical head->tail, - reverse), tail node type",
    "family": "types of intermediate nodes (user=co-learner via enrol, teacher, school, concept)",
    "n_types (n_families)": "number of distinct families with >=1 path for the pair",
    "full": "no action pruning (every non-visited neighbour)",
    "pruned": "action sets from BatchKGEnvironment._get_actions (top-max_acts by embedding score) with the same "
              "embeddings, max_acts and path history",
    "whitelist_patterns": "KG_ARGS.path_pattern is only a reward whitelist (use_pattern); it is not applied as a "
                          "search constraint here",
}


def write_report(out, m, s):
    L = [f"# Spike A: alternative-path reachability\n", f"**Status: {m['status']}**\n"]
    if m["status"].endswith("UNVERIFIED_PROVENANCE"):
        L.append("> Source commit not verified. Treat numbers as provisional.\n")
    if m["cohort"]["sampled"]:
        L.append(f"> SAMPLED cohort (seed {m['cohort']['seed']}), not the exact full cohort.\n")
    L += ["## Provenance and audit\n",
          f"- repo HEAD `{m['git']['head'][:10]}`, claimed source commit `{m['git']['source_commit_claimed']}`, "
          f"config sha256 `{m['config']['sha256'][:12]}`",
          f"- KG nodes: {m['split_sizes']['kg_nodes']}; split sizes: train {m['split_sizes']['train_pairs']}, "
          f"test {m['split_sizes']['test_pairs']}, validation {m['split_sizes']['validation_pairs']}",
          f"- cohort: {m['cohort']['users']} users, {m['cohort']['pairs']} eligible pairs, exclusions "
          f"{m['cohort']['exclusions']}",
          f"- audit errors: {len(m['audit']['errors'])}, warnings: " + "; ".join(w['code'] for w in m['audit']['warnings']),
          f"- pruned analysis: {m['pruned']['status']} (max_acts {m['pruned']['max_acts']})",
          f"- brute-force cross-check (T3): {m['bruteforce']['checked_pairs']} pairs, "
          f"{m['bruteforce']['mismatches']} mismatches, aborted {len(m['bruteforce']['aborted'])}\n"]
    for H, hs in s["horizons"].items():
        L.append(f"## Horizon {H}\n")
        for mode in ("full", "pruned"):
            x = hs.get(mode)
            if not x:
                L.append(f"### {mode}: not available\n")
                continue
            d = x["denominators"]
            L += [f"### {mode}\n",
                  f"- eligible pairs {d['eligible_pairs']}, reachable {d['reachable_pairs']} "
                  f"({_pct(x['share_reachable_of_eligible'])}), users with eligible pair "
                  f"{d['users_with_eligible_pair']}, with reachable pair {d['users_with_reachable_pair']}",
                  f"- censored pairs {x['censored_pairs']} (exact: {x['exact']})",
                  "- share of reachable pairs with a path via: " + ", ".join(
                      f"{t} {_pct(x['share_reachable_with_' + t])}" for t in ("teacher", "school", "concept", "user")),
                  f"- >=2 patterns {_pct(x['share_reachable_with_ge2_patterns'])}, >=2 families "
                  f"{_pct(x['share_reachable_with_ge2_families'])}, >=2 paths {_pct(x['share_reachable_with_ge2_paths'])}"
                  f" (of reachable); of eligible: {_pct(x['share_eligible_with_ge2_families'])} for >=2 families",
                  f"- users with any >=2-family pair: {_pct(x['share_users_reachable_with_any_ge2_families_pair'])} "
                  f"of users with a reachable pair",
                  f"- paths per reachable pair: {x['paths_per_reachable_pair']}",
                  f"- effective length (paths): {x['effective_length_distribution_paths']}; totals {x['totals']}",
                  f"- n_families distribution: {x['n_families_distribution']}"]
            for fl in x["review_flags_below_5pct"]:
                L.append(f"- REVIEW SIGNAL: {fl}")
            L.append("")
        if hs.get("full_vs_pruned_same_pairs"):
            L.append(f"### full vs pruned (same pairs)\n\n{hs['full_vs_pruned_same_pairs']}\n")
    L += ["## Definitions\n"] + [f"- **{k}**: {v}" for k, v in m["definitions"].items()]
    L += ["\n## Limits\n",
          "- Counts show headroom/constraints only. They do not measure controllability or learnability (spike E).",
          "- A share below 5% is a signal to re-examine, not evidence against relation shaping.",
          "- Totals at different horizons are different conditions and must not be compared directly."]
    open(os.path.join(out, "report.md"), "w").write("\n".join(L) + "\n")


def _pct(x):
    return "n/a" if x is None else f"{100 * x:.1f}%"


if __name__ == "__main__":
    sys.exit(main())
