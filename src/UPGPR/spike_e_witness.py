"""Spike E preflight on REAL data: input audit (hashes/versions) + support check for the bonus.

Usage (repo root): python src/UPGPR/spike_e_witness.py --config <effective_config.json> --out <dir>
Does not train anything. Emits input_audit.json and witness.json.
"""
from __future__ import absolute_import, division, print_function

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

EMBED_SHA_AB = "5769c715786ffbd3e635078e047284b60c7e1593b2602b5aabee90eab877e787"
INPUT_FILES = ["effective_config.json", "transe_embed.pkl", "kg.pkl", "dataset.pkl", "train_label.pkl",
               "validation_label.pkl", "test_label.pkl"]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def git(*a):
    try:
        return subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()
    except Exception:
        return None


def audit(tmp_dir, ref_dirs=()):
    out = {"files": {}, "missing": [], "embed_sha_expected_AB": EMBED_SHA_AB}
    for f in INPUT_FILES:
        p = os.path.join(tmp_dir, f)
        if os.path.isfile(p):
            out["files"][f] = {"sha256": sha(p), "bytes": os.path.getsize(p)}
        else:
            out["missing"].append(f)
    e = out["files"].get("transe_embed.pkl")
    out["embeddings_match_AB"] = bool(e and e["sha256"] == EMBED_SHA_AB)
    out["ref_dirs"] = {}
    for d in ref_dirs:
        r = {}
        for f in INPUT_FILES:
            p = os.path.join(d, f)
            if os.path.isfile(p) and f in out["files"]:
                r[f] = sha(p) == out["files"][f]["sha256"]
        out["ref_dirs"][d] = r
    out["repo_head"] = git("rev-parse", "HEAD")
    out["repo_dirty_files"] = git("status", "--porcelain")
    out["python"] = platform.python_version()
    try:
        import numpy, torch, scipy, sklearn
        out["packages"] = {"numpy": numpy.__version__, "torch": torch.__version__, "scipy": scipy.__version__,
                           "sklearn": sklearn.__version__, "cuda": torch.version.cuda}
    except Exception as ex:  # pragma: no cover
        out["packages"] = {"error": str(ex)}
    return out


def witness(cfg_path, max_users=None):
    from easydict import EasyDict as edict
    from kg_env import BatchKGEnvironment
    cfg = edict(json.load(open(cfg_path)))
    a = cfg.TRAIN_AGENT
    env = BatchKGEnvironment(a.tmp_dir, cfg.KG_ARGS, a.max_acts, max_path_len=a.max_path_len,
                             state_history=a.state_history, reward_function=a.reward, use_pattern=a.use_pattern,
                             concept_bonus_lambda=5.0)
    sl = cfg.KG_ARGS.self_loop
    uids = list(env.kg("user").keys())
    if max_users:
        uids = uids[:max_users]
    counts = {"users": len(uids), "users_with_B1_3eff_padded": 0, "users_with_B1_4eff": 0,
              "padded_3eff_B1_paths": 0, "eff4_B1_paths": 0, "padded_3eff_total": 0}
    first = None
    for uid in uids:
        found3, found4 = False, False
        p0 = [(sl, "user", uid)]
        for r1, n1 in env._get_actions(p0, False):
            if r1 == sl:
                continue
            p1 = p0 + [(r1, "item", n1)]
            for r2, n2 in env._get_actions(p1, False):
                if r2 == sl:
                    continue
                t2 = cfg.KG_ARGS.kg_relation["item"][r2]
                p2 = p1 + [(r2, t2, n2)]
                for r3, n3 in env._get_actions(p2, False):
                    if r3 == sl:
                        continue
                    t3 = cfg.KG_ARGS.kg_relation[t2][r3]
                    p3 = p2 + [(r3, t3, n3)]
                    if t3 == "item":
                        padded = p3 + [(sl, "item", n3)]  # ACTION self-loop (distinct from the start tuple)
                        counts["padded_3eff_total"] += 1
                        if env.concept_bonus(padded) == 1.0:
                            counts["padded_3eff_B1_paths"] += 1
                            found3 = True
                            if first is None:
                                first = {"uid": uid, "path": padded,
                                         "step_rewards_lambda5": [
                                             float(env.binary_train_reward(padded[:k]) + (5.0 * env.concept_bonus(padded) if k == len(padded) else 0.0))
                                             for k in range(2, len(padded) + 1)],
                                         "concept_bonus_raw": env.concept_bonus(padded),
                                         "binary_train_reward": env.binary_train_reward(padded)}
                    for r4, n4 in env._get_actions(p3, False):
                        if r4 == sl:
                            continue
                        t4 = cfg.KG_ARGS.kg_relation[t3][r4]
                        p4 = p3 + [(r4, t4, n4)]
                        if t4 == "item" and env.concept_bonus(p4) == 1.0:
                            counts["eff4_B1_paths"] += 1
                            found4 = True
        counts["users_with_B1_3eff_padded"] += found3
        counts["users_with_B1_4eff"] += found4
    return {"counts": counts, "first_witness_3eff_padded_to_4_actions": first}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref", nargs="*", default=[])
    ap.add_argument("--max_users", type=int, default=None)
    r = ap.parse_args()
    os.makedirs(r.out, exist_ok=True)
    cfg = json.load(open(r.config))
    au = audit(cfg["TRAIN_AGENT"]["tmp_dir"], r.ref)
    json.dump(au, open(os.path.join(r.out, "input_audit.json"), "w"), indent=2)
    print(json.dumps({k: au[k] for k in ("missing", "embeddings_match_AB", "repo_head")}))
    w = witness(r.config, r.max_users)
    json.dump(w, open(os.path.join(r.out, "witness.json"), "w"), indent=2, default=int)
    print(json.dumps(w["counts"]))
