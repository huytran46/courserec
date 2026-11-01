"""M5 orchestrator (idempotent: every step is skipped if its output exists; training resumes).

  python src/UPGPR/m5_run.py D --root <m5_root> --config <effective_config.json> --spike_e_init <init_state_dict.pt>
  python src/UPGPR/m5_run.py F --root <m5_root> --config <effective_config.json>
  python src/UPGPR/m5_run.py aggregate --root <m5_root>

Phase D (validation only): seed 23, lambda in {0,1,5}, 50 epochs, epoch-50 checkpoint, validation beam, S0/S3,
selection rule of the spec -> selection.json (written once, never overwritten).
Phase F (test only after selection is locked): seeds 23/24/25 x {control(l=0), shaped(l*)}.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
EPOCHS = 50
LAMBDAS = [0.0, 1.0, 5.0]
SEEDS = [23, 24, 25]
NDCG_FLOOR = 0.95
GAIN_MIN = 0.02
TIE = 0.001


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def uid_hash(uids):
    return hashlib.sha256(json.dumps(sorted(int(u) for u in uids)).encode()).hexdigest()


def lam_name(l):
    return "lambda%g" % l


def arm_dir(root, seed, lam):
    """seed 23 arms are the development runs (reused for the final comparison)."""
    if seed == 23:
        return os.path.join(root, "development", "seed23", lam_name(lam))
    return os.path.join(root, "final", "seed%d" % seed, lam_name(lam))


def spawn(cmd, logfile):
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    return subprocess.Popen(cmd, stdout=open(logfile, "a"), stderr=subprocess.STDOUT, env=env, cwd=os.path.dirname(os.path.dirname(HERE)))


def wait_all(procs):
    bad = []
    for name, p in procs:
        rc = p.wait()
        log("finished", name, "rc", rc)
        if rc != 0:
            bad.append(name)
    if bad:
        raise SystemExit("FAILED steps (log kept, one same-config retry allowed): %s" % bad)


def train_cmd(cfg, lam, seed, init, out):
    return [sys.executable, os.path.join(HERE, "spike_e_train.py"), "train", "--config", cfg, "--branch",
            "control" if lam == 0 else "shaped", "--lam", str(lam), "--init", init, "--out_dir", out,
            "--epochs", str(EPOCHS), "--seed", str(seed), "--save_every", "5", "--resume"]


def train_many(jobs):
    procs = []
    for cfg, lam, seed, init, out in jobs:
        if os.path.isfile(os.path.join(out, "checkpoint_hashes.json")):
            log("skip train (done)", out)
            continue
        os.makedirs(out, exist_ok=True)
        procs.append((out, spawn(train_cmd(cfg, lam, seed, init, out), os.path.join(out, "train.log"))))
        log("launched train", out)
    wait_all(procs)


def beam_many(cfg, jobs, data):
    procs = []
    for seed, out in jobs:
        bdir = os.path.join(out, "beam_" + data)
        pk = os.path.join(bdir, "beam.pkl")
        if os.path.isfile(pk):
            log("skip beam (done)", pk)
            continue
        os.makedirs(bdir, exist_ok=True)
        ckpt = os.path.join(out, "ckpt", "epoch_%d.pt" % EPOCHS)
        cmd = [sys.executable, os.path.join(HERE, "spike_e_train.py"), "beam", "--config", cfg, "--ckpt", ckpt,
               "--path_file", pk + ".tmp", "--data", data, "--seed", str(seed)]
        procs.append((pk, spawn(cmd, os.path.join(bdir, "beam.log"))))
        log("launched beam", pk)
    wait_all(procs)
    for seed, out in jobs:
        pk = os.path.join(out, "beam_" + data, "beam.pkl")
        if not os.path.isfile(pk) and os.path.isfile(pk + ".tmp"):
            os.replace(pk + ".tmp", pk)


def evaluate(cfg, out, data):
    """Evaluate one policy's beam with the M5 metric contract. Writes tables + per-user rows. Returns summary."""
    import m5_metrics as mm
    c = json.load(open(cfg))
    res_path = os.path.join(out, "eval_%s.json" % data)
    if os.path.isfile(res_path):
        return json.load(open(res_path))
    t0 = time.time()
    res, rows = mm.evaluate_beam(os.path.join(out, "beam_" + data, "beam.pkl"), c["TEST_AGENT"]["tmp_dir"],
                                 c["KG_ARGS"], data)
    elig = res.pop("eligible_users")
    res["eligible_users_sha256"] = uid_hash(elig)
    res["beam_sha256"] = sha_file(os.path.join(out, "beam_" + data, "beam.pkl"))
    res["eval_seconds"] = time.time() - t0
    res["units"] = "fractions in [0,1]"
    json.dump(res, open(res_path, "w"), indent=2)
    for sel, rr in rows.items():
        with open(os.path.join(out, "per_user_%s_%s.csv" % (data, sel)), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["uid", "n_items", "hits", "precision", "recall", "ndcg", "hr", "mrr", "n_valid_expl",
                        "n_concept_expl", "concept_at10", "n_candidates", "n_paths", "fam_json"])
            for u in sorted(rr):
                r = rr[u]
                w.writerow([u, r["n_items"], r["hits"], r["precision"], r["recall"], r["ndcg"], r["hr"], r["mrr"],
                            r["n_valid_expl"], r["n_concept_expl"], r["concept_at10"], r["n_candidates"],
                            r["n_paths"], json.dumps(r["fam"], sort_keys=True)])
    return res


def manifest(root, cfg):
    c = json.load(open(cfg))
    tmp = c["TRAIN_AGENT"]["tmp_dir"]
    files = {n: sha_file(os.path.join(tmp, n)) for n in ["kg.pkl", "train_label.pkl", "validation_label.pkl",
                                                        "test_label.pkl", "transe_embed.pkl", "dataset.pkl"]}
    code = {f: sha_file(os.path.join(HERE, f)) for f in sorted(os.listdir(HERE)) if f.endswith(".py")}
    try:
        import torch
        tv = torch.__version__
    except Exception:
        tv = None
    m = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "python": platform.python_version(), "torch": tv,
         "platform": platform.platform(), "cpu_count": os.cpu_count(), "cuda": False, "input_sha256": files,
         "code_sha256": code, "config_sha256": sha_file(cfg), "config": c,
         "note": "code = upstream 7348dc2 + Spike E/M5 patch; deterministic flags: seeds python/numpy/torch set per run"}
    os.makedirs(os.path.join(root, "manifests"), exist_ok=True)
    json.dump(m, open(os.path.join(root, "manifests", "manifest_%s.json" % time.strftime("%Y%m%d_%H%M%S")), "w"), indent=2)


def select_lambda(val):
    """val: {lam: eval dict}. Returns (selection dict)."""
    base = val[0.0]
    floor = NDCG_FLOOR * base["S3"]["ndcg"]
    cand = {}
    for lam, r in val.items():
        if lam == 0.0:
            continue
        gain = r["S3"]["concept_at10"] - base["S3"]["concept_at10"]
        ok_u = r["S3"]["ndcg"] >= floor
        ok_g = gain >= GAIN_MIN
        cand[lam] = {"ndcg_S3": r["S3"]["ndcg"], "ndcg_ratio_vs_control": r["S3"]["ndcg"] / base["S3"]["ndcg"],
                     "concept_at10_S3": r["S3"]["concept_at10"], "gain_vs_control_S3": gain,
                     "utility_ok": ok_u, "gain_ok": ok_g, "eligible": ok_u and ok_g}
    elig = {l: v for l, v in cand.items() if v["eligible"]}
    chosen = None
    if elig:
        best = max(v["concept_at10_S3"] for v in elig.values())
        near = sorted(l for l, v in elig.items() if best - v["concept_at10_S3"] <= TIE)
        chosen = near[0]  # smaller lambda on near-ties
    return cand, chosen


def phase_D(a):
    root = os.path.abspath(a.root)
    os.makedirs(os.path.join(root, "inits"), exist_ok=True)
    init23 = os.path.join(root, "inits", "seed23.pt")
    if not os.path.isfile(init23):
        shutil.copy(a.spike_e_init, init23)
    manifest(root, a.config)
    train_many([(a.config, l, 23, init23, arm_dir(root, 23, l)) for l in LAMBDAS])
    beam_many(a.config, [(23, arm_dir(root, 23, l)) for l in LAMBDAS], "validation")
    val = {l: evaluate(a.config, arm_dir(root, 23, l), "validation") for l in LAMBDAS}
    sel_path = os.path.join(root, "selection.json")
    if os.path.isfile(sel_path):
        log("selection.json already locked; not rewriting")
        return
    cand, chosen = select_lambda(val)
    hashes = {}
    for l in LAMBDAS:
        d = arm_dir(root, 23, l)
        hashes[lam_name(l)] = {"checkpoint_epoch50_sha256": json.load(open(os.path.join(d, "checkpoint_hashes.json"))).get("epoch_50_sha256"),
                               "beam_sha256": val[l]["beam_sha256"], "eligible_users_sha256": val[l]["eligible_users_sha256"]}
    sel = {"locked_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "split": "validation", "seed": 23, "epoch": EPOCHS,
           "target": "macro concept@10 of S3 (fixed-slot, missing slots count 0)",
           "rule": {"ndcg_floor_ratio": NDCG_FLOOR, "min_gain_pp": GAIN_MIN * 100, "tie_pp": TIE * 100,
                    "tie_break": "smaller lambda"},
           "control": {"ndcg_S3": val[0.0]["S3"]["ndcg"], "concept_at10_S3": val[0.0]["S3"]["concept_at10"]},
           "candidates": {lam_name(l): v for l, v in cand.items()},
           "lambda_star": chosen, "status": "SELECTED" if chosen is not None else "NO_FEASIBLE_LAMBDA",
           "hashes": hashes, "validation_grid": {lam_name(l): {s: {k: val[l][s][k] for k in ("ndcg", "recall", "precision", "hr", "mrr", "concept_at10")} for s in ("S0", "S3")} for l in LAMBDAS}}
    json.dump(sel, open(sel_path, "w"), indent=2)
    os.chmod(sel_path, 0o444)
    log("SELECTION LOCKED:", sel["status"], "lambda*", chosen)


def phase_F(a):
    root = os.path.abspath(a.root)
    sel = json.load(open(os.path.join(root, "selection.json")))
    if sel["lambda_star"] is None:
        log("NO_FEASIBLE_LAMBDA: stop (no test evaluation, no seeds)")
        return
    ls = float(sel["lambda_star"])
    manifest(root, a.config)
    jobs = []
    for s in SEEDS:
        init = os.path.join(root, "inits", "seed%d.pt" % s)
        if not os.path.isfile(init):
            subprocess.run([sys.executable, os.path.join(HERE, "spike_e_train.py"), "make-init", "--config", a.config,
                            "--out", init, "--seed", str(s)], check=True, cwd=os.path.dirname(os.path.dirname(HERE)))
        if s != 23:
            for l in (0.0, ls):
                jobs.append((a.config, l, s, init, arm_dir(root, s, l)))
    train_many(jobs)
    pol = [(s, arm_dir(root, s, l)) for s in SEEDS for l in (0.0, ls)]
    beam_many(a.config, pol, "test")
    for s, d in pol:
        evaluate(a.config, d, "test")
    log("phase F done")


def phase_aggregate(a):
    import numpy as np
    root = os.path.abspath(a.root)
    sel = json.load(open(os.path.join(root, "selection.json")))
    ls = float(sel["lambda_star"])
    out = {"lambda_star": ls, "n_seeds": len(SEEDS), "units": "fractions in [0,1]", "per_seed": {}}
    keys = ["ndcg", "recall", "precision", "hr", "mrr", "concept_at10", "valid_expl_per10", "expl_per_item"]
    for s in SEEDS:
        c = json.load(open(os.path.join(arm_dir(root, s, 0.0), "eval_test.json")))
        h = json.load(open(os.path.join(arm_dir(root, s, ls), "eval_test.json")))
        out["per_seed"][s] = {"control": c, "shaped": h}
    def agg(f):
        v = np.array([f(out["per_seed"][s]) for s in SEEDS], dtype=float)
        return {"per_seed": v.tolist(), "mean": float(v.mean()), "sd_sample": float(v.std(ddof=1))}
    out["aggregate"] = {}
    for arm in ("control", "shaped"):
        for sel_ in ("S0", "S3"):
            out["aggregate"]["%s_%s" % (arm, sel_)] = {k: agg(lambda p, arm=arm, sel_=sel_, k=k: p[arm][sel_][k]) for k in keys}
        out["aggregate"]["%s_availability_pooled" % arm] = agg(lambda p, arm=arm: p[arm]["availability"]["pooled"])
        out["aggregate"]["%s_candidate_recall" % arm] = agg(lambda p, arm=arm: p[arm]["availability"]["candidate_recall_pooled"])
    out["paired"] = {
        "shaped_minus_control_S0_concept@10": agg(lambda p: p["shaped"]["S0"]["concept_at10"] - p["control"]["S0"]["concept_at10"]),
        "shaped_minus_control_S3_concept@10": agg(lambda p: p["shaped"]["S3"]["concept_at10"] - p["control"]["S3"]["concept_at10"]),
        "gain_S3_over_S0_control": agg(lambda p: p["control"]["S3"]["concept_at10"] - p["control"]["S0"]["concept_at10"]),
        "gain_S3_over_S0_shaped": agg(lambda p: p["shaped"]["S3"]["concept_at10"] - p["shaped"]["S0"]["concept_at10"]),
        "interaction": agg(lambda p: (p["shaped"]["S3"]["concept_at10"] - p["shaped"]["S0"]["concept_at10"]) - (p["control"]["S3"]["concept_at10"] - p["control"]["S0"]["concept_at10"])),
        "shaped_minus_control_ndcg@10": agg(lambda p: p["shaped"]["S0"]["ndcg"] - p["control"]["S0"]["ndcg"]),
        "shaped_minus_control_availability": agg(lambda p: p["shaped"]["availability"]["pooled"] - p["control"]["availability"]["pooled"]),
    }
    os.makedirs(os.path.join(root, "tables"), exist_ok=True)
    json.dump(out, open(os.path.join(root, "summary.json"), "w"), indent=2)
    with open(os.path.join(root, "tables", "arms_x_selectors.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "arm", "selector"] + keys)
        for s in SEEDS:
            for arm in ("control", "shaped"):
                for sel_ in ("S0", "S3"):
                    w.writerow([s, arm, sel_] + [out["per_seed"][s][arm][sel_][k] for k in keys])
    log("wrote summary.json and tables/arms_x_selectors.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["D", "F", "aggregate"])
    ap.add_argument("--root", required=True)
    ap.add_argument("--config")
    ap.add_argument("--spike_e_init")
    a = ap.parse_args()
    {"D": phase_D, "F": phase_F, "aggregate": phase_aggregate}[a.phase](a)


if __name__ == "__main__":
    main()
