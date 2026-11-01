"""Spike E: reward-only concept controllability. Matched two-branch short training driver.

Reward: r_shaped(t) = r_base(t) + lambda * I(t = T_train) * B(path)  (see kg_env.concept_bonus).
This is a terminal concept bonus conditioned on a positive train endpoint. It is NOT
potential-based shaping; no optimal-policy preservation is claimed.

Sub-commands (run from repo root, inside src/UPGPR on sys.path):
  make-init : build a fresh ActorCritic with seed S and save its state_dict + sha256.
  train     : train one branch (lambda) from the saved init for N epochs, per-epoch logs/ckpts.
  beam      : run the UNCHANGED baseline beam search (test_agent.predict_paths) for a checkpoint.

Training loop is a faithful copy of train_agent.train (same loader, lr schedule, update),
plus logging. Trajectories/RNG consumption diverge between branches once policies diverge.
"""
from __future__ import absolute_import, division, print_function

import argparse
import hashlib
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.optim as optim
from easydict import EasyDict as edict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from actor_critic import ActorCritic  # noqa: E402
from kg_env import BatchKGEnvironment  # noqa: E402
from train_agent import ACDataLoader  # noqa: E402
from utils import set_random_seed  # noqa: E402


def sd_hash(sd):
    """sha256 over (name, dtype, shape, raw bytes) in sorted-name order."""
    h = hashlib.sha256()
    for k in sorted(sd.keys()):
        t = sd[k].detach().cpu().contiguous()
        h.update(k.encode())
        h.update(str(t.dtype).encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def embed_hash(env):
    """sha256 over the embedding arrays the env holds (tensor-level, sorted keys). Embeddings are numpy,
    outside the torch graph/optimizer; the before/after hash proves they were not touched."""
    h = hashlib.sha256()
    for k in sorted(env.embeds.keys()):
        v = env.embeds[k]
        arr = v[0] if isinstance(v, tuple) else v
        a = np.ascontiguousarray(np.asarray(arr, dtype=np.float64))
        h.update(k.encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def file_sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _load_cfg(path):
    with open(path) as f:
        return edict(json.load(f))


def _build(cfg, lam):
    args = cfg.TRAIN_AGENT
    env = BatchKGEnvironment(
        args.tmp_dir, cfg.KG_ARGS, args.max_acts, max_path_len=args.max_path_len,
        state_history=args.state_history, reward_function=args.reward,
        use_pattern=args.use_pattern, concept_bonus_lambda=lam)
    return env


def _new_model(env, args, device):
    return ActorCritic(env.state_dim, env.act_dim, gamma=args.gamma, hidden_sizes=args.hidden,
                       modified_policy=args.modified_policy, embed_size=env.embed_size).to(device)


def make_init(cfg_path, out_path, seed=None):
    cfg = _load_cfg(cfg_path)
    args = cfg.TRAIN_AGENT
    if seed is not None:
        args.seed = seed
    set_random_seed(args.seed)
    env = _build(cfg, 0.0)
    model = _new_model(env, args, "cpu")
    sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(sd, out_path)
    info = {"init_state_dict_sha256": sd_hash(sd), "file_sha256": file_sha256(out_path),
            "seed": args.seed, "state_dim": env.state_dim, "act_dim": env.act_dim}
    json.dump(info, open(out_path + ".json", "w"), indent=2)
    print(json.dumps(info))
    return info


def _terminal_stats(env, base_step, shaped_step):
    """Stats of the terminal paths of the current batch (training trajectories)."""
    sl = env.kg_args.self_loop
    paths = env._batch_path
    n = len(paths)
    eff = np.array([sum(1 for r, _, _ in p[1:] if r != sl) for p in paths], dtype=float)
    has_sl = np.array([any(r == sl for r, _, _ in p[1:]) for p in paths], dtype=float)
    has_concept = np.array([any(r == "item_concept" for r, _, _ in p[1:]) for p in paths], dtype=float)
    end_item = np.array([p[-1][1] == "item" for p in paths], dtype=float)
    pos_end = (np.asarray(base_step) > 0).astype(float)  # terminal base reward > 0 (valid train positive)
    bonus = np.asarray(env._batch_curr_bonus, dtype=float)
    return {"n": n, "eff_len_sum": eff.sum(), "self_loop_sum": has_sl.sum(), "concept_sum": has_concept.sum(),
            "end_item_sum": end_item.sum(), "pos_end_sum": pos_end.sum(), "bonus_sum": bonus.sum(),
            "concept_pos_sum": (has_concept * pos_end).sum(),
            "eff_len_hist": np.bincount(eff.astype(int), minlength=6).tolist()}


def log_bonus_share(acc, lam):
    """lambda*bonus / (base + lambda*bonus) over the epoch's episodes (None if total return is 0)."""
    b = lam * acc["bonus_raw_sum"]
    tot = acc["base_sum"] + b
    return (b / tot) if tot else None


def train_branch(cfg_path, branch, lam, init_path, out_dir, epochs=5, resume=False, seed=None, save_every=1,
                 max_users=None):
    import resource
    cfg = _load_cfg(cfg_path)
    args = cfg.TRAIN_AGENT
    if seed is not None:
        args.seed = seed
    args.epochs = epochs  # lr-decay horizon = the epochs actually run (M5: 50, fixed from epoch 1)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    args.device = device
    os.makedirs(os.path.join(out_dir, "ckpt"), exist_ok=True)

    set_random_seed(args.seed)
    env = _build(cfg, lam)
    uids = list(env.kg("user").keys())
    if max_users:  # smoke only: subset of users, never used for official runs
        uids = uids[:max_users]
    dataloader = ACDataLoader(uids, args.batch_size)
    model = _new_model(env, args, device)  # consumes torch RNG exactly as in baseline
    init_sd = torch.load(init_path, map_location="cpu")
    model.load_state_dict(init_sd)
    init_hash = sd_hash(model.state_dict())
    emb_hash_before = embed_hash(env)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    model.train()

    state_file = os.path.join(out_dir, "ckpt", "state_latest.pt")
    start_epoch, step = 1, 0
    logs = []
    if resume and os.path.isfile(state_file):
        st = torch.load(state_file, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"]); optimizer.load_state_dict(st["opt"])
        random.setstate(st["py_rng"]); np.random.set_state(st["np_rng"])
        torch.set_rng_state(st["torch_rng"])
        if torch.cuda.is_available() and st.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(st["cuda_rng"])
        start_epoch, step, logs = st["epoch"] + 1, st["step"], st["logs"]
        print("RESUMED at epoch", start_epoch)

    meta = {"branch": branch, "lambda": lam, "init_sha256": init_hash, "epochs": epochs, "seed": args.seed,
            "embeddings_sha256_before": emb_hash_before, "save_every": save_every,
            "lr_schedule": "linear decay max(1e-4, 1 - step/(epochs*n_users/batch)) with epochs=%d from epoch 1" % epochs,
            "device": str(device), "torch": torch.__version__, "numpy": np.__version__,
            "note": "trajectories and RNG consumption diverge between branches once policies diverge"}
    meta["effective_train_agent_config"] = {k: (str(v) if k == "device" else v) for k, v in args.items()}
    json.dump(meta, open(os.path.join(out_dir, "train_meta.json"), "w"), indent=2)

    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()
        dataloader.reset()
        acc = {"base_sum": 0.0, "bonus_raw_sum": 0.0, "total_sum": 0.0, "episodes": 0, "n": 0,
               "eff_len_sum": 0.0, "self_loop_sum": 0.0, "concept_sum": 0.0, "end_item_sum": 0.0,
               "pos_end_sum": 0.0, "bonus_sum": 0.0, "concept_pos_sum": 0.0,
               "eff_len_hist": np.zeros(6), "loss": [], "ploss": [], "vloss": [], "ent": [],
               "nonfinite": 0, "early_bonus_violations": 0}
        while dataloader.has_next():
            batch_uids = dataloader.get_batch()
            batch_state = env.reset(batch_uids)
            done = False
            bs_base = np.zeros(len(batch_uids)); bs_total = np.zeros(len(batch_uids))
            term = None
            while not done:
                mask = env.batch_action_mask(dropout=args.act_dropout)
                emb = env.batch_action_embeddings()
                act = model.select_action(batch_state, mask, emb, args.device)
                batch_state, batch_reward, done = env.batch_step(act)
                if not done and np.any(env._batch_curr_bonus != 0):
                    acc["early_bonus_violations"] += 1
                bs_base += env._batch_curr_base_reward
                bs_total += batch_reward
                if done:
                    term = _terminal_stats(env, env._batch_curr_base_reward, batch_reward)
                model.rewards.append(batch_reward)
            acc["base_sum"] += bs_base.sum(); acc["total_sum"] += bs_total.sum()
            acc["bonus_raw_sum"] += term["bonus_sum"]; acc["episodes"] += len(batch_uids)
            for k in ("n", "eff_len_sum", "self_loop_sum", "concept_sum", "end_item_sum", "pos_end_sum",
                      "bonus_sum", "concept_pos_sum"):
                acc[k] += term[k]
            acc["eff_len_hist"] += np.array(term["eff_len_hist"])

            lr = args.lr * max(1e-4, 1.0 - float(step) / (args.epochs * len(uids) / args.batch_size))
            for pg in optimizer.param_groups:
                pg["lr"] = lr
            loss, ploss, vloss, eloss = model.update(optimizer, args.device, args.ent_weight)
            for k, v in (("loss", loss), ("ploss", ploss), ("vloss", vloss), ("ent", eloss)):
                acc[k].append(v)
                if not np.isfinite(v):
                    acc["nonfinite"] += 1
            step += 1

        n = max(acc["n"], 1)
        log = {
            "epoch": epoch, "branch": branch, "lambda": lam, "steps": step, "episodes": acc["episodes"],
            "mean_base_return_per_episode": acc["base_sum"] / acc["episodes"],
            "mean_bonus_raw_per_episode": acc["bonus_raw_sum"] / acc["episodes"],
            "mean_bonus_times_lambda_per_episode": lam * acc["bonus_raw_sum"] / acc["episodes"],
            "mean_total_return_per_episode": acc["total_sum"] / acc["episodes"],
            "frac_terminal_B1": acc["bonus_sum"] / n,
            "frac_terminal_concept_any": acc["concept_sum"] / n,
            "frac_terminal_positive_endpoint": acc["pos_end_sum"] / n,
            "frac_terminal_item_endpoint": acc["end_item_sum"] / n,
            "frac_positive_endpoint_with_concept": (acc["concept_pos_sum"] / acc["pos_end_sum"]) if acc["pos_end_sum"] else None,
            "mean_effective_length": acc["eff_len_sum"] / n,
            "eff_len_hist_0to5": acc["eff_len_hist"].tolist(),
            "frac_with_action_self_loop": acc["self_loop_sum"] / n,
            "mean_loss": float(np.mean(acc["loss"])), "mean_ploss": float(np.mean(acc["ploss"])),
            "mean_vloss": float(np.mean(acc["vloss"])), "mean_neg_entropy_loss": float(np.mean(acc["ent"])),
            "nonfinite_loss_count": acc["nonfinite"], "early_bonus_violations": acc["early_bonus_violations"],
            "wall_time_s": time.time() - t0, "lr_end": lr,
            "bonus_share_of_total_return": (log_bonus_share(acc, lam)),
            "max_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        }
        logs.append(log)
        sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        if epoch % save_every == 0 or epoch == epochs:
            torch.save(sd, os.path.join(out_dir, "ckpt", "epoch_{}.pt".format(epoch)))
        json.dump(logs, open(os.path.join(out_dir, "train_log.json"), "w"), indent=2)
        torch.save({"model": sd, "opt": optimizer.state_dict(), "py_rng": random.getstate(),
                    "np_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                    "epoch": epoch, "step": step, "logs": logs}, state_file)
        print(json.dumps({k: log[k] for k in ("epoch", "mean_base_return_per_episode", "mean_bonus_raw_per_episode",
                                               "frac_terminal_B1", "frac_terminal_positive_endpoint", "wall_time_s")}))
    emb_hash_after = embed_hash(env)
    assert emb_hash_after == emb_hash_before, "embeddings changed during training"
    hashes = {"init_sha256": init_hash, "embeddings_sha256_before": emb_hash_before,
              "embeddings_sha256_after": emb_hash_after, "embeddings_unchanged": True}
    for e in range(1, epochs + 1):
        p = os.path.join(out_dir, "ckpt", "epoch_{}.pt".format(e))
        if os.path.isfile(p):
            hashes["epoch_{}_sha256".format(e)] = sd_hash(torch.load(p))
    json.dump(hashes, open(os.path.join(out_dir, "checkpoint_hashes.json"), "w"), indent=2)
    return logs


def beam(cfg_path, ckpt, path_file, topk=None, data="test", seed=None):
    """Unchanged baseline beam search on the TEST_AGENT config (train4/eval3, topk [25,5,1]).
    data in {"test","validation"} selects which split's users the beam is run for."""
    from test_agent import predict_paths
    cfg = _load_cfg(cfg_path)
    args = cfg.TEST_AGENT
    args.device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    args.use_wandb = False
    set_random_seed(args.seed if seed is None else seed)
    t0 = time.time()
    predict_paths(ckpt, path_file, args, cfg.KG_ARGS, data=data)
    return time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("make-init"); a.add_argument("--config", required=True); a.add_argument("--out", required=True)
    a.add_argument("--seed", type=int, default=None)
    b = sub.add_parser("train")
    b.add_argument("--config", required=True); b.add_argument("--branch", required=True)
    b.add_argument("--lam", type=float, required=True); b.add_argument("--init", required=True)
    b.add_argument("--out_dir", required=True); b.add_argument("--epochs", type=int, default=5)
    b.add_argument("--resume", action="store_true"); b.add_argument("--seed", type=int, default=None)
    b.add_argument("--save_every", type=int, default=1); b.add_argument("--max_users", type=int, default=None)
    c =sub.add_parser("beam")
    c.add_argument("--config", required=True); c.add_argument("--ckpt", required=True)
    c.add_argument("--path_file", required=True); c.add_argument("--data", default="test")
    c.add_argument("--seed", type=int, default=None)
    r = ap.parse_args()
    if r.cmd == "make-init":
        make_init(r.config, r.out, r.seed)
    elif r.cmd == "train":
        train_branch(r.config, r.branch, r.lam, r.init, r.out_dir, r.epochs, r.resume, r.seed, r.save_every, r.max_users)
    else:
        print("beam seconds", beam(r.config, r.ckpt, r.path_file, data=r.data, seed=r.seed))


if __name__ == "__main__":
    main()
