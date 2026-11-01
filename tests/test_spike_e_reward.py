"""Spike E preflight: terminal concept bonus. Expected rewards are hand-computed (see comments)."""
import importlib.util
import logging
import os
import pickle
import random
import subprocess
import sys

import numpy as np
import pytest
import torch
from easydict import EasyDict as edict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src", "UPGPR")
sys.path.insert(0, SRC)
import kg_env  # noqa: E402
import spike_e_train as se  # noqa: E402

SL = "self_loop"
REL = {"user": {"enroll": "item"},
       "item": {"enroll": "user", "item_concept": "concept", "item_teacher": "teacher", "item_school": "school"},
       "school": {"item_school": "item"}, "concept": {"item_concept": "item"}, "teacher": {"item_teacher": "item"}}


class FakeKG:
    def __init__(self, G):
        self.G = G

    def __call__(self, t, i=None, r=None):
        d = self.G[t]
        if i is not None:
            d = d[i]
        if r is not None:
            d = d[r]
        return d


def build_dir(tmp_path, n_items=5, d=4):
    """Hand-built graph (all edges both directions).
    users: u0 enrolls item0; u1 enrolls item0,1,2.   train labels u0={0,1,2}, u1={0,1,2}; test u0={3}
    concept c0 <-> items {0,1,3};  teacher t0 <-> items {0,1,2};  item4 has no edges.
    """
    G = {"user": {u: {"enroll": []} for u in (0, 1)},
         "item": {i: {"enroll": [], "item_concept": [], "item_teacher": [], "item_school": []} for i in range(n_items)},
         "concept": {0: {"item_concept": []}}, "teacher": {0: {"item_teacher": []}}, "school": {0: {"item_school": []}}}
    for u, its in ((0, [0]), (1, [0, 1, 2])):
        for i in its:
            G["user"][u]["enroll"].append(i); G["item"][i]["enroll"].append(u)
    for i in (0, 1, 3):
        G["concept"][0]["item_concept"].append(i); G["item"][i]["item_concept"].append(0)
    for i in (0, 1, 2):
        G["teacher"][0]["item_teacher"].append(i); G["item"][i]["item_teacher"].append(0)
    rng = np.random.default_rng(0)
    emb = {"user": rng.normal(size=(2, d)), "item": rng.normal(size=(n_items, d)), "concept": rng.normal(size=(1, d)),
           "teacher": rng.normal(size=(1, d)), "school": rng.normal(size=(1, d))}
    for r in ("enroll", "item_concept", "item_teacher", "item_school"):
        emb[r] = (rng.normal(size=d), 0.0)
    pickle.dump(FakeKG(G), open(tmp_path / "kg.pkl", "wb"))
    pickle.dump(emb, open(tmp_path / "transe_embed.pkl", "wb"))
    pickle.dump({0: [0, 1, 2], 1: [0, 1, 2]}, open(tmp_path / "train_label.pkl", "wb"))
    pickle.dump({0: [3], 1: [3]}, open(tmp_path / "test_label.pkl", "wb"))
    pickle.dump({0: [3], 1: [3]}, open(tmp_path / "validation_label.pkl", "wb"))
    return edict({"self_loop": SL, "kg_relation": REL, "interaction": "enroll", "path_pattern": {}})


def mk_env(tmp_path, lam, mod=kg_env, path_len=4, **kw):
    kg_args = build_dir(tmp_path)
    kwargs = dict(max_path_len=path_len, state_history=2, reward_function="binary_train", use_pattern=False)
    if lam is not None:
        kwargs["concept_bonus_lambda"] = lam
    env = mod.BatchKGEnvironment(str(tmp_path), kg_args, 250, **kwargs)
    return env, kg_args


def act(env, i, rel, node):
    return [k for k, a in enumerate(env._batch_curr_actions[i]) if a == (rel, node)][0]


def walk(env, uid, steps):
    """steps: list of (relation,node_id). Returns list of per-step rewards and bonuses."""
    env.reset([uid])
    rewards, bonuses = [], []
    for rel, node in steps:
        _, r, _ = env.batch_step([act(env, 0, rel, node)])
        rewards.append(float(r[0]))
        bonuses.append(float(env._batch_curr_bonus[0]))
    return rewards, bonuses


# u0: enroll->item0 -> concept0 -> item1 ; item1 is in u0's train set
P_CONCEPT = [("enroll", 0), ("item_concept", 0), ("item_concept", 1), (SL, 1)]


def test_lambda0_identical_to_original_env_every_step(tmp_path):
    """Original kg_env from git HEAD vs new env with lambda=0 and lambda unset, random walks."""
    src = subprocess.run(["git", "show", "5334bf9:src/UPGPR/kg_env.py"], cwd=ROOT, capture_output=True, text=True).stdout
    spec = importlib.util.spec_from_loader("kg_env_orig", loader=None)
    orig = importlib.util.module_from_spec(spec)
    exec(compile(src, "kg_env_orig", "exec"), orig.__dict__)
    e_old, _ = mk_env(tmp_path, None, mod=orig)
    e_zero, _ = mk_env(tmp_path, 0.0)
    e_unset, _ = mk_env(tmp_path, None)
    rng = random.Random(1)
    n_checked = 0
    for ep in range(60):
        uid = rng.choice([0, 1])
        for e in (e_old, e_zero, e_unset):
            e.reset([uid])
        done = False
        while not done:
            idx = rng.randrange(len(e_old._batch_curr_actions[0]))
            outs = [e.batch_step([idx]) for e in (e_old, e_zero, e_unset)]
            assert outs[0][1][0] == outs[1][1][0] == outs[2][1][0]
            done = outs[0][2]
            n_checked += 1
    assert n_checked >= 120


def test_valid_concept_path_bonus_only_at_terminal(tmp_path):
    # base: step1 path len 2 ->0 ; step2 ends at concept ->0 ; step3 ends at train item1, set size 4 ->1 ;
    # step4 self-loop pad: set size 5 > 3 -> 1.  bonus only at step 4 (terminal, T_train=4): raw B=1, 5*1.
    env, _ = mk_env(tmp_path, 5.0)
    r, b = walk(env, 0, P_CONCEPT)
    assert r == [0.0, 0.0, 1.0, 6.0]
    assert b == [0.0, 0.0, 0.0, 1.0]
    env0, _ = mk_env(tmp_path, 0.0)
    r0, b0 = walk(env0, 0, P_CONCEPT)
    assert r0 == [0.0, 0.0, 1.0, 1.0]
    assert b0 == [0.0, 0.0, 0.0, 1.0]  # raw B still logged; lambda=0 adds nothing


def test_three_effective_hops_padded_with_self_loop_gets_bonus(tmp_path):
    """u0->item0-(concept0)->item1 is 3 effective hops; pad to 4 actions with an ACTION self-loop."""
    env, _ = mk_env(tmp_path, 5.0)
    walk(env, 0, P_CONCEPT)
    path = env._batch_path[0]
    assert path[0] == (SL, "user", 0)  # start tuple self_loop is not an action
    assert [p[0] for p in path[1:]].count(SL) == 1 and path[-1][0] == SL
    assert sum(1 for p in path[1:] if p[0] != SL) == 3
    assert env.concept_bonus(path) == 1.0
    assert env.binary_train_reward(path) == 1.0


def test_four_effective_hops_concept_terminal(tmp_path):
    # u0 -> item0 -> concept0 -> item1 -> teacher0 ... item2 is in train. effective length 4, no self loop.
    env, _ = mk_env(tmp_path, 5.0)
    r, b = walk(env, 0, [("enroll", 0), ("item_concept", 0), ("item_concept", 1), ("item_teacher", 0)])
    # step4 ends at teacher (non item) -> base 0, bonus 0
    assert r == [0.0, 0.0, 1.0, 0.0] and b[-1] == 0.0
    # u1 -> item1 -> teacher0 -> item2 -> self loop : no concept
    r, b = walk(env, 1, [("enroll", 1), ("item_teacher", 0), ("item_teacher", 2), (SL, 2)])
    assert r == [0.0, 0.0, 1.0, 1.0] and b == [0.0] * 4


def test_no_bonus_no_concept_nonitem_or_item_outside_train(tmp_path):
    env, _ = mk_env(tmp_path, 5.0)
    # no concept: u0->item0->teacher0->item1->self loop : base 0,0,1,1 ; bonus 0
    r, b = walk(env, 0, [("enroll", 0), ("item_teacher", 0), ("item_teacher", 1), (SL, 1)])
    assert r == [0.0, 0.0, 1.0, 1.0] and sum(b) == 0
    # non-item endpoint: u0->item0->concept0->(self)->(self) ends at concept
    r, b = walk(env, 0, [("enroll", 0), ("item_concept", 0), (SL, 0), (SL, 0)])
    assert r == [0.0, 0.0, 0.0, 0.0] and sum(b) == 0
    # item outside u0 train set: item3 reached through concept0 (item3 only in test labels)
    r, b = walk(env, 0, [("enroll", 0), ("item_concept", 0), ("item_concept", 3), (SL, 3)])
    assert r == [0.0, 0.0, 0.0, 0.0] and sum(b) == 0


def test_trivial_and_repeated_prefix_gated_to_zero(tmp_path):
    env, _ = mk_env(tmp_path, 5.0)
    # trivial: u0->item0 then self loops only: effective length 1, no concept
    r, b = walk(env, 0, [("enroll", 0), (SL, 0), (SL, 0), (SL, 0)])
    assert r == [0.0, 0.0, 0.0, 0.0] and sum(b) == 0
    # handcrafted: concept edge repeated (prefix repeated) + self loop, endpoint train item0.
    # eff length 2, concept present, endpoint in train, BUT len(set(path)) == 3 with a self loop
    # -> baseline validity gate (binary_train_reward) = 0 -> bonus must be 0.
    X = ("item_concept", "item", 0)
    p = [(SL, "user", 0), X, X, (SL, "item", 0)]
    assert len(set(p)) == 3
    assert env.binary_train_reward(p) == 0.0
    assert env.concept_bonus(p) == 0.0
    # same path without the validity failure (different repeated structure) is bonused
    q = [(SL, "user", 0), ("enroll", "item", 0), ("item_concept", "concept", 0), ("item_concept", "item", 1)]
    assert env.binary_train_reward(q) == 1.0 and env.concept_bonus(q) == 1.0
    # short paths
    assert env.concept_bonus([(SL, "user", 0)]) == 0.0
    assert env.concept_bonus([(SL, "user", 0), ("enroll", "item", 0)]) == 0.0


def test_bonus_uses_train_labels_only_not_test_or_validation(tmp_path):
    env, _ = mk_env(tmp_path, 5.0)
    assert 3 in pickle.load(open(tmp_path / "test_label.pkl", "rb"))[0]
    q = [(SL, "user", 0), ("enroll", "item", 0), ("item_concept", "concept", 0), ("item_concept", "item", 3)]
    assert env.concept_bonus(q) == 0.0  # item3 is a test/validation positive, not train


def test_bonus_not_applied_when_horizon_shorter_than_terminal(tmp_path):
    """Eval-style horizon 3: terminal is step 3; the bonus lands there, not earlier."""
    env, _ = mk_env(tmp_path, 5.0, path_len=3)
    r, b = walk(env, 0, [("enroll", 0), ("item_concept", 0), ("item_concept", 1)])
    assert r == [0.0, 0.0, 6.0] and b == [0.0, 0.0, 1.0]


def test_batch_mixed_terminal_bonus(tmp_path):
    env, _ = mk_env(tmp_path, 5.0)
    env.reset([0, 1])
    for rel0, rel1 in [(("enroll", 0), ("enroll", 1)), (("item_concept", 0), ("item_teacher", 0)),
                       (("item_concept", 1), ("item_teacher", 2)), ((SL, 1), (SL, 2))]:
        idx = [act(env, 0, *rel0), act(env, 1, *rel1)]
        _, r, done = env.batch_step(idx)
    assert done and list(r) == [6.0, 1.0] and list(env._batch_curr_bonus) == [1.0, 0.0]


def test_embeddings_pruning_search_untouched():
    """Diff vs the baseline commit: only kg_env (reward additions) and train_agent (config pass-through) changed."""
    # only files that already existed at the baseline commit and were MODIFIED count (new spike/M5 files are fine)
    out = subprocess.run(["git", "diff", "--name-only", "--diff-filter=M", "5334bf9", "--", "src/UPGPR/"], cwd=ROOT,
                         capture_output=True, text=True).stdout
    changed = {l.strip() for l in out.splitlines() if l.strip()}
    allowed = {"src/UPGPR/kg_env.py", "src/UPGPR/train_agent.py"}
    assert changed <= allowed, changed
    d = subprocess.run(["git", "diff", "5334bf9", "--", "src/UPGPR/kg_env.py"], cwd=ROOT,
                       capture_output=True, text=True).stdout
    removed = [l for l in d.splitlines() if l.startswith("-") and not l.startswith("---")]
    # only the __init__ signature line and the step reward call may be rewritten
    assert len(removed) == 2, removed
    for fn in ("_get_actions", "_get_state", "binary_train_reward", "batch_action_embeddings"):
        assert fn not in "".join(removed)
    d2 = subprocess.run(["git", "diff", "5334bf9", "--", "src/UPGPR/transe_model.py", "src/UPGPR/train_transe_model.py",
                         "src/UPGPR/knowledge_graph.py", "src/UPGPR/test_agent.py", "src/UPGPR/actor_critic.py",
                         "src/UPGPR/spike_b_posthoc.py"], cwd=ROOT, capture_output=True, text=True).stdout
    assert d2 == ""


def _tiny_cfg(tmp_path, epochs):
    kg_args = build_dir(tmp_path)
    train = {"tmp_dir": str(tmp_path), "name": "x", "seed": 23, "gpu": "0", "epochs": epochs, "batch_size": 32,
             "lr": 1e-3, "max_acts": 250, "max_path_len": 4, "gamma": 0.99, "ent_weight": 1e-3, "act_dropout": 0.5,
             "state_history": 2, "hidden": [16, 8], "use_pattern": False, "reward": "binary_train",
             "modified_policy": False, "sum_prob": False, "use_wandb": False}
    return {"KG_ARGS": dict(kg_args), "TRAIN_AGENT": train}


def test_driver_lambda0_matches_original_train_loop(tmp_path):
    """Driver with lambda=0 reproduces the original train_agent.train() weights exactly (CPU, same seed/init)."""
    import json
    import train_agent
    cfg = _tiny_cfg(tmp_path, 2)
    json.dump(cfg, open(tmp_path / "cfg.json", "w"))
    init = str(tmp_path / "init.pt")
    se.make_init(str(tmp_path / "cfg.json"), init)
    # original loop
    train_agent.logger = logging.getLogger("t")
    a = edict(dict(cfg["TRAIN_AGENT"]))
    a.device = "cpu"; a.log_dir = str(tmp_path / "orig"); os.makedirs(a.log_dir)
    from utils import set_random_seed
    set_random_seed(23)
    train_agent.train(a, edict(cfg["KG_ARGS"]))
    ref = torch.load(os.path.join(a.log_dir, "tmp_policy_model_epoch_2.ckpt"))
    se.train_branch(str(tmp_path / "cfg.json"), "control", 0.0, init, str(tmp_path / "ctl"), epochs=2)
    mine = torch.load(str(tmp_path / "ctl" / "ckpt" / "epoch_2.pt"))
    assert se.sd_hash(ref) == se.sd_hash(mine)
    # lambda=5 starts from the identical init, logs show raw and scaled bonus and the same gate
    se.train_branch(str(tmp_path / "cfg.json"), "shaped", 5.0, init, str(tmp_path / "shp"), epochs=2)
    lg = json.load(open(tmp_path / "shp" / "train_log.json"))
    assert json.load(open(tmp_path / "shp" / "train_meta.json"))["init_sha256"] == \
        json.load(open(tmp_path / "ctl" / "train_meta.json"))["init_sha256"]
    for e in lg:
        assert e["early_bonus_violations"] == 0 and e["nonfinite_loss_count"] == 0
        assert abs(e["mean_bonus_times_lambda_per_episode"] - 5 * e["mean_bonus_raw_per_episode"]) < 1e-12
        assert abs(e["mean_total_return_per_episode"] - e["mean_base_return_per_episode"]
                   - e["mean_bonus_times_lambda_per_episode"]) < 1e-9


def test_witness_search_finds_padded_3eff_concept_path(tmp_path):
    import json
    import spike_e_witness as sw
    cfg = _tiny_cfg(tmp_path, 1)
    json.dump(cfg, open(tmp_path / "cfg.json", "w"))
    w = sw.witness(str(tmp_path / "cfg.json"))
    # u0: item0 -> concept0 -> item1 (train), item3 (test, no bonus) ; u1: item0/1 variants.
    assert w["counts"]["users_with_B1_3eff_padded"] >= 1
    p = w["first_witness_3eff_padded_to_4_actions"]["path"]
    assert p[0][0] == SL and p[-1][0] == SL and len(p) == 5
    assert w["first_witness_3eff_padded_to_4_actions"]["step_rewards_lambda5"][-1] == 6.0
    au = sw.audit(str(tmp_path))
    assert "kg.pkl" in au["files"] and "effective_config.json" in au["missing"]
