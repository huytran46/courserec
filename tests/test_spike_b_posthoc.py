import ast, json, math, os, pickle, sys, types
from collections import Counter
from functools import reduce
from math import log

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src", "UPGPR"))
import spike_b_posthoc as sb  # noqa: E402

SL = "self_loop"


def P(uid, rel, typ, eid, last_rel, pid):
    return [(SL, "user", uid), ("enroll", "item", 100), (rel, typ, eid), (last_rel, "item", pid)]


def cand(score, prob, path, idx):
    return (score, prob, path, idx)


# ---------------------------------------------------------------- selectors
def test_multi_path_one_item_picks_preferred_max_prob():
    c = [cand(1.0, 3.0, P(0, "item_concept", "concept", 1, "item_concept", 7), 0),
         cand(1.0, 2.5, P(0, "item_teacher", "teacher", 2, "item_teacher", 7), 1),
         cand(1.0, 2.9, P(0, "item_teacher", "teacher", 3, "item_teacher", 7), 2)]
    s0 = sb.select_path("S0", c, max(c, key=lambda x: x[1]))
    assert s0 is c[0]
    assert sb.select_path("S1", c, s0) is c[2]       # best-prob among teacher paths
    assert sb.select_path("S2", c, s0) is s0          # no school path -> S0
    assert sb.select_path("S3", c, s0) is s0          # S0 already concept


def test_no_target_falls_back_to_s0():
    c = [cand(1, 2.0, P(0, "enroll", "user", 5, "enroll", 7), 0)]
    for s in sb.SELECTORS:
        assert sb.select_path(s, c, c[0]) is c[0]


def test_tie_break_deterministic_earliest_beam_idx():
    a = cand(1, 2.0, P(0, "item_teacher", "teacher", 1, "item_teacher", 7), 4)
    b = cand(1, 2.0, P(0, "item_teacher", "teacher", 2, "item_teacher", 7), 9)
    s0 = cand(1, 3.0, P(0, "enroll", "user", 5, "enroll", 7), 0)
    assert sb.select_path("S1", [s0, b, a], s0) is a
    assert sb.select_path("S1", [s0, a, b], s0) is a


def test_shortest_with_self_loop():
    long_p = P(0, "item_teacher", "teacher", 1, "item_teacher", 7)
    loop_p = [(SL, "user", 0), ("enroll", "item", 100), (SL, "item", 100), ("item_teacher", "item", 7)]
    c = [cand(1, 3.0, long_p, 0), cand(1, 2.0, loop_p, 1)]
    assert sb.effective_len(long_p) == 3 and sb.effective_len(loop_p) == 2
    assert sb.has_self_loop(loop_p)
    assert sb.select_path("SL", c, c[0]) is c[1]
    assert sb.select_path("SL", c, c[1]) is c[1]


# ---------------------------------------------------------------- probs
def test_shifted_probs_audit_and_no_double_shift():
    probs = [np.float32(1.5), np.float32(1.25)]
    assert sb.audit_probs({"probs": [probs]})["shifted_plus1"] is True
    assert sb.audit_probs({"probs": [[np.float32(0.5)]]})["shifted_plus1"] is False
    path = P(0, "item_teacher", "teacher", 1, "item_teacher", 7)
    pp = sb.collect_pred_paths({"paths": [path], "probs": [probs]}, np.ones((1, 10)), {0: {7}})
    assert pp[0][7][0][1] == reduce(lambda x, y: x * y, probs)  # raw product, no +1


# ---------------------------------------------------------------- freezing / ties / few predictions
def test_score_tie_between_items_order_frozen_not_resorted():
    scores = np.zeros((1, 20)); scores[0, 7] = 1.0; scores[0, 8] = 1.0  # tie
    # item 7 best path prob 2.0, item 8 best path prob 3.0 -> legacy order 8,7
    t1 = P(0, "item_concept", "concept", 1, "item_concept", 7)
    t2 = P(0, "item_teacher", "teacher", 2, "item_teacher", 7)
    t3 = P(0, "item_concept", "concept", 3, "item_concept", 8)
    res = {"paths": [t1, t2, t3], "probs": [[1.0, 2.0], [1.0, 1.9], [1.0, 3.0]]}
    pp = sb.collect_pred_paths(res, scores, {0: {7}})
    frozen = sb.freeze_ranking(sb.best_paths_legacy(pp, {0: [99]}))
    ids = [c[2][-1][2] for c in frozen[0]]
    assert ids == [8, 7]
    recs, _ = sb.run_selectors(frozen, pp)
    for sel in sb.SELECTORS:
        assert [r["cid"] for r in sorted((r for r in recs if r["selector"] == sel), key=lambda r: r["rank"])] == ids
    # S1 swaps item 7 to the teacher path with LOWER prob; re-sorting would not be neutral, order must hold
    s1 = [r for r in recs if r["selector"] == "S1" and r["cid"] == 7][0]
    assert s1["legacy_score"] < s1["s0_legacy_score"]


def test_user_with_few_predictions_kept_in_all_users_report():
    labels = {0: {1}, 1: {2}}
    m = sb.rec_metrics({0: [1, 5, 6]}, labels)  # asc: user0 has 3 preds, user1 none
    assert m["n_users"] == 2 and m["n_valid"] == 0 and m["n_invalid"] == 2
    assert m["all_users"]["hit"] == 50.0
    assert math.isnan(m["valid_only"]["hit"])


def test_train_items_removed_and_no_train_users_skipped():
    path = P(0, "item_teacher", "teacher", 1, "item_teacher", 7)
    res = {"paths": [path], "probs": [[1.0, 1.5]]}
    pp = sb.collect_pred_paths(res, np.ones((1, 10)), {0: {7}})
    assert sb.best_paths_legacy(pp, {0: [7]})[0] == []
    assert 0 not in sb.best_paths_legacy(pp, {})


# ---------------------------------------------------------------- end-to-end vs ORIGINAL evaluator source
def _load_original(tmp_path):
    src = open(os.path.join(ROOT, "src", "UPGPR", "test_agent.py")).read()
    tree = ast.parse(src)
    keep = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("evaluate", "evaluate_paths")]
    mod = ast.Module(body=keep, type_ignores=[])
    ns = {"np": np, "log": log, "os": os, "json": json, "pickle": pickle, "reduce": reduce,
          "Counter": Counter, "wandb": None,
          "load_embed": lambda d: pickle.load(open(f"{d}/transe_embed.pkl", "rb")),
          "evaluate_validation": None,
          "args": types.SimpleNamespace(tmp_dir=str(tmp_path))}
    exec(compile(mod, "test_agent_extract", "exec"), ns)
    return ns


def _synthetic(tmp_path, n_users=40, n_items=60, seed=0):
    rng = np.random.default_rng(seed)
    d = 8
    emb = {"user": rng.normal(size=(n_users, d)), "item": rng.normal(size=(n_items, d)),
           "enroll": (rng.normal(size=d),)}
    emb["item"][11] = emb["item"][10]  # force score ties
    pickle.dump(emb, open(tmp_path / "transe_embed.pkl", "wb"))
    rels = [("item_teacher", "teacher"), ("item_school", "school"), ("item_concept", "concept"), ("enroll", "user")]
    train, test, paths, probs = {}, {}, [], []
    for u in range(n_users):
        if u != 3:  # user 3 has no train labels
            train[u] = set(rng.choice(n_items, 5, replace=False).tolist())
        test[u] = set(rng.choice(n_items, 4, replace=False).tolist())
        n_targets = 3 if u == 5 else int(rng.integers(8, 25))  # user 5 has few predictions
        for t in rng.choice(n_items, n_targets, replace=False):
            for _ in range(int(rng.integers(1, 5))):
                r, ty = rels[int(rng.integers(0, 4))]
                if rng.random() < 0.15:
                    p = [(SL, "user", u), ("enroll", "item", 1), (SL, "item", 1),
                         (r if r != "enroll" else "item_teacher", "item", int(t))]
                else:
                    p = [(SL, "user", u), ("enroll", "item", 1), (r, ty, int(rng.integers(0, 9))), (r, "item", int(t))]
                paths.append(p)
                probs.append((1.0 + rng.random(3)).astype(np.float32).tolist())
    paths.append([(SL, "user", 0), ("enroll", "item", 1), ("item_teacher", "teacher", 2)])  # non-item endpoint
    probs.append([1.5, 1.5])
    pickle.dump(train, open(tmp_path / "train_label.pkl", "wb"))
    pickle.dump(test, open(tmp_path / "test_label.pkl", "wb"))
    pickle.dump({"paths": paths, "probs": probs}, open(tmp_path / "policy_paths_epoch_2.pkl", "wb"))
    return train, test


def test_end_to_end_matches_original_evaluator(tmp_path):
    train, test = _synthetic(tmp_path)
    orig = _load_original(tmp_path)
    kg = types.SimpleNamespace(interaction="enroll")
    orig["evaluate_paths"](str(tmp_path), str(tmp_path / "policy_paths_epoch_2.pkl"), train, test, kg,
                           False, "res", "run_0.json", sum_prob=False)
    kg_args = {"kg_relation": {"user": {"enroll": "item"},
                               "item": {"enroll": "user", "item_concept": "concept", "item_teacher": "teacher", "item_school": "school"},
                               "school": {"item_school": "item"}, "concept": {"item_concept": "item"},
                               "teacher": {"item_teacher": "item"}},
               "interaction": "enroll", "self_loop": SL, "entity_files": {}}
    cfg = {"KG_ARGS": kg_args, "TRAIN_AGENT": {"epochs": 50, "seed": 1},
           "TEST_AGENT": {"tmp_dir": str(tmp_path), "data_dir": str(tmp_path)}}
    json.dump(cfg, open(tmp_path / "cfg.json", "w"))
    out = tmp_path / "out"
    rc = sb.run(str(tmp_path / "cfg.json"), str(tmp_path / "policy_paths_epoch_2.pkl"), str(out),
                logs_json=str(tmp_path / "res" / "run_0.json"))
    assert rc == 0
    rec = json.load(open(out / "baseline_reconstruction.json"))
    assert rec["vs_logs"]
    for k, v in rec["vs_logs"].items():
        assert v["abs_diff"] <= 1e-12, (k, v)
    summ = json.load(open(out / "summary.json"))
    assert summ["smoke"] is True and "SMOKE" in summ["claim_scope"]
    for s in ("S1", "S2", "S3", "SL"):
        assert summ["explanations"][s]["oracle"]["at_ceiling"]
    beam = pickle.load(open(tmp_path / "policy_paths_epoch_2.pkl", "rb"))["paths"]
    bset = {json.dumps(p, default=int) for p in beam}
    rows = [json.loads(l) for l in open(out / "selected_paths.jsonl")]
    assert all(json.dumps(r["path"], default=int) in bset for r in rows)
    for s in sb.SELECTORS:
        assert [(r["uid"], r["cid"], r["rank"]) for r in rows if r["selector"] == s] == \
               [(r["uid"], r["cid"], r["rank"]) for r in rows if r["selector"] == "S0"]
    assert rec["n_users_lt10"] >= 1 and rec["metrics_reconstructed"]["n_invalid"] >= 2  # users 3 and 5
    for f in ("manifest.json", "per_user.csv", "report.md", "summary.json", "selected_paths.jsonl"):
        assert os.path.isfile(out / f)


def test_blocked_without_beam(tmp_path, capsys):
    cfg = {"KG_ARGS": {}, "TRAIN_AGENT": {"epochs": 50}, "TEST_AGENT": {"tmp_dir": str(tmp_path), "data_dir": str(tmp_path)}}
    json.dump(cfg, open(tmp_path / "c.json", "w"))
    assert sb.run(str(tmp_path / "c.json"), str(tmp_path / "nope.pkl"), str(tmp_path / "o")) == 3
    assert "BLOCKED" in capsys.readouterr().out
