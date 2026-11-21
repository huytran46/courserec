import random
from types import SimpleNamespace as NS

import numpy as np
import pytest

import tiny
from spike_a.bruteforce import brute_force
from spike_a import core

BIG = 10 ** 9


def run_engine(g, actions, uid, targets, horizon):
    res = core.enumerate_user(g, uid, targets, horizon, actions)
    paths = {t: sum(c.values()) for t, c in res.counts.items()}
    seqs = {t: core.action_sequences(c, horizon) for t, c in res.counts.items()}
    return res, paths, seqs


@pytest.mark.parametrize("horizon", [2, 3, 4, 5])
def test_full_matches_env_bruteforce_tiny(horizon):
    kg, args = tiny.build_kg()
    g = core.Graph.from_kg(kg, args)
    env = core.make_env(kg, tiny.build_embeds(), args, BIG)
    bf = brute_force(env, args, 0, {2, 3}, horizon)
    res, paths, seqs = run_engine(g, core.FullActions(g), 0, {2, 3}, horizon)
    assert paths[2] == bf["paths"][2]
    assert seqs[2] == bf["seqs"][2]
    assert paths[3] == bf["paths"].get(3, 0)
    assert bf["repeated"] == 0 and res.repeated_node_paths == 0
    # self-loop count histogram: loops = horizon - effective length
    for k in {k for (_, k) in res.counts[2]}:
        assert any(h == horizon - k for h in bf["loops"])


def random_world(seed, n_users=6, n_items=8):
    rng = random.Random(seed)
    from knowledge_graph import KnowledgeGraph

    args = tiny.kg_args()
    train = {u: sorted(rng.sample(range(n_items), rng.randint(2, 4))) for u in range(n_users)}
    n_t, n_s, n_k = 3, 2, 4
    ds = NS(
        user=NS(vocab_size=n_users), item=NS(vocab_size=n_items), school=NS(vocab_size=n_s),
        teacher=NS(vocab_size=n_t), concept=NS(vocab_size=n_k),
        interactions=NS(data=[(u, c) for u, cs in train.items() for c in cs]),
        item_teacher=NS(data=[rng.sample(range(n_t), rng.randint(0, 2)) for _ in range(n_items)]),
        item_school=NS(data=[[rng.randrange(n_s)] for _ in range(n_items)]),
        item_concept=NS(data=[rng.sample(range(n_k), rng.randint(0, 3)) for _ in range(n_items)]),
    )
    kg = KnowledgeGraph(ds, args)
    sizes = {"user": n_users, "item": n_items, "school": n_s, "teacher": n_t, "concept": n_k}
    nprng = np.random.RandomState(seed)
    emb = {t: nprng.randn(k, 6) for t, k in sizes.items()}
    for r in ("enroll", "item_teacher", "item_school", "item_concept"):
        emb[r] = (nprng.randn(6), 0.0)
    return kg, args, emb, train, n_items


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("horizon", [3, 4])
def test_fuzz_full_and_pruned_match_env(seed, horizon):
    kg, args, emb, train, n_items = random_world(seed)
    g = core.Graph.from_kg(kg, args)
    targets = set(range(n_items)) - set(train[0])
    # full
    env_full = core.make_env(kg, emb, args, BIG)
    bf = brute_force(env_full, args, 0, targets, horizon)
    res, paths, seqs = run_engine(g, core.FullActions(g), 0, targets, horizon)
    for t in targets:
        assert paths[t] == bf["paths"].get(t, 0), (seed, horizon, t)
        assert seqs[t] == bf["seqs"].get(t, 0)
    assert res.repeated_node_paths == 0 and bf["repeated"] == 0
    # pruned with a tiny max_acts so trimming really happens
    for max_acts in (1, 2, 3):
        env = core.make_env(kg, emb, args, max_acts)
        bfp = brute_force(env, args, 0, targets, horizon)
        pa = core.PrunedActions(g, env, max_acts)
        resp, pathsp, seqsp = run_engine(g, pa, 0, targets, horizon)
        for t in targets:
            assert pathsp[t] == bfp["paths"].get(t, 0), (seed, horizon, max_acts, t)
            assert seqsp[t] == bfp["seqs"].get(t, 0)
            assert pathsp[t] <= paths[t]  # pruning only removes


def test_pruning_delegates_to_env_get_actions():
    kg, args, emb, train, n_items = random_world(1)
    g = core.Graph.from_kg(kg, args)
    env = core.make_env(kg, emb, args, 1)
    calls = []
    orig = env._get_actions
    env._get_actions = lambda path, done: (calls.append(list(path)), orig(path, done))[1]
    pa = core.PrunedActions(g, env, 1)
    core.enumerate_user(g, 0, set(range(n_items)) - set(train[0]), 3, pa)
    assert calls and pa.env_calls == len(calls)
    assert all(c[0] == (args.self_loop, "user", 0) for c in calls)


def test_visited_filter_blocks_revisits_but_self_loop_allowed():
    kg, args = tiny.build_kg()
    env = core.make_env(kg, tiny.build_embeds(), args, BIG)
    acts = env._get_actions([(args.self_loop, "user", 0), ("enroll", "item", 0)], False)
    assert (args.self_loop, 1) == acts[0] or acts[0][0] == args.self_loop
    assert ("enroll", 0) not in acts  # cannot go back to user 0
    g = core.Graph.from_kg(kg, args)
    ch = core.FullActions(g).children([(args.self_loop, "user", 0), ("enroll", "item", 0)],
                                      {("user", 0), ("item", 0)})
    assert sorted((r, i) for r, _, i in ch) == sorted(acts[1:])
