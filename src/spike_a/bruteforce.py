"""Independent brute force: walks the real env._get_actions step by step, self-loops included.

Used by tests (tiny graphs) and by the CLI to cross-check sampled real pairs. Exponential,
so it takes a call cap and reports `aborted` instead of a count when the cap is hit."""
from collections import Counter


class _Cap(Exception):
    pass


def brute_force(env, kg_args, uid, targets, horizon, max_calls=None):
    sl = kg_args.self_loop
    targets = set(targets)
    seqs = Counter()
    dedup = set()
    loops_hist = Counter()
    state = {"repeated": 0, "calls": 0}

    def rec(path):
        if len(path) == horizon + 1:
            _, ty, i = path[-1]
            if ty == "item" and i in targets:
                seqs[i] += 1
                nodes = tuple((r, t, n) for r, t, n in path[1:] if r != sl)
                dedup.add((i, nodes))
                loops_hist[sum(1 for r, _, _ in path[1:] if r == sl)] += 1
                ids = [("user", uid)] + [(t, n) for _, t, n in nodes]
                if len(set(ids)) != len(ids):
                    state["repeated"] += 1
            return
        state["calls"] += 1
        if max_calls is not None and state["calls"] > max_calls:
            raise _Cap()
        _, ctype, _ = path[-1]
        for rel, nid in env._get_actions(path, False):
            ntype = ctype if rel == sl else kg_args.kg_relation[ctype][rel]
            rec(path + [(rel, ntype, nid)])

    aborted = False
    try:
        rec([(sl, "user", uid)])
    except _Cap:
        aborted = True
    paths = Counter(t for (t, _) in dedup)
    return {"seqs": seqs, "paths": paths, "loops": loops_hist, "repeated": state["repeated"],
            "calls": state["calls"], "aborted": aborted, "dedup_keys": dedup}
