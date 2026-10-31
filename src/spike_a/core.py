"""Bounded path enumeration on the UPGPR knowledge graph (Spike A).

Conventions (match src/UPGPR/kg_env.py):
- A path is a list of (relation, node_type, node_id); the first tuple is the start
  (self_loop, 'user', uid) and is NOT a hop.
- The agent takes exactly `horizon` actions (max_path_len). Each action is either the
  self-loop (index 0, always available) or an edge to a node not yet visited, where
  "visited" is the set of (type, id) over all path tuples.
- Effective length k = number of non-self-loop edges. Because a self-loop does not change
  the visited set nor the current node, every non-self-loop path of effective length k
  <= horizon that ends at the target corresponds to exactly C(horizon, k) action
  sequences (choice of which steps are self-loops).  Tests verify this identity against
  an independent brute force that calls BatchKGEnvironment._get_actions step by step.
- Ordinary traversal can never repeat a node; `repeated_node_paths` is an audit counter
  and must stay 0.
"""
from bisect import bisect_left
from collections import Counter
from math import comb


class Graph:
    def __init__(self, adj, kg_relation, canon, self_loop, interaction):
        self.adj = adj  # {type: {id: {rel: sorted tuple of ids}}}
        self.kg_relation = kg_relation
        self.canon = canon  # rel -> (head_type, tail_type)
        self.self_loop = self_loop
        self.interaction = interaction
        self._sets = {}

    @classmethod
    def from_kg(cls, kg, kg_args):
        canon = {kg_args.interaction: ("user", "item")}
        for rel, (_, tail) in kg_args.item_relation.items():
            canon[rel] = ("item", tail)
        for rel, (_, tail) in (kg_args.get("user_relation") or {}).items():
            canon[rel] = ("user", tail)
        for rel, spec in (kg_args.get("entity_relation") or {}).items():
            canon[rel] = (spec[1], spec[2])
        kg_relation = {h: dict(r) for h, r in kg_args.kg_relation.items()}
        return cls(kg.G, kg_relation, canon, kg_args.self_loop, kg_args.interaction)

    def neigh(self, ntype, nid, rel):
        return self.adj[ntype][nid][rel]

    def neigh_set(self, ntype, nid, rel):
        key = (ntype, nid, rel)
        s = self._sets.get(key)
        if s is None:
            s = self._sets[key] = frozenset(self.adj[ntype][nid][rel])
        return s

    def has_edge(self, ntype, nid, rel, other_id):
        t = self.adj[ntype][nid][rel]
        i = bisect_left(t, other_id)
        return i < len(t) and t[i] == other_id

    def hop(self, head_type, rel, tail_type):
        sign = "+" if self.canon[rel] == (head_type, tail_type) else "-"
        return (rel, sign, tail_type)


def family(pattern):
    """Node types of the intermediate nodes, e.g. 'teacher', 'user' (co-learner), 'school+concept'."""
    mids = {h[2] for h in pattern[:-1]} - {"item"}
    return "+".join(sorted(mids)) if mids else "item"


def pattern_label(pattern):
    return ">".join(f"{r}{s}{t[0]}" if False else f"{r}{s}" for r, s, t in pattern)


def pattern_str(pattern, start_type="user"):
    out = start_type
    for r, s, t in pattern:
        out += f" -{r}{s}-> {t}"
    return out


def action_sequences(counts, horizon):
    """counts: Counter[(pattern, k)] of de-duplicated non-self-loop paths -> number of action sequences."""
    return sum(n * comb(horizon, k) for (_, k), n in counts.items())


class UserResult:
    def __init__(self, targets):
        self.counts = {t: Counter() for t in targets}
        self.expansions = 0
        self.censored = False
        self.repeated_node_paths = 0
        self.witnesses = {}  # (target, pattern) -> list of paths (<=3)


class _Budget(Exception):
    pass


class FullActions:
    """No action pruning: every non-visited neighbour is an action."""

    pruned = False

    def __init__(self, graph):
        self.g = graph

    def reset(self):
        pass

    def children(self, path, visited):
        _, ctype, cid = path[-1]
        out = []
        for rel, ids in self.g.adj[ctype][cid].items():
            ntype = self.g.kg_relation[ctype][rel]
            for n in ids:
                if (ntype, n) not in visited:
                    out.append((rel, ntype, n))
        return out

    def hits(self, path, visited, targets):
        _, ctype, cid = path[-1]
        out = []
        for rel in self.g.adj[ctype][cid]:
            if self.g.kg_relation[ctype][rel] != "item":
                continue
            s = self.g.neigh_set(ctype, cid, rel)
            for t in targets:
                if t in s and ("item", t) not in visited:
                    out.append((rel, t))
        return out

    def kept_set(self, path, visited):
        return None  # None == everything non-visited

    def keeps(self, path, visited, rel, ntype, nid):
        return True


class PrunedActions(FullActions):
    """Action set exactly as BatchKGEnvironment._get_actions (top-max_acts trimming).

    When the number of non-visited candidates is <= max_acts the env returns all of them
    (no scoring), so we skip the call; otherwise we delegate to env._get_actions."""

    pruned = True

    def __init__(self, graph, env, max_acts):
        super().__init__(graph)
        self.env = env
        self.max_acts = max_acts
        self.env_calls = 0
        self.fast_calls = 0
        self._cache = {}

    def reset(self):
        self._cache = {}

    def _n_candidates(self, path, visited):
        _, ctype, cid = path[-1]
        n = 0
        adj = self.g.adj[ctype][cid]
        for rel, ids in adj.items():
            n += len(ids)
        for vt, vid in visited:
            for rel in adj:
                if self.g.kg_relation[ctype][rel] == vt and self.g.has_edge(ctype, cid, rel, vid):
                    n -= 1
        return n

    def _env_actions(self, path):
        self.env_calls += 1
        acts = self.env._get_actions(path, False)
        assert acts[0][0] == self.g.self_loop
        return acts[1:]  # [(rel, nid)]

    def kept_list(self, path, visited):
        key = tuple(path)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        _, ctype, cid = path[-1]
        if self._n_candidates(path, visited) <= self.max_acts:
            self.fast_calls += 1
            out = [(r, self.g.kg_relation[ctype][r], n)
                   for r, ids in self.g.adj[ctype][cid].items()
                   for n in ids if (self.g.kg_relation[ctype][r], n) not in visited]
        else:
            out = [(r, self.g.kg_relation[ctype][r], n) for r, n in self._env_actions(path)]
        self._cache[key] = out
        return out

    def children(self, path, visited):
        return self.kept_list(path, visited)

    def hits(self, path, visited, targets):
        return [(r, n) for r, ty, n in self.kept_list(path, visited) if ty == "item" and n in targets]

    def kept_set(self, path, visited):
        return {(r, n) for r, _, n in self.kept_list(path, visited)}

    def keeps(self, path, visited, rel, ntype, nid):
        _, ctype, cid = path[-1]
        if self._n_candidates(path, visited) <= self.max_acts:
            return self.g.has_edge(ctype, cid, rel, nid) and (ntype, nid) not in visited
        return (rel, nid) in self.kept_set(path, visited)


def make_env(kg, embeds, kg_args, max_acts):
    """Real BatchKGEnvironment instance with only the attributes _get_actions reads."""
    import numpy as np
    from kg_env import BatchKGEnvironment

    env = object.__new__(BatchKGEnvironment)
    env.max_acts = max_acts
    env.act_dim = max_acts + 1
    env.kg = kg
    env.kg_args = kg_args
    # embeds=None is only valid for unpruned use (max_acts huge): _get_actions then never scores.
    env.embeds = dict(embeds) if embeds is not None else {"user": np.zeros((1, 1))}
    env.embed_size = env.embeds["user"].shape[1]
    env.embeds[kg_args.self_loop] = (np.zeros(env.embed_size), 0.0)
    return env


def enumerate_user(graph, uid, targets, horizon, actions, max_expansions=None, witnesses_per_pattern=2):
    """Count de-duplicated non-self-loop simple paths user -> target of effective length <= horizon.

    Exact unless res.censored (then counts are lower bounds)."""
    g = graph
    targets = set(targets)
    res = UserResult(targets)
    actions.reset()
    start = (g.self_loop, "user", uid)
    path = [start]
    visited = {("user", uid)}
    hops = []

    def tick(n=1):
        res.expansions += n
        if max_expansions is not None and res.expansions > max_expansions:
            raise _Budget()

    def record(t, rel, ctype, extra_hops):
        pattern = tuple(hops) + tuple(extra_hops)
        nodes = [(ty, i) for _, ty, i in path] + [("item", t)]
        if len(set(nodes)) != len(nodes):
            res.repeated_node_paths += 1
        res.counts[t][(pattern, len(pattern))] += 1
        key = (t, pattern)
        w = res.witnesses.setdefault(key, [])
        if len(w) < witnesses_per_pattern:
            w.append(list(path) + [(rel, "item", t)])

    def final_hits(ctype):
        for rel, t in actions.hits(path, visited, targets):
            record(t, rel, ctype, [g.hop(ctype, rel, "item")])

    def join(a_type, a_id):
        """Count paths a -> m -> t (two more edges) without enumerating every m."""
        kept_a = actions.kept_set(path, visited)
        for t in targets:
            if ("item", t) in visited:
                continue
            for rel2 in g.adj["item"][t]:
                m_type = g.kg_relation["item"][rel2]
                if g.kg_relation[m_type].get(rel2) != "item":
                    continue
                cand = g.neigh_set("item", t, rel2)
                for rel1 in g.adj[a_type][a_id]:
                    if g.kg_relation[a_type][rel1] != m_type:
                        continue
                    s1 = g.neigh_set(a_type, a_id, rel1)
                    small, big = (s1, cand) if len(s1) <= len(cand) else (cand, s1)
                    for m in small:
                        if m not in big:
                            continue
                        tick()
                        if (m_type, m) in visited:
                            continue
                        if kept_a is not None and (rel1, m) not in kept_a:
                            continue
                        path.append((rel1, m_type, m))
                        visited.add((m_type, m))
                        hops.append(g.hop(a_type, rel1, m_type))
                        try:
                            if actions.keeps(path, visited, rel2, "item", t):
                                record(t, rel2, m_type, [g.hop(m_type, rel2, "item")])
                        finally:
                            path.pop()
                            visited.discard((m_type, m))
                            hops.pop()

    def rec(depth):
        _, ctype, cid = path[-1]
        tick()
        final_hits(ctype)  # effective length depth+1
        if depth == horizon - 2:
            join(ctype, cid)  # effective length depth+2 == horizon
            return
        if depth > horizon - 2:
            return
        for rel, ntype, nid in actions.children(path, visited):
            path.append((rel, ntype, nid))
            visited.add((ntype, nid))
            hops.append(g.hop(ctype, rel, ntype))
            try:
                rec(depth + 1)
            finally:
                path.pop()
                visited.discard((ntype, nid))
                hops.pop()

    try:
        rec(0)
    except _Budget:
        res.censored = True
    return res
