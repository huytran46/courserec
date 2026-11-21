"""Helpers for the matched-to-Spike-B analysis: S0 top-K cohort, beam index, path acceptance,
availability funnel and provenance comparison. Pure functions, no I/O beyond the S0 jsonl."""
import json
from collections import Counter, defaultdict

from spike_a import core

FAMILIES = ("concept", "teacher", "school", "user")  # 'user' = co-learner via enrol


# ------------------------------------------------------------------ cohort
def load_s0_cohort(path):
    """Selector S0 rows only, de-duplicated by (uid, cid) keeping the lowest rank, file order preserved."""
    info = {"rows_total": 0, "s0_rows": 0, "duplicates_dropped": 0, "other_selector_rows_ignored": 0}
    seen = {}
    order = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            info["rows_total"] += 1
            if r["selector"] != "S0":
                info["other_selector_rows_ignored"] += 1
                continue
            info["s0_rows"] += 1
            k = (r["uid"], r["cid"])
            if k in seen:
                info["duplicates_dropped"] += 1
                if r["rank"] < seen[k]["rank"]:
                    seen[k]["rank"] = r["rank"]
                    seen[k]["path"] = [tuple(x) for x in r["path"]]
                continue
            seen[k] = {"uid": r["uid"], "cid": r["cid"], "rank": r["rank"],
                       "path": [tuple(x) for x in r["path"]]}
            order.append(k)
    return [seen[k] for k in order], info


def check_cohort(cohort, train, n_items):
    problems = []
    per_user = defaultdict(list)
    pairs = set()
    for c in cohort:
        k = (c["uid"], c["cid"])
        if k in pairs:
            problems.append(f"duplicate pair {k}")
        pairs.add(k)
        per_user[c["uid"]].append(c["rank"])
        if c["cid"] in set(train.get(c["uid"], ())):
            problems.append(f"train item in cohort {k}")
        if not 0 <= c["cid"] < n_items:
            problems.append(f"item out of range {k}")
    for u, ranks in per_user.items():
        if len(set(ranks)) != len(ranks):
            problems.append(f"duplicate ranks for user {u}")
    return {"problems": problems, "users": len(per_user), "pairs": len(cohort),
            "users_lt10": sum(1 for r in per_user.values() if len(r) < 10),
            "max_per_user": max((len(r) for r in per_user.values()), default=0)}


def pairs_in_test_positives(cohort, test):
    return sum(1 for c in cohort if c["cid"] in set(test.get(c["uid"], ())))


# ------------------------------------------------------------------ paths
def nonloop(path, self_loop):
    return [tuple(x) for x in path if x[0] != self_loop]


def path_accepted(path, g, actions):
    """Would the environment (full or pruned `actions`) allow this path? Returns (ok, reason).

    Self-loops are the env exception and are skipped; every other hop must be an existing edge to an
    unvisited (type,id) and, for pruned actions, survive the top-max_acts trimming given the prefix."""
    sl = g.self_loop
    start = tuple(path[0])
    if start[0] != sl or start[1] != "user":
        return False, "bad start"
    prefix = [start]
    visited = {("user", start[2])}
    cur = ("user", start[2])
    for x in path[1:]:
        rel, nt, nid = x
        if rel == sl:
            if (nt, nid) != cur:
                return False, "self-loop changes node"
            continue
        ctype, cid = cur
        tail = g.kg_relation[ctype].get(rel)
        if tail is None or tail != nt:
            return False, f"relation {rel} not valid from {ctype}"
        if nid not in g.adj[nt] or not g.has_edge(ctype, cid, rel, nid):
            return False, f"no edge {ctype}({cid}) -{rel}-> {nt}({nid})"
        if (nt, nid) in visited:
            return False, f"revisits {nt}({nid})"
        if not actions.keeps(prefix, visited, rel, nt, nid):
            return False, f"pruned out at {ctype}({cid}) -{rel}-> {nt}({nid})"
        prefix.append((rel, nt, nid))
        visited.add((nt, nid))
        cur = (nt, nid)
    return True, "ok"


def beam_index(beam, wanted_pairs, self_loop):
    """(uid, cid) -> {'seqs': beam action sequences ending at item cid, 'paths': Counter of non-self-loop paths}."""
    out = {}
    for path in beam["paths"]:
        last = path[-1]
        if last[1] != "item":
            continue
        k = (path[0][2], last[2])
        if k not in wanted_pairs:
            continue
        e = out.setdefault(k, {"seqs": 0, "paths": Counter()})
        e["seqs"] += 1
        e["paths"][tuple(nonloop(path[1:], self_loop))] += 1
    return out


def path_family(path_nl, g):
    """Family of one concrete path given as non-self-loop tuples after the start (rel, type, id)."""
    hops = []
    prev = "user"
    for rel, nt, _ in path_nl:
        hops.append(g.hop(prev, rel, nt))
        prev = nt
    return core.family(tuple(hops))


def schema_equivalence(patterns):
    """Is 'family (type of intermediate node)' interchangeable with 'path contains relation X' (Spike B)
    and with 'terminal relation' for every observed pattern? Checked, never assumed."""
    f2r = {}
    violations = []
    for pat in patterns:
        fam = core.family(pat)
        rels = {h[0] for h in pat} - {"enroll"}
        terminal = pat[-1][0]
        if "+" in fam:
            violations.append((core.pattern_str(pat), "mixed family"))
            continue
        if fam == "user":
            ok = not rels and terminal == "enroll"
            rel = "enroll"
        else:
            ok = rels == {terminal}
            rel = terminal
        if not ok:
            violations.append((core.pattern_str(pat), f"relations {sorted(rels)} vs terminal {terminal}"))
            continue
        if f2r.setdefault(fam, rel) != rel:
            violations.append((core.pattern_str(pat), f"family {fam} maps to two relations"))
    if len(set(f2r.values())) != len(f2r):
        violations.append(("*", "two families share one relation"))
    return {"equivalent": not violations, "family_to_relation": f2r, "violations": violations}


# ------------------------------------------------------------------ funnel
def funnel(recs, target):
    """Availability funnel for one target family over ONE denominator (len(recs)).
    recs: dicts with sets 'full', 'pruned', 'beam', 's0' of family names."""
    n = len(recs)
    c = Counter()
    bad = Counter()
    for r in recs:
        f, p, b, s = (target in r[k] for k in ("full", "pruned", "beam", "s0"))
        c["full"] += f
        c["pruned"] += p
        c["beam"] += b
        c["selected"] += s
        c["pruning_loss"] += f and not p
        c["beam_miss"] += p and not b
        c["selector_miss"] += b and not s
        bad["pruned_not_in_full"] += p and not f
        bad["beam_not_in_pruned"] += b and not p
        bad["selected_not_in_beam"] += s and not b
    keys = ["full", "pruned", "beam", "selected", "pruning_loss", "beam_miss", "selector_miss"]
    out = {"n": n, **{k: c[k] for k in keys}}
    out["proportions"] = {k: (c[k] / n if n else None) for k in keys}
    out["inconsistent"] = {k: bad[k] for k in ("pruned_not_in_full", "beam_not_in_pruned", "selected_not_in_beam")}
    return out


# ------------------------------------------------------------------ provenance
def provenance_table(spec):
    rows = []
    for name, srcs in spec.items():
        vals = {k: v for k, v in srcs.items() if v is not None}
        if not vals:
            status = "missing"
        elif len(set(vals.values())) > 1:
            status = "mismatch"
        elif len(vals) == 1:
            status = "single_source"
        else:
            status = "match"
        rows.append({"artifact": name, **srcs, "status": status})
    return rows
