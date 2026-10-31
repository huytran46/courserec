import json
from collections import Counter

import pytest

import tiny
from spike_a import core, matched
from test_bruteforce import random_world

SL = "self_loop"


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def row(uid, cid, rank, sel="S0", mid=("item_concept", "concept", 0), hist=0):
    return {"uid": uid, "cid": cid, "rank": rank, "selector": sel,
            "path": [[SL, "user", uid], ["enroll", "item", hist], [mid[0], mid[1], mid[2]], [mid[0], "item", cid]]}


# ------------------------------------------------------------------ cohort
def test_s0_loader_keeps_only_s0_and_dedups_by_pair(tmp_path):
    p = tmp_path / "sel.jsonl"
    write_jsonl(p, [row(0, 2, 1), row(0, 3, 2), row(0, 2, 3), row(0, 2, 1, sel="S3"), row(1, 1, 1, sel="SL")])
    cohort, info = matched.load_s0_cohort(str(p))
    assert [(c["uid"], c["cid"], c["rank"]) for c in cohort] == [(0, 2, 1), (0, 3, 2)]
    assert info["rows_total"] == 5 and info["s0_rows"] == 3 and info["duplicates_dropped"] == 1
    assert info["other_selector_rows_ignored"] == 2


def test_cohort_checks_train_items_fail_and_short_users_counted():
    cohort = [{"uid": 0, "cid": 2, "rank": 1}, {"uid": 0, "cid": 3, "rank": 2}, {"uid": 1, "cid": 1, "rank": 1}]
    rep = matched.check_cohort(cohort, tiny.TRAIN, n_items=5)
    assert rep["problems"] == [] and rep["users"] == 2 and rep["users_lt10"] == 2
    assert matched.pairs_in_test_positives(cohort, tiny.TEST) == 1  # (0,2) only; (0,3), (1,1) are not positives
    bad = matched.check_cohort(cohort + [{"uid": 0, "cid": 0, "rank": 3}], tiny.TRAIN, n_items=5)
    assert any("train" in p for p in bad["problems"])


# ------------------------------------------------------------------ funnel
def rec(full, pruned, beam, s0):
    return {"full": set(full), "pruned": set(pruned), "beam": set(beam), "s0": {s0} if s0 else set()}


def test_funnel_counts_and_same_denominator():
    recs = [
        rec(["concept", "user"], ["concept", "user"], ["concept"], "user"),   # selector miss
        rec(["concept"], ["concept"], [], None),                                # beam miss
        rec(["concept"], [], [], None),                                         # pruning loss
        rec(["concept"], ["concept"], ["concept"], "concept"),                  # all have it
        rec(["user"], ["user"], ["user"], "user"),                              # no concept anywhere
    ]
    f = matched.funnel(recs, "concept")
    assert f["n"] == 5
    assert (f["full"], f["pruned"], f["beam"], f["selected"]) == (4, 3, 2, 1)
    assert (f["pruning_loss"], f["beam_miss"], f["selector_miss"]) == (1, 1, 1)
    assert f["inconsistent"] == {"pruned_not_in_full": 0, "beam_not_in_pruned": 0, "selected_not_in_beam": 0}
    assert f["proportions"]["pruning_loss"] == pytest.approx(0.2)
    # inconsistency is reported, not hidden
    g = matched.funnel([rec([], ["concept"], ["concept"], "concept")], "concept")
    assert g["inconsistent"]["pruned_not_in_full"] == 1


# ------------------------------------------------------------------ acceptance
def test_path_accepted_full_vs_pruned_and_rejections():
    kg, args, emb, train, n_items = random_world(2)
    g = core.Graph.from_kg(kg, args)
    targets = set(range(n_items)) - set(train[0])
    full = core.FullActions(g)
    env = core.make_env(kg, emb, args, 1)
    pr = core.PrunedActions(g, env, 1)
    wf = core.enumerate_user(g, 0, targets, 3, full, witnesses_per_pattern=1000).witnesses
    wp = core.enumerate_user(g, 0, targets, 3, pr, witnesses_per_pattern=1000).witnesses
    fset = {tuple(p) for ws in wf.values() for p in ws}
    pset = {tuple(p) for ws in wp.values() for p in ws}
    assert pset < fset  # pruning removes something on this graph
    for p in pset:
        assert matched.path_accepted(list(p), g, pr)[0], p
    for p in fset:
        assert matched.path_accepted(list(p), g, full)[0]
    lost = next(iter(fset - pset))
    ok, why = matched.path_accepted(list(lost), g, pr)
    assert not ok and "pruned" in why


def test_path_accepted_rejects_revisit_missing_edge_and_handles_selfloop():
    kg, args = tiny.build_kg()
    g = core.Graph.from_kg(kg, args)
    full = core.FullActions(g)
    good = [(SL, "user", 0), ("enroll", "item", 0), ("item_concept", "concept", 1), ("item_concept", "item", 2)]
    assert matched.path_accepted(good, g, full)[0]
    revisit = [(SL, "user", 0), ("enroll", "item", 0), ("enroll", "user", 0), ("enroll", "item", 1)]
    assert not matched.path_accepted(revisit, g, full)[0]
    noedge = [(SL, "user", 0), ("enroll", "item", 0), ("item_concept", "concept", 0), ("item_concept", "item", 2)]
    assert not matched.path_accepted(noedge, g, full)[0]  # c0 not linked to k0
    looped = [good[0], (SL, "user", 0), good[1], good[2], good[3]]
    assert matched.path_accepted(looped, g, full)[0]  # self-loop is the env exception


# ------------------------------------------------------------------ beam index
def test_beam_index_collapses_selfloops_and_counts_sequences():
    a = [(SL, "user", 0), ("enroll", "item", 0), ("item_concept", "concept", 1), ("item_concept", "item", 2)]
    b = [(SL, "user", 0), (SL, "user", 0), ("enroll", "item", 0), ("item_concept", "concept", 1)]  # not item end
    c = [(SL, "user", 0), ("enroll", "item", 0), ("item_concept", "concept", 1), ("item_concept", "item", 2)]
    d = [(SL, "user", 0), ("enroll", "item", 1), ("item_school", "school", 0), ("item_school", "item", 2)]
    beam = {"paths": [a, b, c, d], "probs": [[1.1]] * 4}
    idx = matched.beam_index(beam, {(0, 2)}, SL)
    assert idx[(0, 2)]["seqs"] == 3
    assert len(idx[(0, 2)]["paths"]) == 2  # a and c are the same non-self-loop path
    assert idx[(0, 2)]["paths"][tuple(a[1:])] == 2
    assert (0, 3) not in idx


# ------------------------------------------------------------------ schema equivalence
def test_schema_equivalence_pattern_vs_family_vs_relation():
    kg, args = tiny.build_kg()
    g = core.Graph.from_kg(kg, args)
    res = core.enumerate_user(g, 0, {2}, 3, core.FullActions(g))
    pats = {p for (p, _) in res.counts[2]}
    eq = matched.schema_equivalence(pats)
    assert eq["equivalent"] is True
    assert eq["family_to_relation"] == {"concept": "item_concept", "school": "item_school",
                                         "teacher": "item_teacher", "user": "enroll"}
    broken = set(pats) | {(("enroll", "+", "item"), ("item_concept", "+", "concept"), ("item_teacher", "-", "item"))}
    assert matched.schema_equivalence(broken)["equivalent"] is False


# ------------------------------------------------------------------ provenance table
def test_provenance_table_statuses():
    rows = matched.provenance_table({
        "kg": {"A_manifest": "x", "B_manifest": "x", "recomputed_B": "x", "recomputed_A": "x"},
        "emb": {"A_manifest": "a", "B_manifest": "b", "recomputed_B": "b", "recomputed_A": "a"},
        "cfg": {"A_manifest": None, "B_manifest": "c", "recomputed_B": "c", "recomputed_A": None},
        "gone": {"A_manifest": None, "B_manifest": None, "recomputed_B": None, "recomputed_A": None},
        "lie": {"A_manifest": "x", "B_manifest": "x", "recomputed_B": "y", "recomputed_A": None},
    })
    st = {r["artifact"]: r["status"] for r in rows}
    assert st == {"kg": "match", "emb": "mismatch", "cfg": "match", "gone": "missing", "lie": "mismatch"}
