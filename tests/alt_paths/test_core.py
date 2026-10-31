from collections import Counter
from math import comb

import pytest

import tiny
from spike_a import core


@pytest.fixture(scope="module")
def world():
    kg, args = tiny.build_kg()
    return core.Graph.from_kg(kg, args), kg, args


def fam_counts(res, target):
    out = Counter()
    for (pat, k), n in res.counts[target].items():
        out[core.family(pat)] += n
    return out


def test_full_h3_known_counts(world):
    g, kg, args = world
    res = core.enumerate_user(g, 0, {2}, horizon=3, actions=core.FullActions(g))
    assert not res.censored
    assert fam_counts(res, 2) == {"user": 2, "teacher": 1, "school": 2, "concept": 2}
    assert sum(res.counts[2].values()) == 7
    # only effective length 3 exists (graph is bipartite items vs rest)
    assert {k for (_, k) in res.counts[2]} == {3}
    assert res.repeated_node_paths == 0


def test_sequences_count_self_loops_multiplicity(world):
    g, _, _ = world
    res = core.enumerate_user(g, 0, {2}, horizon=4, actions=core.FullActions(g))
    paths = sum(res.counts[2].values())
    assert paths == 7
    assert core.action_sequences(res.counts[2], 4) == paths * comb(4, 3) == 28


def test_unreachable_target_has_no_paths(world):
    g, _, _ = world
    res = core.enumerate_user(g, 0, {3}, horizon=3, actions=core.FullActions(g))  # c3 isolated
    assert sum(res.counts[3].values()) == 0


def test_censoring_gives_lower_bound(world):
    g, _, _ = world
    res = core.enumerate_user(g, 0, {2}, horizon=3, actions=core.FullActions(g), max_expansions=2)
    assert res.censored
    assert sum(res.counts[2].values()) <= 7


def test_pattern_records_relations_types_direction(world):
    g, _, _ = world
    res = core.enumerate_user(g, 0, {2}, horizon=3, actions=core.FullActions(g))
    pats = {p for (p, _) in res.counts[2]}
    assert (("enroll", "+", "item"), ("item_teacher", "+", "teacher"), ("item_teacher", "-", "item")) in pats
    assert (("enroll", "+", "item"), ("enroll", "-", "user"), ("enroll", "+", "item")) in pats
