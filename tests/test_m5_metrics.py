import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "UPGPR"))
import m5_metrics as m  # noqa: E402

SL = "self_loop"


def P(uid, *hops):
    return [(SL, "user", uid)] + list(hops)


def test_metrics_empty_list_scores_zero_not_dropped():
    r = m.user_metrics([], {1, 2})
    assert r["precision"] == r["recall"] == r["ndcg"] == r["hr"] == r["mrr"] == 0.0


def test_precision_denominator_is_10_for_short_list():
    r = m.user_metrics([5], {5})
    assert r["precision"] == 0.1 and r["recall"] == 1.0 and r["hr"] == 1.0 and r["mrr"] == 1.0
    assert r["ndcg"] == 1.0  # idcg uses min(10, npos)=1


def test_ndcg_hand_computed_and_idcg_cap():
    r = m.user_metrics([9, 1, 8, 2], {1, 2})
    dcg = 1 / math.log2(3) + 1 / math.log2(5)
    idcg = 1 / math.log2(2) + 1 / math.log2(3)
    assert r["ndcg"] == pytest.approx(dcg / idcg)
    assert r["mrr"] == 0.5
    many = set(range(100))
    r = m.user_metrics(list(range(10)), many)
    assert r["ndcg"] == pytest.approx(1.0) and r["recall"] == 0.1


def test_duplicate_items_rejected():
    with pytest.raises(AssertionError):
        m.user_metrics([1, 1], {1})


def test_rank_tie_break_by_item_id():
    a = (1.0, 0.5, P(0, ("enroll", "item", 7)), 0)
    b = (1.0, 0.5, P(0, ("enroll", "item", 3)), 1)
    assert [c[2][-1][2] for c in sorted([a, b], key=m.rank_key, reverse=True)] == [3, 7]


def test_family_mapping_and_padding():
    co = P(0, ("enroll", "item", 1), ("enroll", "user", 2), ("enroll", "item", 3))
    cn = P(0, ("enroll", "item", 1), ("item_concept", "concept", 4), ("item_concept", "item", 3))
    padded = cn + [(SL, "item", 3)]
    assert m.family_of_path(co) == "co_learner"
    assert m.family_of_path(cn) == "concept" == m.family_of_path(padded)
    assert m.has_concept(padded) and not m.has_concept(co)


def _toy():
    # user 0: item 10 has co-learner (higher prob) and concept path; item 11 only co-learner
    co10 = P(0, ("enroll", "item", 1), ("enroll", "user", 2), ("enroll", "item", 10))
    cn10 = P(0, ("enroll", "item", 1), ("item_concept", "concept", 4), ("item_concept", "item", 10))
    co11 = P(0, ("enroll", "item", 1), ("enroll", "user", 2), ("enroll", "item", 11))
    # saved probs are legacy shifted (+1) values; product is the legacy score
    results = {"paths": [co10, cn10, co11], "probs": [[1.9, 1.9], [1.2, 1.2], [1.5, 1.5]]}
    import numpy as np
    scores = {0: {10: 2.0, 11: 1.0}}
    return results, scores, {0: {10}}, {0: [1]}


def test_s0_s3_same_items_and_s3_picks_concept():
    results, scores, labels, train = _toy()
    pred, frozen = m.build(results, scores, labels, train)
    r0 = m.per_user_rows(pred, frozen, labels, "S0")[0]
    r3 = m.per_user_rows(pred, frozen, labels, "S3")[0]
    assert r0["ids"] == r3["ids"] == [10, 11]
    assert r0["n_concept_expl"] == 0 and r3["n_concept_expl"] == 1
    assert r3["concept_at10"] == 0.1 and r0["ndcg"] == r3["ndcg"] == 1.0
    av = m.availability(pred, labels, train)
    assert av["pooled"] == 0.5 and av["pairs_with_concept"] == 1 and av["candidate_recall_pooled"] == 1.0


def test_user_without_candidates_stays_eligible_with_zero():
    results, scores, _, train = _toy()
    labels = {0: {10}, 5: {99}}
    pred, frozen = m.build(results, scores, labels, train)
    rows = m.per_user_rows(pred, frozen, labels, "S0")
    assert set(rows) == {0, 5} and rows[5]["ndcg"] == 0.0 and rows[5]["n_items"] == 0
    agg = m.aggregate(rows, labels)
    assert agg["users_zero_returned"] == 1 and agg["n_users"] == 2


def test_train_items_excluded():
    results, scores, labels, _ = _toy()
    pred, frozen = m.build(results, scores, labels, {0: [10]})
    assert [c[2][-1][2] for c in frozen[0]] == [11]
