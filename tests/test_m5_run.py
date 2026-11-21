import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "UPGPR"))
import m5_run as r  # noqa: E402


def V(ndcg, c):
    return {"S3": {"ndcg": ndcg, "concept_at10": c}}


def test_selects_highest_concept_among_eligible():
    cand, ch = r.select_lambda({0.0: V(0.20, 0.05), 1.0: V(0.20, 0.10), 5.0: V(0.19, 0.20)})
    assert ch == 5.0 and cand[5.0]["eligible"]


def test_ndcg_floor_excludes():
    cand, ch = r.select_lambda({0.0: V(0.20, 0.05), 1.0: V(0.20, 0.08), 5.0: V(0.18, 0.40)})  # 0.18 < 0.19
    assert ch == 1.0 and not cand[5.0]["utility_ok"]


def test_gain_threshold_exact_two_pp_ok_and_below_not():
    cand, ch = r.select_lambda({0.0: V(0.2, 0.05), 1.0: V(0.2, 0.0701), 5.0: V(0.2, 0.0699)})
    assert cand[1.0]["gain_ok"] and not cand[5.0]["gain_ok"] and ch == 1.0


def test_tie_prefers_smaller_lambda():
    _, ch = r.select_lambda({0.0: V(0.2, 0.05), 1.0: V(0.2, 0.20), 5.0: V(0.2, 0.2005)})
    assert ch == 1.0


def test_no_feasible_lambda():
    _, ch = r.select_lambda({0.0: V(0.2, 0.05), 1.0: V(0.2, 0.06), 5.0: V(0.1, 0.5)})
    assert ch is None


def test_arm_dir_reuses_seed23_dev_runs():
    assert r.arm_dir("/m5", 23, 5.0) == "/m5/development/seed23/lambda5"
    assert r.arm_dir("/m5", 24, 0.0) == "/m5/final/seed24/lambda0"
