import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "UPGPR"))
import m5_funnel as fn  # noqa: E402


def R(beam, s0, s3):
    return {"beam_c": beam, "s0_c": s0, "s3_c": s3, "n_beam": 1}


C = lambda fc, pc, st="reachable", fp=5, pp=3: (st, fp, fc, pp, pc)  # noqa: E731


def test_buckets_distinguish_uncovered_from_no_path():
    assert fn.bucket(R(0, 0, 0), None) == "uncovered"
    assert fn.bucket(R(0, 0, 0), C(0, 0, st="unreachable", fp=0, pp=0)) == "no_path_in_graph"
    assert fn.bucket(R(0, 0, 0), C(0, 0)) == "no_concept_path_in_graph"
    assert fn.bucket(R(0, 0, 0), C(4, 0)) == "pruning_loss"
    assert fn.bucket(R(0, 0, 0), C(4, 2)) == "beam_miss"
    assert fn.bucket(R(1, 0, 1), C(4, 2)) == "selector_miss_s0"
    assert fn.bucket(R(1, 1, 1), C(4, 2)) == "selected_s0"


def _arm(beam, s0, covered=10, full=10, pruned=9):
    b = {k: 0 for k in fn.BUCKETS}
    b.update({"pruning_loss": 1, "beam_miss": covered - 1 - beam, "selector_miss_s0": beam - s0, "selected_s0": s0})
    return {"all": {"pairs": 10, "covered": covered, "full_c": full, "pruned_c": pruned, "beam_c": beam, "s0_c": s0,
                    "s3_c": beam, "beam_c_all": beam, "s0_c_all": s0, "s3_c_all": beam, "buckets": b}}


def test_summarize_decomposition_identities():
    F = {"slots": 10, "seeds": {}}
    for sd, (cb, cs, sb_, ss) in {"1": (2, 1, 6, 5), "2": (3, 1, 7, 4)}.items():
        arm_c, arm_s = _arm(cb, cs), _arm(sb_, ss)
        for a in (arm_c, arm_s):
            a["common"] = a["only"] = a["all"]
        F["seeds"][sd] = {"control": arm_c, "shaped": arm_s}
    d = fn.summarize(F)["decomposition_slots"]
    for i in range(2):
        assert abs(d["total"]["per_seed"][i] - (d["availability_reward"]["per_seed"][i] + d["selector_control"]["per_seed"][i])) < 1e-12
        assert abs(d["reward_under_s0"]["per_seed"][i] -
                   (d["availability_reward"]["per_seed"][i] - d["interaction"]["per_seed"][i])) < 1e-12
    assert all(abs(x - 0.4) < 1e-12 for x in d["availability_reward"]["per_seed"])


def test_funnel_counts_and_violations():
    pairs = {(0, 1): R(1, 1, 1), (0, 2): R(1, 0, 1), (0, 3): R(0, 0, 0), (0, 4): R(1, 1, 1)}
    cache = {(0, 1): C(3, 2), (0, 2): C(3, 2), (0, 3): C(3, 1)}  # (0,4) uncovered
    f = fn.funnel(pairs, cache)
    assert f["pairs"] == 4 and f["covered"] == 3 and f["buckets"]["uncovered"] == 1
    assert f["beam_c"] == 2 and f["s0_c"] == 1 and f["s3_c"] == 2 and f["beam_c_all"] == 3
    assert f["buckets"]["selected_s0"] == 1 and f["buckets"]["selector_miss_s0"] == 1 and f["buckets"]["beam_miss"] == 1
    assert all(v == 0 for v in f["violations"].values())
    bad = fn.funnel({(0, 1): R(1, 0, 1)}, {(0, 1): C(3, 0)})
    assert bad["violations"]["beam_without_pruned"] == 1
    bad2 = fn.funnel({(0, 1): R(0, 1, 1)}, {})
    assert bad2["violations"]["s0_not_in_beam"] == 1 and bad2["violations"]["s3_ne_beam"] == 1
