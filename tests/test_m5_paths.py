import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "UPGPR"))
import m5_paths as mp  # noqa: E402

SL = "self_loop"


def P(*steps):
    """steps: (relation, type). Builds a path with start user 0 and dummy ids."""
    return [(SL, "user", 0)] + [(r, t, i + 1) for i, (r, t) in enumerate(steps)]


CO = P(("enroll", "item"), ("enroll", "user"), ("enroll", "item"))
CON = P(("enroll", "item"), ("item_concept", "concept"), ("item_concept", "item"))
TEA = P(("enroll", "item"), ("item_teacher", "teacher"), ("item_teacher", "item"))
SCH = P(("enroll", "item"), ("item_school", "school"), ("item_school", "item"))


def test_groups_pure_by_intermediate_entity():
    g = lambda p: mp.group_of(mp.pattern(p, SL))  # noqa: E731
    assert g(CO) == "co_learner" and g(CON) == "concept" and g(TEA) == "teacher" and g(SCH) == "school"


def test_self_loops_ignored_and_same_pattern():
    padded = [(SL, "user", 0), (SL, "user", 0)] + CON[1:]
    assert mp.pattern(padded, SL) == mp.pattern(CON, SL)


def test_hybrid_is_separate_and_other_cases():
    hyb = P(("enroll", "item"), ("item_concept", "concept"), ("item_concept", "item"), ("item_teacher", "teacher"),
            ("item_teacher", "item"))
    assert mp.group_of(mp.pattern(hyb, SL)) == "hybrid"
    assert mp.group_of(mp.pattern(P(("enroll", "item")), SL)) == "other"  # direct, no intermediate entity
    assert mp.group_of(mp.pattern(P(("enroll", "item"), ("item_concept", "concept")), SL)) == "other"  # not ending at item
    assert mp.group_of(()) == "other"


def test_category_absent_from_kg_groups_have_zero():
    assert "category" in mp.GROUPS and mp.ENTITY_GROUP["user"] == "co_learner"


def test_pattern_str_readable():
    assert mp.pattern_str(mp.pattern(CON, SL)) == "user -enroll-> item -item_concept-> concept -item_concept-> item"


def test_build_report_shares_and_consistency():
    def sel(co, con, expl):
        g = {k: 0 for k in mp.GROUPS}
        g["co_learner"], g["concept"] = co, con
        return {"slots_with_explanation": expl, "groups": g, "concept_any": con, "hybrid_with_concept": 0,
                "patterns": {"a": co, "b": con}}
    C = {"seeds": {}}
    for i, s in enumerate(mp.SEEDS):
        one = {"S0": sel(6, 2, 8), "S3": sel(4, 4, 8), "eval_concept_at10": {"S0": 0.2, "S3": 0.4}}
        C["seeds"][str(s)] = {"control": one, "shaped": one}
    R = mp.build_report(C, slots=10)
    g = R["arms"]["control_S0"]["groups"]["co_learner"]
    assert abs(g["share_of_all_slots"]["mean"] - 0.6) < 1e-12 and abs(g["share_of_explained"]["mean"] - 0.75) < 1e-12
    assert R["arms"]["control_S0"]["missing_slots"]["per_seed"] == [2, 2, 2]
    assert R["consistency"]["control_S0"]["max_abs_diff_any_concept_vs_eval"] < 1e-12
    assert R["top_patterns"]["control_S0"]["patterns"][0]["pattern"] == "a"
