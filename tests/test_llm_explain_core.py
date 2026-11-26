import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from llm_explain import core

NAMES = {("item", 1): "自然灾害", ("item", 2): "C++语言", ("concept", 5): "before and after analysis",
         ("item", 3): "微积分", ("item", 4): "操作系统", ("teacher", 7): "王伟"}
name_of = lambda t, i: NAMES.get((t, i), f"{'learner' if t == 'user' else t} {i}")

CONCEPT = [("self_loop", "user", 10), ("enroll", "item", 1), ("item_concept", "concept", 5), ("item_concept", "item", 2)]
COL = [("self_loop", "user", 10), ("enroll", "item", 3), ("enroll", "user", 20), ("enroll", "item", 4)]
F = core.facts(CONCEPT, name_of)
G = core.facts(COL, name_of)


def claim(h, r, t, q=""):
    return {"head": h, "relation": r, "tail": t, "quote": q}


def lab(c, f=F, text=None):
    return core.label_claim(c, f, c["quote"] if text is None else text)


def test_facts_codes_and_normalised_direction():
    assert [e["code"] for e in G["entities"]] == ["L1", "C1", "L2", "C2"]
    assert G["triples"] == [("L1", "enrolled_in", "C1"), ("L2", "enrolled_in", "C1"), ("L2", "enrolled_in", "C2")]
    assert (G["target"], G["final"]) == ("L1", "C2")
    assert F["triples"] == [("L1", "enrolled_in", "C1"), ("C1", "teaches_concept", "K1"), ("C2", "teaches_concept", "K1")]


def test_supported_requires_relation_and_direction():
    assert lab(claim("[L1]", "enrolled_in", "[C1] 自然灾害")) == "supported"
    assert lab(claim("C2", "teaches_concept", "K1")) == "supported"
    assert lab(claim("[C1]", "enrolled_in", "[L1]")) == "entities_in_path_relation_mismatch"
    assert lab(claim("[L1]", "taught_by", "[C1]")) == "entities_in_path_relation_mismatch"
    assert lab(claim("[C1]", "teaches_concept", "[C2]")) == "entities_in_path_relation_mismatch"


def test_shares_learner_only_when_shared_entity_is_in_path():
    assert lab(claim("[L2]", "shares_learner_with", "[L1]"), G) == "supported"
    assert lab(claim("[C2]", "shares_learner_with", "[C1]"), G) == "supported"
    assert lab(claim("[C2]", "shares_learner_with", "[C1]"), F) == "entities_in_path_relation_mismatch"


def test_unknown_entity_is_never_fuzzy_matched():
    assert lab(claim("the learner", "enrolled_in", "[C1]")) == "unknown_entity"
    assert lab(claim("[L1]", "enrolled_in", "[C9]")) == "unknown_entity"
    assert lab(claim("自然灾害", "teaches_concept", "[K1]")) == "unknown_entity"


def test_other_and_unknown_relation_are_unsupported_claim():
    assert lab(claim("[L1]", "other", "[C1]")) == "unsupported_claim"
    assert lab(claim("[L1]", "likes", "[C1]")) == "unsupported_claim"


def test_wording_overrides_matching_triple():
    for q in ["[L1] finished [C1].", "[L2] then took [C2].", "[C2] builds on [C1].", "[L1] completed [C1]."]:
        assert lab(claim("[L1]", "enrolled_in", "[C1]", q)) == "unsupported_claim", q


def test_framing_label_and_its_limits():
    q = "[C2] is recommended to [L1] because [L1] enrolled in [C1]."
    assert lab(claim("[C2]", "recommended_to", "[L1]", q)) == "recommendation_framing"
    bad = "[C2] is recommended to [L1] because it builds on [C1]."
    assert lab(claim("[C2]", "recommended_to", "[L1]", bad)) == "unsupported_claim"
    assert lab(claim("[C2]", "recommended_to", "[L2]", ""), G) == "entities_in_path_relation_mismatch"
    assert lab(claim("[C1]", "recommended_to", "[L1]", "")) == "entities_in_path_relation_mismatch"


def test_trimmed_or_invented_quote_cannot_hide_wording():
    text = "[L1] enrolled in [C1] 自然灾害. Then [L1] finished [C1]."
    # invented quote: not in the explanation -> whole explanation is checked
    assert lab(claim("[L1]", "enrolled_in", "[C1]", "[L1] took [C1]"), text=text) == "unsupported_claim"
    # trimmed quote: the containing sentence is checked
    assert lab(claim("[L1]", "enrolled_in", "[C1]", "[L1] finished"), text=text) == "unsupported_claim"
    assert lab(claim("[L1]", "enrolled_in", "[C1]", "finished [C1]"), text=text) == "unsupported_claim"
    assert lab(claim("[L1]", "enrolled_in", "[C1]", "enrolled in [C1]"), text=text) == "supported"
    one = "[L2] enrolled in [C1] and then took [C2]."
    assert lab(claim("[L2]", "enrolled_in", "[C2]", "took [C2]"), G, one) == "unsupported_claim"


def test_trigger_words_and_cjk_inside_entity_names_are_ignored():
    q = "[C1] 自然灾害 teaches [K1] before and after analysis."
    assert lab(claim("[C1]", "teaches_concept", "[K1]", q)) == "supported"
    assert core.cjk_leak(q, F) is False
    assert core.cjk_leak("[L1] 报名了 [C1] 自然灾害.", F) is True
    assert core.has_flag_wording(q, F) is False
    assert core.has_flag_wording("[L1] has finished [C1] 自然灾害.", F) is True


def test_code_name_mismatches():
    assert core.code_name_mismatches("[L1] learner 10 enrolled in [C1] 自然灾害.", F) == 0
    assert core.code_name_mismatches("[C1] C++语言 and [C9] x", F) == 2


def test_parse_claims():
    ok = 'Sure:\n[{"head":"[L1]","relation":"enrolled_in","tail":"[C1]","quote":"q"}]'
    assert core.parse_claims(ok)[0]["relation"] == "enrolled_in"
    for bad in ["no json", '[{"head": 1', '["a","b"]', '[{"head":"[L1]"}]', "[]"]:
        assert core.parse_claims(bad) is None, bad
