import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import pytest
from llm_explain import report


def row(i, group, labels, failed=False):
    claims = None if failed else [{"head": "[L1]", "relation": "enrolled_in", "tail": "[C1]", "quote": "q", "label": l}
                                  for l in labels]
    return {"example_id": i, "group": group, "path_facts": "f", "explanation": "e", "claims": claims,
            "cjk_leak": False, "flag_wording": False, "code_name_mismatches": 0}


def test_summary_excludes_framing_from_rates_and_counts_failures():
    rows = [row(0, "concept", ["supported", "recommendation_framing", "unsupported_claim"]),
            row(1, "concept", ["supported", "unknown_entity"]), row(2, "concept", [], failed=True)]
    s = report.summarize(rows)["concept"]
    assert (s["examples"], s["extraction_failures"], s["claims_total"], s["claims_scored"]) == (3, 1, 5, 4)
    assert s["claims_per_explanation"] == 2.5
    assert s["rate"]["supported"] == 0.5 and s["rate"]["unsupported_claim"] == 0.25
    assert s["recommendation_framing"] == 1


def test_summary_with_no_scored_claims_gives_none_rates():
    s = report.summarize([row(0, "school", [], failed=True)])["school"]
    assert s["rate"]["supported"] is None and s["claims_per_explanation"] is None


def test_pick_is_stratified_deterministic_and_fills_small_groups():
    rows = [row(i, g, ["supported"]) for i, g in
            enumerate(["co_learner"] * 100 + ["concept"] * 100 + ["teacher"] * 3 + ["school"] * 25)]
    a = report.pick_for_labeling(rows, n=40, seed=0)
    assert [r["example_id"] for r in a] == [r["example_id"] for r in report.pick_for_labeling(rows, 40, 0)]
    count = lambda g: sum(r["group"] == g for r in a)
    assert len(a) == 40 and count("teacher") == 3 and count("school") >= 10
    assert abs(count("co_learner") - count("concept")) <= 1


def test_label_rows_hide_checker_label_and_keep_failed_examples():
    out = report.label_rows([row(0, "concept", ["supported", "unknown_entity"]), row(1, "concept", [], True)])
    assert len(out) == 3 and all("label" not in r and r["human_true"] == "" for r in out)
    assert out[2]["claim_idx"] == ""
    assert list(out[0]) == report.CSV_FIELDS


def test_score_precision_recall_of_flags():
    rows = [row(0, "concept", ["unsupported_claim", "unknown_entity", "supported", "supported",
                               "recommendation_framing"])]
    sheet = report.label_rows(rows)
    for r, v in zip(sheet, ["false", "TRUE", "False", "true", "true"]):
        r["human_true"] = v
    s = report.score_labels(sheet, rows)["all"]
    assert (s["tp"], s["fp"], s["fn"], s["tn"]) == (1, 1, 1, 2)
    assert s["precision"] == 0.5 and s["recall"] == 0.5


def test_score_rejects_blank_typo_or_changed_rows():
    rows = [row(0, "concept", ["supported"])]
    for bad in ["", "yes?"]:
        sheet = report.label_rows(rows)
        sheet[0]["human_true"] = bad
        with pytest.raises(ValueError):
            report.score_labels(sheet, rows)
    sheet = report.label_rows(rows)
    sheet[0].update(human_true="true", head="[C9]")
    with pytest.raises(ValueError):
        report.score_labels(sheet, rows)


def test_explanation_level_rate_counts_flagged_claim_or_wording():
    rows = [row(0, "concept", ["supported", "recommendation_framing"]),
            row(1, "concept", ["supported", "unknown_entity"]),
            {**row(2, "concept", ["recommendation_framing"]), "flag_wording": True},
            row(3, "concept", [], failed=True)]
    s = report.summarize(rows)["concept"]
    assert (s["explanations_judged"], s["explanations_flagged"], s["explanations_flagged_rate"]) == (3, 2, 2 / 3)
    assert s["explanations_with_framing"] == 2


def test_splits_are_disjoint_deterministic_with_fixed_composition():
    ex = [{"example_id": i, "group": g} for i, g in
          enumerate(["co_learner"] * 100 + ["concept"] * 100 + ["teacher"] * 25 + ["school"] * 4)]
    dev, val = report.split_ids(ex)
    assert report.split_ids(list(reversed(ex))) == (dev, val)
    assert not set(dev) & set(val)
    comp = lambda ids: {g: sum(ex[i]["group"] == g for i in ids) for g in report.DEV_QUOTAS}
    assert comp(dev) == {"co_learner": 2, "concept": 2, "teacher": 2, "school": 2}
    assert comp(val) == {"co_learner": 6, "concept": 10, "teacher": 2, "school": 2}
    assert {0, 100, 200, 225} <= set(dev)   # the four already inspected in the first smoke run


def gate_case(human, flagged):
    rows = [row(i, "concept", ["unknown_entity" if f else "supported"]) for i, f in enumerate(flagged)]
    sheet = report.gate_rows(rows)
    for r, h in zip(sheet, human):
        r["human_has_unsupported"] = "true" if h else "false"
    return report.score_gate(sheet, rows)


def test_gate_sheet_hides_checker_output():
    sheet = report.gate_rows([row(0, "concept", ["unknown_entity"])])
    assert list(sheet[0]) == report.GATE_FIELDS and sheet[0]["human_has_unsupported"] == ""


def test_gate_passes_only_above_threshold_and_majority_baseline():
    human = [True] * 12 + [False] * 8
    good = gate_case(human, [True] * 11 + [False] * 9)
    assert good["agreement"] == 0.95 and good["passed"] is True
    skew = [True] * 17 + [False] * 3
    always = gate_case(skew, [True] * 20)
    assert always["agreement"] == 0.85 and always["majority_baseline"] == 0.85 and always["passed"] is False
    assert gate_case(human, [True] * 20)["passed"] is False


def test_gate_extraction_failure_counts_as_disagreement_and_blanks_error():
    rows = [row(0, "concept", [], failed=True), row(1, "concept", ["supported"])]
    sheet = report.gate_rows(rows)
    for r in sheet:
        r["human_has_unsupported"] = "false"
    assert report.score_gate(sheet, rows)["agreement"] == 0.5
    sheet[0]["human_has_unsupported"] = ""
    with pytest.raises(ValueError):
        report.score_gate(sheet, rows)
    with pytest.raises(ValueError):
        report.score_gate(sheet[1:], rows)
