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
