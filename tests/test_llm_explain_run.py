import json
import sys
from types import SimpleNamespace
import numpy as np
import pytest
from llm_explain import core, prompts, run


def path(u, c1, mid_rel, mid_t, mid, last_rel, c2):
    return [("self_loop", "user", u), ("enroll", "item", c1), (mid_rel, mid_t, mid), (last_rel, "item", c2)]


def toy():
    # user 0: course 2 has two paths (higher prob = displayed), course 3 one path, course 0 is a train item
    ps = [path(0, 0, "item_concept", "concept", 0, "item_concept", 2),
          path(0, 0, "enroll", "user", 1, "enroll", 2),
          path(0, 0, "enroll", "user", 1, "enroll", 3),
          path(0, 2, "enroll", "user", 1, "enroll", 0)]
    probs = [[1.1, 1.1, 1.1], [1.9, 1.9, 1.9], [1.5, 1.5, 1.5], [2.0, 2.0, 2.0]]
    scores = np.array([[9.0, 0.0, 5.0, 1.0]])
    return {"paths": ps, "probs": probs}, scores, {0: [1]}, {0: [0]}


def test_displayed_takes_only_the_selected_path_of_each_top10_item():
    got = run.displayed(*toy())
    assert [(d["uid"], d["rank"], d["item"], d["group"]) for d in got] == [(0, 1, 2, "co_learner"),
                                                                          (0, 2, 3, "co_learner")]
    assert got[0]["path"][2][1] == "user"


def test_sample_quotas_deterministic():
    d = [{"uid": u, "rank": 1, "item": 1, "group": g, "path": []}
         for u, g in enumerate(["co_learner"] * 300 + ["concept"] * 300 + ["school"] * 4)]
    a = run.sample(d, run.QUOTAS, 0)
    assert a == run.sample(list(reversed(d)), run.QUOTAS, 0)
    assert [sum(x["group"] == g for x in a) for g in ("co_learner", "concept", "teacher", "school")] == [100, 100, 0, 4]
    assert [x["example_id"] for x in a] == list(range(len(a)))


def test_edge_check_catches_misaligned_vocab():
    tables = {"item_concept": [{0}, set(), {0}], "item_teacher": [set()] * 3, "item_school": [set()] * 3,
              "enroll": {(0, 0), (1, 0), (1, 2)}}
    ps = toy()[0]["paths"]
    run.check_edges(ps[:2], tables)
    with pytest.raises(ValueError):
        run.check_edges([path(0, 0, "item_concept", "concept", 7, "item_concept", 2)], tables)
    with pytest.raises(ValueError):
        run.check_edges([ps[2]], tables)


def test_gate_counts():
    run.check_counts(dict(run.EXPECTED_S0_SEED23))
    with pytest.raises(ValueError):
        run.check_counts({**run.EXPECTED_S0_SEED23, "concept": 7602})


def test_writer_prompt_has_codes_and_english_rule():
    f = core.facts(toy()[0]["paths"][0], lambda t, i: f"{t}-{i}")
    p = prompts.writer(f)
    assert "- [C1] item-0 teaches concept [K1] concept-0" in p
    assert "recommended to learner [L1]" in p and "[C2] item-2" in p
    assert "Write in English. Keep entity names exactly as given." in p
    e = prompts.extractor("TEXT")
    assert e.endswith("TEXT") and all(r in e for r in core.RELATIONS)


def test_process_example_end_to_end_with_fake_model():
    ex = {"example_id": 3, "group": "concept", "uid": 0, "rank": 1, "item": 2,
          "facts": core.facts(toy()[0]["paths"][0], lambda t, i: f"{t}-{i}")}
    claims = [{"head": "[L1]", "relation": "enrolled_in", "tail": "[C1]", "quote": "[L1] user-0 enrolled in [C1] item-0"}]
    answers = iter(["[L1] user-0 enrolled in [C1] item-0.", json.dumps(claims)])
    r = run.process(ex, lambda prompt, n: next(answers))
    assert r["claims"][0]["label"] == "supported" and r["extractor_raw"].startswith("[")
    assert r["cjk_leak"] is False and r["code_name_mismatches"] == 0
    bad = iter(["x", "not json"])
    assert run.process(ex, lambda prompt, n: next(bad))["claims"] is None


def test_namer_falls_back_to_id_when_name_is_empty(tmp_path):
    for t, v in [("course", "courses"), ("concept", "concepts"), ("teacher", "teachers"), ("school", "schools")]:
        (tmp_path / f"{v}.txt").write_text("X_1")
        (tmp_path / f"{t}.json").write_text(json.dumps({"id": "X_1", "name": " "}))
    n = run.namer(tmp_path, tmp_path)
    assert n("school", 0) == "X_1" and n("concept", 0) == "X_1"


def test_load_model_pins_and_logs_hub_sha_without_config_commit_hash(monkeypatch):
    """transformers 5.22.1 configs have no _commit_hash."""
    seen = []

    class Loader:
        @staticmethod
        def from_pretrained(name, revision=None, **kw):
            seen.append(revision)
            return SimpleNamespace(config=SimpleNamespace())

    class Api:
        def model_info(self, name, revision):
            return SimpleNamespace(sha="abc123" if revision == "main" else revision)

    fakes = {"torch": SimpleNamespace(float16="f16", __version__="t"),
             "transformers": SimpleNamespace(set_seed=lambda s: None, __version__="5.22.1", AutoTokenizer=Loader,
                                             AutoModelForCausalLM=Loader, BitsAndBytesConfig=lambda **kw: None),
             "huggingface_hub": SimpleNamespace(HfApi=Api)}
    for k, v in fakes.items():
        monkeypatch.setitem(sys.modules, k, v)
    _, info = run.load_model("m", "main")
    assert info["revision"] == "abc123" and seen == ["abc123", "abc123"]


def test_extractor_prompt_has_a_worked_example_with_fake_entities():
    p = prompts.extractor("TEXT")
    ex = p.split("Example", 1)[1]
    claims = core.parse_claims(ex.split("Output:\n", 1)[1].split("\n\n")[0])
    assert {c["relation"] for c in claims} >= {"enrolled_in", "teaches_concept", "recommended_to", "other"}
    assert sum(c["relation"] == "other" for c in claims) >= 2
    assert all(core.BRACKET.search(c["head"]) for c in claims)
    assert not core.CJK.search(ex)
    source = ex.split("Explanation:\n", 1)[1].split("\nOutput:")[0]
    assert all(c["quote"] in source for c in claims)
    framing = next(c for c in claims if c["relation"] == "recommended_to")
    assert "because" not in framing["quote"]


def test_full_run_needs_a_passed_gate_for_the_same_code(tmp_path):
    with pytest.raises(ValueError):
        run.require_gate(tmp_path, "fp")
    p = tmp_path / "gate_result.json"
    for bad in [{"passed": False, "fingerprint": "fp"}, {"passed": True, "fingerprint": "old"}]:
        p.write_text(json.dumps(bad))
        with pytest.raises(ValueError):
            run.require_gate(tmp_path, "fp")
    p.write_text(json.dumps({"passed": True, "fingerprint": "fp"}))
    run.require_gate(tmp_path, "fp")
