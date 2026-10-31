import csv
import json
import os
import pickle

import pytest

import tiny
from spike_a import cli


def make_world(tmp_path, train=tiny.TRAIN, with_embeds=True, with_kg=True, kg_train=None, max_acts=2,
               test=tiny.TEST):
    art = tmp_path / "art"
    art.mkdir()
    data = tmp_path / "data"
    data.mkdir()
    kg, args = tiny.build_kg(kg_train if kg_train is not None else train)
    if with_kg:
        pickle.dump(kg, open(art / "kg.pkl", "wb"))
    pickle.dump({u: list(c) for u, c in train.items()}, open(art / "train_label.pkl", "wb"))
    pickle.dump({u: list(c) for u, c in test.items()}, open(art / "test_label.pkl", "wb"))
    if with_embeds:
        pickle.dump(tiny.build_embeds(), open(art / "transe_embed.pkl", "wb"))
    cfg = json.load(open(os.path.join(tiny.ROOT, "config", "UPGPR", "mooc.json")))
    for k in ("TEST_AGENT", "TRAIN_AGENT"):
        cfg[k]["tmp_dir"] = str(art)
        cfg[k]["data_dir"] = str(data)
        cfg[k]["max_acts"] = max_acts
    p = tmp_path / "cfg.json"
    json.dump(cfg, open(p, "w"))
    return str(p), str(tmp_path / "out")


def run(cfg, out, *extra):
    return cli.main(["--config", cfg, "--out-dir", out, "--allow-unverified", *extra])


def rows(out, name="pairs.csv"):
    return list(csv.DictReader(open(os.path.join(out, name))))


def test_happy_path_writes_all_outputs(tmp_path):
    cfg, out = make_world(tmp_path)
    assert run(cfg, out, "--bruteforce-pairs", "1") == 0
    for f in ("manifest.json", "pairs.csv", "summary.json", "report.md", "witnesses.json"):
        assert os.path.exists(os.path.join(out, f)), f
    r = rows(out)
    assert len(r) == 1 and (r[0]["uid"], r[0]["cid"]) == ("0", "2")
    assert r[0]["full_paths"] == "7" and r[0]["full_n_families"] == "4" and r[0]["full_n_patterns"] == "4"
    assert r[0]["full_seqs"] == "7" and r[0]["full_censored"] == "False"
    assert int(r[0]["pruned_paths"]) <= 7
    s = json.load(open(os.path.join(out, "summary.json")))
    h3 = s["horizons"]["3"]
    assert h3["denominators"] == {"eligible_pairs": 1, "reachable_pairs_full": 1, "users_with_eligible_pair": 1}
    assert h3["full"]["share_reachable_with_ge2_families"] == 1.0
    m = json.load(open(os.path.join(out, "manifest.json")))
    assert m["status"] == "COMPLETE_UNVERIFIED_PROVENANCE"
    assert m["artifacts"]["kg.pkl"]["sha256"] and m["split_sizes"]["test_pairs"] == 1
    assert m["bruteforce"]["checked_pairs"] == 1 and m["bruteforce"]["mismatches"] == 0
    w = json.load(open(os.path.join(out, "witnesses.json")))
    assert w["all_edges_verified"] is True and len(w["witnesses"]) >= 4


def test_missing_embeddings_completes_full_only(tmp_path):
    cfg, out = make_world(tmp_path, with_embeds=False)
    assert run(cfg, out, "--bruteforce-pairs", "1") == 0
    r = rows(out)[0]
    assert r["full_paths"] == "7" and r["pruned_paths"] == ""
    m = json.load(open(os.path.join(out, "manifest.json")))
    assert m["bruteforce"]["checked_pairs"] == 1 and m["bruteforce"]["mismatches"] == 0
    assert {d["mode"] for d in m["bruteforce"]["details"]} == {"full"}
    assert m["pruned"]["status"].startswith("BLOCKED_missing")
    assert "transe_embed.pkl" in m["pruned"]["missing"][0]


def test_missing_kg_is_blocked(tmp_path):
    cfg, out = make_world(tmp_path, with_kg=False)
    assert run(cfg, out) == 2
    m = json.load(open(os.path.join(out, "manifest.json")))
    assert m["status"] == "BLOCKED" and any("kg.pkl" in x for x in m["missing"])
    assert not os.path.exists(os.path.join(out, "pairs.csv"))


def test_leakage_stops_numerical_output(tmp_path):
    cfg, out = make_world(tmp_path, kg_train={**tiny.TRAIN, 0: [0, 1, 2]})
    assert run(cfg, out) == 3
    m = json.load(open(os.path.join(out, "manifest.json")))
    assert m["status"] == "AUDIT_FAILED"
    assert any(e["code"] == "kg_contains_test_enrollment" for e in m["audit"]["errors"])
    assert not os.path.exists(os.path.join(out, "pairs.csv"))
    assert os.path.exists(os.path.join(out, "diagnostics.json"))


def test_missing_source_commit_fails_unless_allowed(tmp_path):
    cfg, out = make_world(tmp_path)
    assert cli.main(["--config", cfg, "--out-dir", out]) == 3
    m = json.load(open(os.path.join(out, "manifest.json")))
    assert any(e["code"] == "source_commit_missing" for e in m["audit"]["errors"])


def test_refuses_to_overwrite_previous_run(tmp_path):
    cfg, out = make_world(tmp_path)
    assert run(cfg, out) == 0
    before = open(os.path.join(out, "pairs.csv")).read()
    assert run(cfg, out) == 4
    assert open(os.path.join(out, "pairs.csv")).read() == before


def test_h4_requires_explicit_flag(tmp_path):
    cfg, out = make_world(tmp_path)
    with pytest.raises(SystemExit):
        run(cfg, out, "--horizons", "3", "4")
    assert run(cfg, out, "--horizons", "3", "4", "--enable-h4") == 0
    assert os.path.exists(os.path.join(out, "pairs_h4.csv"))
    r4 = rows(out, "pairs_h4.csv")[0]
    assert r4["full_paths"] == "7" and r4["full_seqs"] == "28"


def test_sampling_is_recorded_and_not_exact_cohort(tmp_path):
    cfg, out = make_world(tmp_path, test={0: [2], 1: [1]})
    assert run(cfg, out, "--sample-users", "1", "--seed", "5") == 0
    m = json.load(open(os.path.join(out, "manifest.json")))
    assert m["cohort"]["sampled"] is True and m["cohort"]["seed"] == 5
    assert m["cohort"]["users"] == 1 and m["cohort"]["exact_full_cohort"] is False
    assert len(rows(out)) == 1
