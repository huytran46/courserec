import csv
import hashlib
import json
import os
import pickle
import subprocess

import pytest

import tiny
from spike_a import matched_cli

SL = "self_loop"
P_C2 = [(SL, "user", 0), ("enroll", "item", 0), ("item_concept", "concept", 1), ("item_concept", "item", 2)]
P_C1 = [(SL, "user", 1), ("enroll", "item", 0), ("enroll", "user", 0), ("enroll", "item", 1)]


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def head():
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=tiny.ROOT, capture_output=True, text=True).stdout.strip()


def jl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def make_world(tmp, s0_extra=(), beam_paths=None, with_beam=True):
    art = tmp / "art"
    art.mkdir()
    bdir = tmp / "bdir"
    bdir.mkdir()
    kg, args = tiny.build_kg()
    pickle.dump(kg, open(art / "kg.pkl", "wb"))
    pickle.dump({u: list(c) for u, c in tiny.TRAIN.items()}, open(art / "train_label.pkl", "wb"))
    pickle.dump({u: list(c) for u, c in tiny.TEST.items()}, open(art / "test_label.pkl", "wb"))
    pickle.dump(tiny.build_embeds(), open(art / "transe_embed.pkl", "wb"))
    cfg = json.load(open(os.path.join(tiny.ROOT, "config", "UPGPR", "mooc.json")))
    for k in ("PREPROCESS", "TRAIN_EMBEDS", "TRAIN_AGENT", "TEST_AGENT"):
        cfg[k]["tmp_dir"] = str(art)
    cfg_path = tmp / "effective_config.json"
    json.dump(cfg, open(cfg_path, "w"))
    beam = tmp / "beam.pkl"
    if with_beam:
        paths = beam_paths if beam_paths is not None else [P_C2, P_C1]
        pickle.dump({"paths": paths, "probs": [[1.5, 1.5, 1.5]] * len(paths)}, open(beam, "wb"))
    S0 = [{"uid": 0, "cid": 2, "rank": 1, "selector": "S0", "path": [list(x) for x in P_C2]},
          {"uid": 1, "cid": 1, "rank": 1, "selector": "S0", "path": [list(x) for x in P_C1]}, *s0_extra]
    S3 = [{"uid": 0, "cid": 2, "rank": 1, "selector": "S3", "path": [list(x) for x in P_C2],
           "new_property": True, "changed_path": False},
          {"uid": 1, "cid": 1, "rank": 1, "selector": "S3", "path": [list(x) for x in P_C1],
           "new_property": False, "changed_path": False}]
    jl(bdir / "selected_paths.jsonl", S0 + S3)
    json.dump({"n_users_lt10": 2, "n_users_with_frozen_list": 2, "n_beam_paths": 2}, open(bdir / "baseline_reconstruction.json", "w"))
    hashes = {"kg": sha(art / "kg.pkl"), "train_label": sha(art / "train_label.pkl"),
              "test_label": sha(art / "test_label.pkl"), "embeddings": sha(art / "transe_embed.pkl"),
              "config": sha(cfg_path), "beam": sha(beam) if with_beam else None}
    json.dump({"hashes": hashes}, open(bdir / "manifest.json", "w"))
    return art, bdir, cfg_path, beam, hashes


def run(tmp, out, art, bdir, cfg, beam, *extra):
    return matched_cli.main([
        "--config", str(cfg), "--base-config", os.path.join(tiny.ROOT, "config", "UPGPR", "mooc.json"),
        "--artifacts-dir", str(art), "--beam-paths", str(beam), "--b-dir", str(bdir), "--out-dir", str(out),
        "--env-source-commit", head(), "--tool-commit", "testcommit",
        "--expect-positive-pairs", "1", "--expect-topk-pairs", "2", "--expect-topk-users", "2",
        "--bruteforce-users", "2", *extra])


def rows(out, name):
    return list(csv.DictReader(open(os.path.join(out, name))))


def test_tool_identity_ignores_hidden_files(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n")
    (tmp_path / "._a.py").write_text("macOS resource fork junk")
    (tmp_path / ".hidden.py").write_text("y = 2\n")
    ident = matched_cli.tool_identity(str(tmp_path))
    assert list(ident["files_sha256"]) == ["a.py"]
    (tmp_path / "._b.py").write_text("more junk")
    assert matched_cli.tool_identity(str(tmp_path))["bundle_sha256"] == ident["bundle_sha256"]


def test_matched_end_to_end_tiny(tmp_path):
    art, bdir, cfg, beam, _ = make_world(tmp_path)
    out = tmp_path / "out"
    assert run(tmp_path, out, art, bdir, cfg, beam) == 0
    m = json.load(open(out / "manifest.json"))
    assert m["status"] == "COMPLETE"
    assert m["tool_identity"]["git_commit"] == "testcommit" and m["environment"]["pipeline_source_commit"] == head()
    assert m["bruteforce"]["mismatches"] == 0 and m["bruteforce"]["pairs_checked"] >= 2
    s = json.load(open(out / "summary.json"))
    f = s["topk"]["funnel"]["concept"]
    # both pairs have a concept path in the graph (u1 via c2-k0-c1), only (0,2) has it in the beam and in S0
    assert (f["n"], f["full"], f["pruned"], f["beam"], f["selected"]) == (2, 2, 2, 1, 1)
    assert (f["pruning_loss"], f["beam_miss"], f["selector_miss"]) == (0, 1, 0)
    assert s["topk"]["b_reference_check"]["all_match"] is True
    assert s["beam_consistency"]["beam_paths_not_accepted_by_pruned"] == 0
    assert s["schema_equivalence"]["equivalent"] is True
    top = rows(out, "topk_pairs.csv")
    assert [(r["uid"], r["cid"], r["rank"]) for r in top] == [("0", "2", "1"), ("1", "1", "1")]
    assert top[0]["is_test_positive"] == "True" and top[1]["is_test_positive"] == "False"
    pos = rows(out, "positive_pairs.csv")
    assert [(r["uid"], r["cid"]) for r in pos] == [("0", "2")] and pos[0]["full_paths"] == "7"
    w = json.load(open(out / "witnesses.json"))
    assert w["all_edges_and_actions_verified"] is True and w["witnesses"]
    prov = json.load(open(out / "provenance_comparison.json"))
    assert {r["artifact"]: r["status"] for r in prov["table"]}["kg"] in ("match", "single_source")


def test_beam_miss_and_selector_miss_are_located(tmp_path):
    # beam has only the enroll path for (0,2): concept is available in pruned but missing from beam
    only_enroll = [(SL, "user", 0), ("enroll", "item", 0), ("enroll", "user", 1), ("enroll", "item", 2)]
    art, bdir, cfg, beam, _ = make_world(tmp_path, beam_paths=[only_enroll, P_C1])
    rows_ = [json.loads(x) for x in open(bdir / "selected_paths.jsonl")]
    for r in rows_:
        if r["selector"] == "S0" and r["uid"] == 0:
            r["path"] = [list(x) for x in only_enroll]
        if r["selector"] == "S3" and r["uid"] == 0:
            r["new_property"] = False
    jl(bdir / "selected_paths.jsonl", rows_)
    out = tmp_path / "out"
    assert run(tmp_path, out, art, bdir, cfg, beam) == 0
    s = json.load(open(out / "summary.json"))
    f = s["topk"]["funnel"]["concept"]
    # (1,1) also has a concept path in the graph (via c2-k0-c1) that its beam never contains
    assert (f["beam_miss"], f["selector_miss"], f["pruning_loss"], f["beam"]) == (2, 0, 0, 0)
    top = rows(out, "topk_pairs.csv")
    assert top[0]["concept_beam_miss"] == "True" and top[0]["user_beam_miss"] == "False"


def test_blocked_when_beam_missing(tmp_path):
    art, bdir, cfg, beam, _ = make_world(tmp_path, with_beam=False)
    out = tmp_path / "out"
    assert run(tmp_path, out, art, bdir, cfg, beam) == 2
    assert json.load(open(out / "manifest.json"))["status"] == "BLOCKED"


def test_reference_hash_mismatch_stops_before_numbers(tmp_path):
    art, bdir, cfg, beam, _ = make_world(tmp_path)
    out = tmp_path / "out"
    assert run(tmp_path, out, art, bdir, cfg, beam, "--ref-hash", "embeddings=" + "0" * 64) == 3
    m = json.load(open(out / "manifest.json"))
    assert m["status"] == "AUDIT_FAILED" and not os.path.exists(out / "topk_pairs.csv")
    d = json.load(open(out / "diagnostics.json"))
    assert any(e["code"] == "provenance_mismatch_B" for e in d["errors"])


def test_train_item_in_topk_cohort_is_fatal(tmp_path):
    bad = {"uid": 0, "cid": 0, "rank": 2, "selector": "S0", "path": [list(x) for x in P_C2]}
    art, bdir, cfg, beam, _ = make_world(tmp_path, s0_extra=[bad])
    out = tmp_path / "out"
    assert run(tmp_path, out, art, bdir, cfg, beam) == 3
    d = json.load(open(out / "diagnostics.json"))
    assert any(e["code"] == "topk_cohort_invalid" for e in d["errors"])


def test_resume_reuses_checkpoint_and_matches_fresh_run(tmp_path):
    art, bdir, cfg, beam, _ = make_world(tmp_path)
    fresh = tmp_path / "fresh"
    assert run(tmp_path, fresh, art, bdir, cfg, beam) == 0
    out = tmp_path / "resumed"
    assert run(tmp_path, out, art, bdir, cfg, beam, "--max-users", "1") == 0
    assert json.load(open(out / "manifest.json"))["status"] == "PARTIAL"
    os.remove(out / "manifest.json")  # simulate a crash before the final write
    assert run(tmp_path, out, art, bdir, cfg, beam, "--resume") == 0
    assert open(out / "topk_pairs.csv").read() == open(fresh / "topk_pairs.csv").read()
    assert json.load(open(out / "manifest.json"))["status"] == "COMPLETE"
