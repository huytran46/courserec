import os
import pickle

import pytest

import tiny
from spike_a import artifacts


def clean_inputs():
    kg, args = tiny.build_kg()
    return kg, args, {u: list(c) for u, c in tiny.TRAIN.items()}, {u: list(c) for u, c in tiny.TEST.items()}


def codes(report, level="errors"):
    return {e["code"] for e in report[level]}


def test_clean_artifacts_pass_audit():
    kg, args, train, test = clean_inputs()
    rep = artifacts.audit(kg.G, args, train, test, None, tiny.build_embeds())
    assert rep["errors"] == [], rep["errors"]
    assert rep["info"]["node_counts"] == {"user": 3, "item": 5, "school": 1, "teacher": 2, "concept": 2}
    assert rep["info"]["kg_enroll_edges"] == 6


def test_kg_containing_test_enrolment_is_leakage():
    _, args, train, test = clean_inputs()
    leaky, _ = tiny.build_kg({**tiny.TRAIN, 0: [0, 1, 2]})
    rep = artifacts.audit(leaky.G, args, train, test, None, tiny.build_embeds())
    assert "kg_contains_test_enrollment" in codes(rep)
    assert "kg_enroll_differs_from_train" in codes(rep)


def test_train_test_overlap_detected():
    kg, args, train, test = clean_inputs()
    rep = artifacts.audit(kg.G, args, train, {0: [1, 2]}, None, tiny.build_embeds())
    assert "split_overlap_train_test" in codes(rep)


def test_validation_leak_into_kg_detected():
    kg, args, train, test = clean_inputs()
    rep = artifacts.audit(kg.G, args, train, test, {0: [0]}, tiny.build_embeds())
    assert "split_overlap_train_validation" in codes(rep)


def test_embedding_shape_mismatch_detected():
    kg, args, train, test = clean_inputs()
    emb = tiny.build_embeds()
    emb["item"] = emb["item"][:4]
    rep = artifacts.audit(kg.G, args, train, test, None, emb)
    assert "embedding_rows_mismatch" in codes(rep)


def test_asymmetric_edge_detected():
    kg, args, train, test = clean_inputs()
    kg.G["item"][2]["enroll"] = tuple(x for x in kg.G["item"][2]["enroll"] if x != 1)
    rep = artifacts.audit(kg.G, args, train, test, None, tiny.build_embeds())
    assert "kg_asymmetric_edge" in codes(rep)


def test_vocab_files_mismatch_detected(tmp_path):
    kg, args, train, test = clean_inputs()
    for fname, n in (("users.txt", 3), ("courses.txt", 5), ("schools.txt", 1), ("teachers.txt", 2), ("concepts.txt", 9)):
        (tmp_path / fname).write_text("\n".join(f"x{i}" for i in range(n)))
    rep = artifacts.audit(kg.G, args, train, test, None, tiny.build_embeds(), data_dir=str(tmp_path))
    assert "vocab_size_mismatch" in codes(rep)


def test_restricted_loader_roundtrips_real_kg_pickle(tmp_path):
    kg, args = tiny.build_kg()
    p = tmp_path / "kg.pkl"
    with open(p, "wb") as f:
        pickle.dump(kg, f)
    stub = artifacts.load_pickle(str(p))
    assert stub.G == kg.G
    assert stub("item", 2) == kg("item", 2)
    assert stub.kg_args == args


def test_restricted_loader_roundtrips_embeds(tmp_path):
    p = tmp_path / "e.pkl"
    emb = tiny.build_embeds()
    with open(p, "wb") as f:
        pickle.dump(emb, f)
    out = artifacts.load_pickle(str(p))
    assert (out["item"] == emb["item"]).all()
    assert (out["enroll"][0] == emb["enroll"][0]).all()


def test_restricted_loader_rejects_arbitrary_callable(tmp_path):
    class Evil:
        def __reduce__(self):
            return (os.system, ("echo pwned",))

    p = tmp_path / "evil.pkl"
    with open(p, "wb") as f:
        pickle.dump(Evil(), f)
    with pytest.raises(artifacts.UntrustedPickle):
        artifacts.load_pickle(str(p))
