"""Trusted-project pickle loading (allowlist, no arbitrary callables) and artifact audits."""
import hashlib
import io
import os
import pickle
from collections import Counter


class UntrustedPickle(Exception):
    pass


class KGStub:
    """Stand-in for knowledge_graph.KnowledgeGraph: restores state, same __call__ as the original."""

    def __call__(self, eh_type, eh_id=None, relation=None):
        data = self.G
        if eh_type is not None:
            data = data[eh_type]
        if eh_id is not None:
            data = data[eh_id]
        if relation is not None:
            data = data[relation]
        return data


_NUMPY_MODS = {"numpy", "numpy.core.multiarray", "numpy._core.multiarray", "numpy.core.numeric",
               "numpy._core.numeric"}
_NUMPY_NAMES = {"ndarray", "dtype", "_reconstruct", "scalar", "_frombuffer"}
_ALLOWED = {
    ("knowledge_graph", "KnowledgeGraph"): KGStub,
    ("easydict", "EasyDict"): None,
    ("copyreg", "_reconstructor"): None,
    ("copyreg", "__newobj__"): None,
    ("builtins", "object"): None,
    ("builtins", "set"): None,
    ("builtins", "frozenset"): None,
    ("builtins", "dict"): None,
    ("builtins", "list"): None,
    ("builtins", "tuple"): None,
    ("collections", "OrderedDict"): None,
    ("collections", "defaultdict"): None,
}


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in _ALLOWED:
            repl = _ALLOWED[(module, name)]
            if repl is not None:
                return repl
            return super().find_class(module, name)
        if module in _NUMPY_MODS and name in _NUMPY_NAMES:
            return super().find_class(module, name)
        raise UntrustedPickle(f"blocked global {module}.{name}")


def load_pickle(path):
    with open(path, "rb") as f:
        return _Unpickler(io.BytesIO(f.read())).load()


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _err(rep, code, detail, level="errors"):
    rep[level].append({"code": code, "detail": detail})


def audit(G, kg_args, train, test, validation, embeds, data_dir=None):
    """Structural + leakage audit. Any entry in report['errors'] means: do not interpret numbers."""
    rep = {"errors": [], "warnings": [], "info": {}}
    rel = kg_args.kg_relation
    if set(G) != set(kg_args.entities):
        _err(rep, "kg_entity_types_mismatch", f"kg={sorted(G)} config={sorted(kg_args.entities)}")
    counts = {t: len(G[t]) for t in G}
    rep["info"]["node_counts"] = counts
    rep["info"]["total_nodes"] = sum(counts.values())
    for t in G:
        if sorted(G[t]) != list(range(len(G[t]))):
            _err(rep, "kg_ids_not_contiguous", t)

    asym, dangling, badkeys = [], [], []
    for t in G:
        for i, rels in G[t].items():
            if set(rels) != set(rel[t]):
                badkeys.append((t, i))
                continue
            for r, ids in rels.items():
                tt = rel[t][r]
                for n in ids:
                    if tt not in G or n not in G[tt]:
                        dangling.append((t, i, r, n))
                    elif i not in G[tt][n].get(r, ()):
                        asym.append((t, i, r, n))
    if badkeys:
        _err(rep, "kg_relation_keys_mismatch", f"{len(badkeys)} e.g. {badkeys[:3]}")
    if dangling:
        _err(rep, "kg_dangling_edge", f"{len(dangling)} e.g. {dangling[:3]}")
    if asym:
        _err(rep, "kg_asymmetric_edge", f"{len(asym)} e.g. {asym[:3]}")

    inter = kg_args.interaction
    kg_enroll = {(u, c) for u, rels in G.get("user", {}).items() for c in rels.get(inter, ())}
    rep["info"]["kg_enroll_edges"] = len(kg_enroll)
    n_users, n_items = counts.get("user", 0), counts.get("item", 0)

    def to_pairs(lab):
        return {(u, c) for u, cs in (lab or {}).items() for c in cs}

    tr, te, va = to_pairs(train), to_pairs(test), to_pairs(validation)
    rep["info"]["split_pairs"] = {"train": len(tr), "test": len(te),
                                  "validation": len(va) if validation is not None else None}
    for name, lab in (("train", train), ("test", test), ("validation", validation)):
        if lab is None:
            continue
        bad = [(u, c) for u, cs in lab.items() for c in cs if not (0 <= u < n_users and 0 <= c < n_items)]
        if bad:
            _err(rep, "label_id_out_of_range", f"{name}: {len(bad)} e.g. {bad[:3]}")
        dup = sum(len(cs) - len(set(cs)) for cs in lab.values())
        if dup:
            _err(rep, "label_duplicates", f"{name}: {dup}", "warnings")

    if tr & te:
        _err(rep, "split_overlap_train_test", f"{len(tr & te)} e.g. {sorted(tr & te)[:3]}")
    if validation is not None:
        if tr & va:
            _err(rep, "split_overlap_train_validation", f"{len(tr & va)} e.g. {sorted(tr & va)[:3]}")
        if te & va:
            _err(rep, "split_overlap_test_validation", f"{len(te & va)} e.g. {sorted(te & va)[:3]}")
    if kg_enroll & te:
        _err(rep, "kg_contains_test_enrollment", f"{len(kg_enroll & te)} e.g. {sorted(kg_enroll & te)[:3]}")
    if validation is not None and kg_enroll & va:
        _err(rep, "kg_contains_validation_enrollment", f"{len(kg_enroll & va)}")
    if kg_enroll != tr:
        _err(rep, "kg_enroll_differs_from_train",
             f"kg-only={len(kg_enroll - tr)} train-only={len(tr - kg_enroll)}")

    if embeds is not None:
        dims = set()
        for t in G:
            if t not in embeds:
                _err(rep, "embedding_missing_key", t)
                continue
            if embeds[t].shape[0] != counts[t]:
                _err(rep, "embedding_rows_mismatch", f"{t}: embed={embeds[t].shape[0]} kg={counts[t]}")
            dims.add(embeds[t].shape[1])
        for t in rel:
            for r in rel[t]:
                if r not in embeds:
                    _err(rep, "embedding_missing_key", r)
                else:
                    dims.add(len(embeds[r][0]))
        if len(dims) > 1:
            _err(rep, "embedding_dim_mismatch", str(sorted(dims)))
        rep["info"]["embed_dim"] = sorted(dims)

    if data_dir is not None:
        for t, fname in kg_args.entity_files.items():
            p = os.path.join(data_dir, fname)
            if not os.path.exists(p):
                _err(rep, "vocab_file_missing", p, "warnings")
                continue
            with open(p, encoding="utf-8") as f:
                n = len(f.read().splitlines()) if os.path.getsize(p) else 0
            if n != counts.get(t):
                _err(rep, "vocab_size_mismatch", f"{t}: file={n} kg={counts.get(t)} ({p})")
    return rep
