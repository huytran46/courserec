"""Tiny hand-built KG with known counts, built with the repo's real KnowledgeGraph class.

Users u0..u2, items c0..c4, teachers t0,t1, school s0, concepts k0,k1.
Train enrolments: u0:{c0,c1}  u1:{c0,c2}  u2:{c1,c2}   Test positive: (u0, c2).
Item relations:
  teacher: c0-t0, c2-t0, c1-t1
  school : c0-s0, c1-s0, c2-s0
  concept: c0-k1, c2-k1, c1-k0, c2-k0
Expected length-3 paths u0 -> a -> m -> c2 (a in {c0,c1}):
  enroll : c0-u1-c2, c1-u2-c2                 = 2
  teacher: c0-t0-c2                           = 1
  school : c0-s0-c2, c1-s0-c2                 = 2
  concept: c0-k1-c2, c1-k0-c2                 = 2     total 7, 4 families
"""
import json
import os
from types import SimpleNamespace as NS

import numpy as np
from easydict import EasyDict as edict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TRAIN = {0: [0, 1], 1: [0, 2], 2: [1, 2]}
TEST = {0: [2]}
SIZES = {"user": 3, "item": 5, "school": 1, "teacher": 2, "concept": 2}


def kg_args():
    with open(os.path.join(ROOT, "config", "UPGPR", "mooc.json")) as f:
        return edict(json.load(f))["KG_ARGS"]


def build_kg(train=TRAIN):
    from knowledge_graph import KnowledgeGraph

    args = kg_args()
    inter = [(u, c) for u, cs in train.items() for c in cs]
    ds = NS(
        user=NS(vocab_size=3), item=NS(vocab_size=5), school=NS(vocab_size=1),
        teacher=NS(vocab_size=2), concept=NS(vocab_size=2),
        interactions=NS(data=inter),
        item_teacher=NS(data=[[0], [1], [0], [], []]),
        item_school=NS(data=[[0], [0], [0], [], []]),
        item_concept=NS(data=[[1], [0], [0, 1], [], []]),
    )
    return KnowledgeGraph(ds, args), args


def build_embeds(seed=0, dim=8):
    rng = np.random.RandomState(seed)
    e = {t: rng.randn(k, dim) for t, k in SIZES.items()}
    for r in ("enroll", "item_teacher", "item_school", "item_concept"):
        e[r] = (rng.randn(dim), 0.0)
    return e
