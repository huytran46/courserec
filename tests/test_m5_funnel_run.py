import csv
import json
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "UPGPR"))
import m5_funnel_run as fr  # noqa: E402
import m5_run  # noqa: E402


def _write_users(d, n_items, n_conc):
    os.makedirs(d, exist_ok=True)
    for sel in ("S0", "S3"):
        with open(os.path.join(d, f"per_user_test_{sel}.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["uid", "n_items", "n_valid_expl", "n_concept_expl"])
            for u, (ni, nc) in enumerate(zip(n_items, n_conc)):
                w.writerow([u, ni, ni, nc])


def test_slots_separates_missing_slots_from_short_users(tmp_path):
    root = str(tmp_path)
    json.dump({"lambda_star": 5}, open(os.path.join(root, "summary.json"), "w"))
    os.makedirs(os.path.join(root, "tables"))
    for s in fr.SEEDS:
        for lam in (0.0, 5.0):
            # 4 users with lists of 10, 10, 7, 0: 3 + 10 = 13 missing slots over 2 short users, 40 slots
            _write_users(m5_run.arm_dir(root, s, lam), [10, 10, 7, 0], [4, 3, 2, 0])
    fr.cmd_slots(SimpleNamespace(root=root))
    r = json.load(open(os.path.join(root, "tables", "slot_reconciliation.json")))["per_seed"]["23_shaped"]["S3"]
    assert r["slots"] == 40 and r["returned_slots"] == 27 and r["missing_slots"] == 13
    assert r["users_lt10"] == 2 and r["users_zero"] == 1 and r["missing_slots_in_short_users"] == 13
    assert r["concept_slots"] == 9 and r["returned_non_concept_slots"] == 18 and r["no_concept_slots"] == 31
