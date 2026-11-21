"""Spike E evaluation of one beam: unchanged baseline evaluator + Spike B selectors (S0..S3, SL).

python src/UPGPR/spike_e_eval.py --config <effective_config.json> --path_file <beam.pkl> --out_dir <dir> --epochs 5
Writes <out_dir>/baseline_eval/run.json (original evaluate_paths metrics) and <out_dir>/spike_b/*.
"""
from __future__ import absolute_import, division, print_function

import argparse
import io
import json
import os
import sys
import time
import types
from contextlib import redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--path_file", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--epochs", type=int, default=5)
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    ta = cfg["TEST_AGENT"]
    import test_agent
    from easydict import EasyDict as edict
    from utils import load_labels
    import spike_b_posthoc as sb

    os.makedirs(a.out_dir, exist_ok=True)
    test_agent.args = types.SimpleNamespace(tmp_dir=ta["tmp_dir"])  # evaluate() reads this module global
    kg_args = edict(cfg["KG_ARGS"])
    train_labels = load_labels(ta["tmp_dir"], "train")
    test_labels = load_labels(ta["tmp_dir"], "test")
    buf = io.StringIO()
    t0 = time.time()
    with redirect_stdout(buf):
        test_agent.evaluate_paths(ta["tmp_dir"], a.path_file, train_labels, test_labels, kg_args, False,
                                  os.path.abspath(os.path.join(a.out_dir, "baseline_eval")), "run.json",
                                  validation=False, sum_prob=ta["sum_prob"])
    t_eval = time.time() - t0
    open(os.path.join(a.out_dir, "baseline_eval", "stdout.txt"), "w").write(buf.getvalue())
    logs = os.path.join(a.out_dir, "baseline_eval", "run.json")
    t0 = time.time()
    rc = sb.run(a.config, a.path_file, os.path.join(a.out_dir, "spike_b"), epochs=a.epochs, logs_json=logs)
    json.dump({"baseline_eval_seconds": t_eval, "spike_b_seconds": time.time() - t0, "spike_b_rc": rc},
              open(os.path.join(a.out_dir, "eval_runtime.json"), "w"))
    sys.exit(rc)


if __name__ == "__main__":
    main()
