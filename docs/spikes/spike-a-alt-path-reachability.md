---
type: howto
status: current
---

# Spike A: alternative-path reachability

Question: for each test (learner, positive course not in train), how many distinct paths exist, and how many relation types do they span? Separates full-graph reachability, environment-pruned reachability (`BatchKGEnvironment._get_actions`, top-`max_acts`) and beam availability. Counts show headroom only, not controllability (that is spike E).

Code: `src/spike_a/` (`core.py` enumeration, `artifacts.py` safe load + audit, `bruteforce.py` independent env walker, `cli.py`). Tests: `tests/alt_paths/`.

## Run

Needs the repo venv (Python 3.10, numpy<2, torch, easydict, scipy 1.11, scikit-learn 1.3).

```
python src/spike_a/cli.py \
  --config config/UPGPR/mooc.json \
  --artifacts-dir <run dir holding kg.pkl train_label.pkl test_label.pkl [validation_label.pkl] [transe_embed.pkl]> \
  --data-dir <data/mooc/processed_files> \
  --source-commit <sha that produced the artifacts> \
  --out-dir <new dir> \
  [--beam-paths <policy_paths_epoch_N.pkl>] [--horizons 3 4 --enable-h4] [--sample-users N --seed S]
```

Without `--source-commit` it exits 3 (AUDIT_FAILED) unless `--allow-unverified`, which labels outputs `COMPLETE_UNVERIFIED_PROVENANCE`. Missing kg/labels exits 2 (BLOCKED). Missing `transe_embed.pkl` completes full-graph only and marks pruned `BLOCKED_missing_embeddings`. Any audit error (test/validation enrolment in KG, split overlap, ID or shape mismatch, KG_ARGS drift, code drift since source commit) stops before any numbers.

Outputs in `--out-dir`: `manifest.json`, `pairs.csv` (T3; `pairs_h4.csv` etc. for extra horizons), `summary.json`, `report.md`, `witnesses.json`, `excluded_pairs.csv`, `diagnostics.json` (audit failures only).

Tests: `python -m pytest tests -q`

## Key definitions

- Counts are de-duplicated non-self-loop paths. `*_seqs` are env action sequences including self-loops (`sum_k n_k * C(horizon, k)`); they differ only when paths shorter than the horizon exist. In this KG items and non-items are bipartite, so at T3 every path has effective length 3 and the two counts coincide.
- Horizons 4 and 5 are opt-in. 4 is cheap but adds no new effective lengths on a bipartite KG. 5 is very expensive (about 320 s for 10 users, expansion cap hit), so expect censored lower bounds.
- `KG_ARGS.path_pattern` is a reward whitelist, not a search constraint, and is not applied.
