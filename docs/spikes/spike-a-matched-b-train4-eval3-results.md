---
type: spike
status: current
---

# Spike A matched to Spike B primary run (train 4 / eval 3)

Status: COMPLETE (Colab CPU, 1198 s, exact on both cohorts, 0 censored). Availability and headroom only. No claim on reward shaping, causal attribution to the policy, or learner usefulness.

Outputs: Drive `MyDrive/courserec/spike_a/mooc_T3_full_embeds_matched_B_train4_eval3/` (`manifest.json`, `provenance_comparison.json`, `positive_pairs.csv`, `topk_pairs.csv`, `summary.json`, `witnesses.json`, `report.md`, `diagnostics.json`). `checkpoint.jsonl` stayed on Colab. Old outputs untouched (new dir, `exist_ok=False`).
Tool: `src/spike_a/matched_cli.py` at commit `3aa263b` (no code change for this run). Env source commit `7348dc2` (stated by Spike B). Runtime was rebuilt (Python 3.10, numpy<2, torch CPU, scipy 1.11.4, scikit-learn 1.3.2, setuptools<81), tool shipped by base64 tarball with sha256 check (`247be750...`).

## 1. Provenance (hashes recomputed from files)

| artifact | A (stage 1) | B 4/3 manifest | recomputed B | status |
|---|---|---|---|---|
| kg | 193776f1c9e3 | 193776f1c9e3 | 193776f1c9e3 | match |
| train_label | 2e211fd4afe5 | 2e211fd4afe5 | 2e211fd4afe5 | match |
| test_label | 6c9ac4eca92c | 6c9ac4eca92c | 6c9ac4eca92c | match |
| validation_label | db13246c13ec | n/a | db13246c13ec | match |
| embeddings | 7ae25a958a1e | 5769c715786f | 5769c715786f | A differs (old 2-epoch smoke run), B = 3/3 embeddings |
| effective_config | n/a | bb18f328c10b | bb18f328c10b | match (differs from 3/3 `996b9d1a...`) |
| beam | b7b5ab92c954 | 5e6bae1ef937 | 5e6bae1ef937 | match (differs from 3/3 `7fe3509a...`) |

- KG, labels and embeddings are byte-identical to the 3/3 run. Only config and beam differ, so the full and pruned enumeration depends only on unchanged inputs.
- Audit: 0 errors. Reuse of old A pruned numbers: not allowed (other embeddings). Old A full-graph counts agree with this run for all 12,860 positive pairs (0 differences).
- Not audited on Colab: ID-map files (no `processed_files`). Pipeline commit is as stated by Spike B.

## 2. Cohorts (separate denominators)

| | positive (held-out) | top-K (S0) |
|---|---|---|
| pairs / users | 12,860 / 6,507 | 64,459 / 6,507 |
| note | identical to 3/3 | not pair-identical to 3/3 (64,461): the baseline ranking and beam changed |

## 3. Top-K funnel (denominator = all 64,459 pairs)

| target | full | pruned | beam | S0 selected | pruning loss | beam miss | selector miss |
|---|---|---|---|---|---|---|---|
| concept | 63,725 (98.86%) | 61,625 (95.60%) | 8,434 (13.08%) | 5,160 (8.01%) | 2,100 (3.26%) | 53,191 (82.52%) | 3,274 (5.08%) |
| teacher | 10,498 (16.29%) | 1,740 (2.70%) | 85 (0.13%) | 77 (0.12%) | 8,758 (13.59%) | 1,655 (2.57%) | 8 |
| school | 56,012 (86.90%) | 49,979 (77.54%) | 6 (0.009%) | 5 (0.008%) | 6,033 (9.36%) | 49,973 (77.53%) | 1 |
| co-learner | 64,459 | 64,447 | 60,541 (93.92%) | 59,217 (91.87%) | 12 | 3,906 | 1,324 |

Check against Spike B (exact counts from `selected_paths.jsonl`): concept beam 8,434 / S0 5,160 / gain 3,274, teacher 85 / 77 / 8, school 6 / 5 / 1. **All nine match.** Every beam path and every S0 path is accepted by both the full and the pruned enumeration (0 violations, 0 S0 paths outside the pair's beam).

## 4. Positive cohort (12,860 pairs)

| | full | pruned | beam |
|---|---|---|---|
| reachable | 100% | 99.97% | 32.00% |
| concept | 98.78% | 96.30% | 4.31% |
| teacher | 22.60% | 4.72% | 0.04% |
| school | 76.26% | 67.85% | 0.01% |
| co-learner | 99.91% | 99.71% | 30.75% |
| paths per reachable pair (median / p90 / max) | 902 / 2,842 / 16,477 | 336 / 1,022 / 7,562 | 2 / 7 / 22 |

Full and pruned values equal the 3/3 run exactly (same graph and embeddings). Only the beam column differs.

## 5. Change vs 3/3 (same graph, embeddings, pruning; different beam)

| | 3/3 | 4/3 |
|---|---|---|
| top-K pairs | 64,461 | 64,459 |
| concept beam / S0 / selector miss | 16,811 / 11,986 / 4,825 | 8,434 / 5,160 / 3,274 |
| teacher beam / S0 | 268 / 261 | 85 / 77 |
| school beam / S0 | 117 / 93 | 6 / 5 |
| positive pairs with a beam path | 30.45% | 32.00% |
| positive pairs with a concept beam path | 8.77% | 4.31% |

- Full and pruned top-K counts shift a little (concept 63,565 to 63,725, teacher 9,674 to 10,498) only because the cohort is not pair-identical. Compare funnels within one run; cross-run beam comparisons are on different denominators (64,461 vs 64,459).
- The 4/3 beam holds far fewer concept, teacher and school alternatives than the 3/3 beam, while pruned availability is unchanged. The headroom is lost mostly between pruned and beam in both runs.
- Not concluded: why the beam differs (training horizon, seed and search all differ and are not separated here).

## 6. Verification

- 56 unit tests pass (committed state, unchanged). Brute force through the real `env._get_actions`: 66 pair-modes (full and pruned), 0 mismatches, none aborted.
- `exact_full_cohorts` true, 0 censored pairs. Witnesses: edges and pruning actions verified, 0 problems (`witnesses.json`).
- Tool bundle sha256 `221c1e95aed8b9aa...` (hidden `._*` files ignored, per-file hashes equal the committed ones).

## 7. Limits

- Single training seed per run; top-K cohort depends on the baseline ranking and beam, so it is not a random sample.
- Embedding epoch 30 is established by checkpoint equality in the 3/3 audit; the 4/3 embeddings file is byte-identical to it.
- ID-map files not audited on Colab. Nothing pushed (`origin` is upstream `epfl-ml4ed/courserec`).

## 8. Reproduce

```
python src/spike_a/matched_cli.py \
  --config <B43>/effective_config.json --base-config config/UPGPR/mooc.json \
  --artifacts-dir <B43> --beam-paths <B43>/train_agent/policy_paths_epoch_50.pkl \
  --b-dir <B43>/spike_b_t4e3 --a-dir <drive>/spike_a/mooc_T3 --a-artifacts-dir <drive>/tmp/mooc \
  --out-dir <new dir> --env-source-commit 7348dc251a000f0bea593f998d4b1e69d3c24728 \
  --tool-commit 3aa263b --expect-topk-pairs 64459 \
  --ref-hash kg=193776f1c9e3181cf8e96335d5d4c6fc18783a7a5b2761dcf68d26959c0bb97f \
  --ref-hash train_label=2e211fd4afe5dbb8a85140accca4627fa7c7824d69d0d5b47719845bc2bf006b \
  --ref-hash test_label=6c9ac4eca92cad058b8784b0e97fcb67ce7f1bb5ab7a4ad13f96013539f53a66 \
  --ref-hash embeddings=5769c715786ffbd3e635078e047284b60c7e1593b2602b5aabee90eab877e787 \
  --ref-hash beam=5e6bae1ef937d33ca38b96427e75b832add7f5d1a7c29911b6cbf3a0716e990f \
  --ref-hash effective_config=bb18f328c10b1b3e3ec39b73f9ae49b641d0025dc556f3d0dfac3a6e5b1b42e9 [--resume]
```

`<B43>` = `MyDrive/courserec/tmp/mooc_train4_eval3`.
