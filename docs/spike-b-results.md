# Spike B report: same-item post-hoc path selection

Scope: UPGPR on MOOCCube (Xuetang), saved beam outputs only. No retraining of the selectors, no change to embeddings, candidate items or ranking. Wording follows the thesis rule: results are about explanation-oriented path properties (relation composition), not human perception.

## Summary

- Primary protocol is train path length 4 / eval path length 3 ("4/3"). Sensitivity baseline is 3/3. Both: TransE 30 epochs (identical embeddings), agent 50 epochs, seed 23, one training run each.
- For the exact same user / item / top-10 as the baseline, a same-pair path selector that prefers a concept path raises concept presence from 8.01% to 13.08% (4/3; 18.59% to 26.08% in 3/3). Item order and all ranking metrics are unchanged (diff 0.0).
- 13.08% is the ceiling of the beam for that binary target: 86.92% of top-10 pairs have no concept path at all in the beam of the same pair.
- Teacher and school paths are almost absent in the beam (4/3: 0.12% and 0.011% of candidate pairs have any). Post-hoc selection cannot move them. This is an availability bottleneck of the beam. It does not show that reward shaping is useless, because shaping can find paths outside the beam.
- The concept lift is a strong post-hoc baseline, not evidence about the value of a reward. It stays as the concept baseline for shaping comparisons.
- Limits: one seed per run, no training variance; 4/3 vs 3/3 difference is not tested for significance; manifest lacks `id_map_files` and `commit`.

## Design

Three stages are kept separate:
1. Candidate generation: the saved beam, untouched.
2. Item ranking: reconstructed once from the legacy evaluator logic, then frozen (exact ordered item IDs per user).
3. Path selection: selectors choose a path only among the beam paths of the same (user, item) pair. Items are never re-sorted afterwards.

Selectors, with targets fixed before any real beam was read (binary pairwise, no combined objective):

| Selector | Preferred path |
|---|---|
| S0 | legacy max score (baseline) |
| S1 | contains `item_teacher` |
| S2 | contains `item_school` |
| S3 | contains `item_concept` |
| SL | minimal effective length (non self-loop hops) in own beam |

Tie-break: higher legacy score, then earlier beam index. S0 is kept if it already meets the target, and also if no preferred path exists. "Contains relation" means any hop after the start node; in this beam it equals the terminal hop (all paths have 4 nodes).

## Legacy evaluator trace (`test_agent.py`, `sum_prob=false`)

- Saved probs are already shifted by +1 (`probs + actmask` in `batch_beam_search`). The tool uses them as saved and never adds 1.
- Legacy path score = product of saved probs (`path_prob`). Best path per pair = max `path_prob` (stable, first in beam on ties).
- Train items are dropped. Users with no train labels are skipped. Items sort by `(dot(user+enroll, item), path_prob)` descending, stable, top 10.
- Real beam start node is `(self_loop, user, uid)`.
- Policy network dimensions do not depend on `max_path_len`, so a model trained at 4 hops runs in a 3-hop environment; `topk` length must equal the eval `max_path_len`.

## Runs

| | 4/3 (primary) | 3/3 (sensitivity) |
|---|---|---|
| Train / eval path length | 4 / 3 | 3 / 3 |
| Eval `topk` | 25, 5, 1 | 25, 5, 1 |
| Embeddings | 30-epoch TransE, shared | same file |
| Agent epochs, seed | 50, 23 | 50, 23 |
| Drive dir (`MyDrive/courserec/tmp/`) | `mooc_train4_eval3` | `mooc_full` |
| Spike B output | `spike_b_t4e3/` | `spike_b_full_e50/` |
| Pipeline | upstream `7348dc2`, T4 | upstream `7348dc2`, T4 |
| Timing (UTC, 2026-10-31) | train 15:17 to 16:07, test to 16:15 | start 09:05, beam done 09:45 |

The 4/3 run reused `dataset.pkl`, `kg.pkl`, labels and `transe_embed.pkl` copied byte for byte from the 3/3 run (hashes verified before training), so the split is identical and TransE was not retrained. An earlier 2-epoch smoke run is in the appendix.

## Baseline reconstruction and invariants

| | 4/3 | 3/3 |
|---|---|---|
| Reconstructed vs `run_0.json` | abs diff 0.0 | abs diff 0.0 |
| NDCG all-user (%) | 19.1470 | 18.4063 |
| Recall / HR / Precision all-user (%) | 26.1315 / 43.1228 / 5.0646 | 24.7196 / 41.3862 / 4.7973 |
| NDCG valid-only (%) | 18.7782 (6,249 users) | 18.0772 (6,230 users) |
| Recall / HR / Precision valid-only (%) | 25.8502 / 42.9509 / 4.9832 | 24.5346 / 41.3804 / 4.7448 |
| Users with fewer than 10 items (kept in all-user) | 258 | 277 |
| Beam paths | 393,920 | 393,917 |
| Top-K pairs | 64,459 | 64,461 |

- Probs audit: shifted, min 1.0, max 2.0. ID audit: 0 type inconsistencies and 0 out-of-range ids, both runs.
- For all five selectors in both runs, ordered item IDs equal the baseline for every user and NDCG, Recall, HR, Precision differ by 0.0 (tolerance 1e-12), all-user and valid-only.
- Paper reference for UPGPR 3 hop on Xuetang is NDCG 18.43; 3/3 gives 18.41 for one seed. Read as near paper level, not as a full reproduction. The paper's 4-hop figure uses 4-hop evaluation, so it is not comparable with 4/3.

## Results on top-K final explanations

Top-K pairs: 64,459 (4/3), 64,461 (3/3).

| Selector | Target | S0 already has (4/3) | Ceiling = achieved (4/3) | Pairs improved (4/3) | Pairs worsened |
|---|---|---|---|---|---|
| S1 | teacher | 77 (0.12%) | 85 (0.13%) | 8 | 0 |
| S2 | school | 5 (0.01%) | 6 (0.01%) | 1 | 0 |
| S3 | concept | 5,160 (8.01%) | 8,434 (13.08%) | 3,274 (5.08%) | 0 |
| SL | min effective length | 100% | 100% | 0 | 0 |

S1 and S2 achieved counts are derived (S0 plus improved pairs). Every selector reaches its ceiling. The ceiling counts every pair with at least one path having the property, including pairs where S0 already chose it.

Comparison with 3/3:

| | 3/3 | 4/3 |
|---|---|---|
| S0 pattern mix: enroll_user / concept / teacher / school | 80.86% / 18.59% / 0.40% / 0.14% | 91.87% / 8.01% / 0.12% / 0.01% |
| S0 counts: enroll_user / concept / teacher / school | 52,121 / 11,986 / 261 / 93 | 59,217 / 5,160 / 77 / 5 |
| S3 concept: S0 already, ceiling = achieved | 18.59%, 26.08% | 8.01%, 13.08% |
| S3 pairs gained (pair-level) | 4,825 (7.49 pp) | 3,274 (5.08 pp) |
| S3 user-mean paired change, 95% CI | +7.45 pp [7.24, 7.66] | +5.04 pp [4.86, 5.23] |
| S1 / S2 user-mean paired change | +0.01 pp / +0.04 pp | +0.01 pp / 0.00 pp |
| Candidate pairs with any concept path (beam-wide) | 29.3% | 15.9% |
| Candidate pairs with any teacher / school path | 0.36% / 0.16% | 0.12% / 0.011% |
| Pattern entropy S0 to S3 (bits, base 2) | 0.745 to 0.879 | 0.417 to 0.574 |
| Normalized entropy S0 to S3 | 0.373 to 0.439 | 0.208 to 0.287 |
| Historical-item diversity S0 to S3 | 0.6855 to 0.6848 | 0.6844 to 0.6861 |
| Shared-entity diversity S0 to S3 | 0.9981 to 0.9984 | 0.9898 to 0.9903 |
| Selector runtime S3 | 1.20 s | 0.88 s |

Notes on definitions:
- Paired change: per-user mean of the own-target indicator, selector minus S0. Uncertainty is a user-level bootstrap over test users, seed 20240131, 2000 resamples. It is not training-seed variance.
- Pattern = per-node `<relation>_<node_type>` sequence including the self_loop start node (legacy key). Normalized entropy divides by log2 of the observed pattern count; every selector has 4 patterns, so all use log2(4) = 2.
- Diversity = distinct historical items (or shared entities) per user list divided by list size, only for lists whose paths are all 4-node.
- Effective length is 3.0 for every path in every selector and there are no self-loops, in both runs (also at 4-hop training). SL is degenerate.
- Beam-wide numbers cover all candidate pairs and are secondary, not top-K explanation metrics.

Path transitions (S0 kind to selector kind, changed pairs only):
- 4/3: S1 8 pairs (5 enroll_user, 2 concept, 1 school to teacher). S2 1 pair (concept to school). S3 3,274 pairs, all enroll_user to concept. SL 0.
- 3/3: S1 7 (5 enroll_user, 2 concept to teacher). S2 24 (17 enroll_user, 6 concept, 1 teacher to school). S3 4,825 (4,820 enroll_user, 5 teacher to concept; teacher 261 to 256). SL 0.
- So each selector keeps its own target, but can move paths that carried another relation (3/3 S3: teacher 261 to 256). The correct statement is "the target presence never drops", not "all properties are preserved".

## Interpretation

- Concept is the only property with real headroom in the beam. Post-hoc selection lifts it with zero ranking change and needs no retraining (the selector costs about 1 s).
- The 4/3 policy ranks better (NDCG +0.74 over 3/3, single seed) but its 3-hop beam is more dominated by co-learner paths, so concept headroom roughly halves. Headroom depends on the training protocol.
- Teacher and school: nothing to select, so no post-hoc movement. Treat them as stress-test candidates for shaping, and recheck their reachability first (Spike A).
- High post-hoc headroom means a strong baseline for that property. It does not show the reward has no value, since most pairs have no concept path in their beam and selection cannot reach them.
- 4/3 and 3/3 top-K cohorts differ (different pairs), so counts across runs are not pair-matched.

## Relation to Spike A

- Spike A's matched rerun on the 3/3 top-K pairs reproduced all nine counts above exactly (concept ceiling 16,811, S0 11,986, S3 gain 4,825; teacher 268, 261, 7; school 117, 93, 24). Source: Spike A's own results note, not re-verified in this spike.
- The 4/3 run keeps the embeddings hash (`5769c715...`), so Spike A's embedding-dependent pruning does not need a rerun; only its cohort (the 4/3 top-K pairs) would change.

## Limits and verification scope

- One training seed per run. No training variance. The 4/3 vs 3/3 difference is not tested for significance.
- Independently rechecked from output files and beam pickles (not the tool's in-run asserts): all 322,305 (3/3) and 322,295 (4/3) selected rows are paths of the same (user, item) pair in the original beam, 0 violations; the transition counts above.
- Not independently verified: the manifest holds only the TEST_AGENT config and no copies of inputs, so hashes cannot be rechecked from the manifest alone. Train horizon and epochs rest on `effective_config.json` in each Drive run dir (3/3: TRAIN_AGENT max_path_len 3; 4/3: max_path_len 4, TEST 3), the checkpoint files seen on Drive, and `run.log` heads (3/3 `run.log` was read only in part because Drive syncs appended logs slowly).
- `id_map_files` hashes are null and manifest `commit` is null in both manifests (the spike ran outside a git checkout and without `processed_files`). The pipelines ran on upstream `7348dc251a000f0bea593f998d4b1e69d3c24728`. The tool is identified by sha256.
- Not done: list-level diversity selector (optional, would be exploratory and list-dependent), push (origin is upstream `epfl-ml4ed/courserec`, not a fork), local copy of the Drive outputs.

## Provenance (sha256)

| Item | 4/3 | 3/3 |
|---|---|---|
| Beam | `5e6bae1ef937d33ca38b96427e75b832add7f5d1a7c29911b6cbf3a0716e990f` | `7fe3509a1ad44f9f199341b3abcae89dbea3874e841c11a4e778877077232b06` |
| Agent checkpoint (epoch 50) | `1dcc2f85d99bc5117c6c51ec7055c39f0ea7c475f4e9b35d14fac1ed46e3e460` | `29bf511fbc10174ad4182a20c2362ac683eef5caecdb7619f28686efca0375b6` |
| `effective_config.json` | `bb18f328c10b1b3e3ec39b73f9ae49b641d0025dc556f3d0dfac3a6e5b1b42e9` | `996b9d1a29944182c9333209cb173fcc44b4276d6b916ce98358e1de3feb28a6` |
| `transe_embed.pkl` (shared) | `5769c715786ffbd3e635078e047284b60c7e1593b2602b5aabee90eab877e787` | same |
| `kg.pkl` (shared) | `193776f1c9e3181cf8e96335d5d4c6fc18783a7a5b2761dcf68d26959c0bb97f` | same |
| `train_label.pkl` (shared) | `2e211fd4afe5dbb8a85140accca4627fa7c7824d69d0d5b47719845bc2bf006b` | same |
| `test_label.pkl` (shared) | `6c9ac4eca92cad058b8784b0e97fcb67ce7f1bb5ab7a4ad13f96013539f53a66` | same |
| Tool `spike_b_posthoc.py` | `654f8e286e3642fcf06952ef08808c4f1010da8412ad49c71f74437ee7ebcae4` (commit `66ca79c`) | same |

## Reproduce

Tool `src/UPGPR/spike_b_posthoc.py`. Tests `tests/test_spike_b_posthoc.py` (10 pass, including a comparison against the original `evaluate_paths` source extracted by AST).

```
# 4/3 (primary)
python src/UPGPR/spike_b_posthoc.py --config tmp/mooc_train4_eval3/effective_config.json \
  --path_file tmp/mooc_train4_eval3/train_agent/policy_paths_epoch_50.pkl \
  --out_dir tmp/mooc_train4_eval3/spike_b_t4e3 \
  --logs_json tmp/mooc_train4_eval3/evaluation/UPGPR_train4_eval3/run_0.json

# 3/3 (sensitivity)
python src/UPGPR/spike_b_posthoc.py --config tmp/mooc_full/effective_config.json \
  --path_file tmp/mooc_full/train_agent/policy_paths_epoch_50.pkl \
  --out_dir tmp/mooc_full/spike_b_full_e50 \
  --logs_json tmp/mooc_full/evaluation/UPGPR_max_len_3/run_0.json
```

Outputs per run: `baseline_reconstruction.json`, `selected_paths.jsonl` (fields uid, cid, rank, selector, path, old_property, new_property, legacy_score, s0_legacy_score, changed_path), `per_user.csv`, `summary.json`, `report.md`, `manifest.json`. A run with epochs below the config value is labeled SMOKE automatically.

## Next steps (proposed)

1. Spike A: rerun the matched analysis on the 4/3 top-K cohort if 4/3 numbers are needed.
2. Spike E (small, reward-only): does shaping find paths outside the beam, and does it give a benefit beyond selector S3?
3. Teacher and school stress tests, after rechecking their reachability under full-embedding pruning.
4. Multiple training seeds for the chosen protocol before any claim about differences.

## Appendix: 2-epoch smoke run (feasibility only, do not cite)

Beam `tmp/mooc/train_agent/policy_paths_epoch_2.pkl` (TransE 2, agent 2). Baseline NDCG 15.10, not comparable to the paper. Reconstruction matched its log exactly and all invariants held. S3 lifted concept presence from 10.4% to 19.5% (ceiling); teacher and school were essentially absent. Outputs in `tmp/mooc/spike_b_smoke_e2/` on Drive.
