# Spike B: same-item post-hoc path selection (smoke run)

**Status: SMOKE (epoch 2). Feasibility only. No claim of baseline or paper reproduction, and no claim about reward shaping.**

## Question

For the exact same user / item / topK as the baseline UPGPR, does choosing a different explanation path from the saved beam of that same pair satisfy a target relation property more often? This is path-only selection on a frozen item list. It is not item reranking.

Three stages are kept separate: candidate generation (saved beam, untouched), item ranking (reconstructed once, then frozen), path selection (S0..SL, never re-sorts items).

## Legacy evaluator trace (`test_agent.py`, `sum_prob=false`)

- Saved probs are already shifted by +1 (`probs + actmask` in `batch_beam_search`). The tool uses them as saved and never adds 1.
- Legacy path score = product of saved probs (`path_prob`). Best path per pair = max `path_prob` (stable, first in beam on ties).
- Train items are dropped. Users with no train labels are skipped. Items sort by `(dot(user+enroll, item), path_prob)` descending, stable, top 10.
- Real beam start node is `(self_loop, user, uid)`.

## Inputs

| Item | Value |
|---|---|
| Beam | `tmp/mooc/train_agent/policy_paths_epoch_2.pkl` (UPGPR, Xuetang), 393,921 paths |
| Epochs | TransE 2, agent 2 (config full run is 50 agent epochs) |
| Location | Drive `MyDrive/courserec/tmp/mooc` |
| Test users | 6,507 |
| Config | `config/UPGPR/mooc.json` (`topk` 25/5/1, `max_path_len` 3, `reward` binary_train) |
| Other beam on Drive | `tmp/mooc_pgpr3` (PGPR, epoch 2), not used |

Hashes (sha256): beam `b7b5ab92...3f8f98`, embeddings `7ae25a95...316512`, train_label `2e211fd4...bf006b`, test_label `6c9ac4ec...f53a66`, kg `193776f1...0bb97f`, checkpoint `269ecfaf...c48a65`, config `6e939b27...a2371ef9`. Full values in `manifest.json` on Drive.

## Baseline reconstruction

- Reconstructed metrics match `run_0.json` exactly (abs diff 0.0 on all four).
- All-user: NDCG 15.0954, Recall 22.5094, HR 37.2522, Precision 4.2435 (percent).
- Valid-only (6,502 users with at least 10 items): NDCG 15.0943, Recall 22.5114, HR 37.2655, Precision 4.2433.
- 5 users have fewer than 10 items. They stay in the all-user report.
- Probs audit: shifted, min 1.0004, max 2.0. ID audit: 0 type inconsistencies, 0 out-of-range ids.
- Paper reference for UPGPR 3 hop is NDCG about 18.4. This run is 15.1, expected at 2 epochs. Not comparable.

## Ranking invariants

For all five selectors, the ordered item IDs equal the baseline for every user, and NDCG/Recall/HR/Precision differ by 0.0 (tolerance 1e-12). Every selected path was asserted to belong to the original beam of the same (user, item) pair.

## Pre-registered targets (fixed before reading the real beam)

| Selector | Preferred path |
|---|---|
| S0 | legacy max score (baseline) |
| S1 | contains `item_teacher` |
| S2 | contains `item_school` |
| S3 | contains `item_concept` |
| SL | minimal effective length (non self-loop hops) in own beam |

Tie-break: higher legacy score, then earlier beam index. S0 is kept if it already meets the target. If no preferred path exists, S0 is kept.

## Results on topK final explanations (65,062 pairs)

| Selector | S0 already has | Pairs with preferred alt (ceiling) | Achieved | Pairs improved | Pairs worsened |
|---|---|---|---|---|---|
| S1 teacher | 0.00% | 0.00% | 0.00% | 0 | 0 |
| S2 school | 0.01% | 0.01% | 0.01% | 0 | 0 |
| S3 concept | 10.43% | 19.46% | 19.46% | 5,872 (9.03%) | 0 |
| SL shortest | 100% | 100% | 100% | 0 | 0 |

- S3 reaches the binary pairwise ceiling of the beam. S1, S2, SL also sit at their ceilings (nothing to gain).
- S3 paired change (per-user mean, selector minus S0): +0.0903, 95% CI [0.0880, 0.0927]. User-level bootstrap over test users, seed 20240131, 2000 resamples. This is not variance over training seeds.
- Pattern distribution (S0): `enroll_user` 89.56%, concept 10.43%, school 0.01%, teacher about 0%.
- Pattern entropy (base 2, pattern = per-node `<relation>_<node_type>` sequence including self_loop): S0 0.484 bits (4 patterns, normalized 0.242). S3 0.712 bits (3 patterns, normalized 0.449).
- Effective length is 3.0 for every path in every selector. No self-loops in this beam.
- Historical-item diversity (distinct per list / list size): S0 0.671, S3 0.681. Shared-entity diversity: 0.999 for both.
- Selector runtime: S0 0.34 s, S1 0.86 s, S2 0.57 s, S3 1.05 s, SL 1.24 s.

Beam-wide (secondary, all candidate pairs, not topK explanation metrics): 200,357 pairs. Share of pairs with at least one path containing the relation: teacher 0.13%, school 0.13%, concept 27.0%.

## Manual validation

Three S3-changed pairs from the real beam were inspected (uid 3618, 500, 6068). Candidates were recomputed from the raw beam. S0 was the max-product `enroll_user` path, S3 was the best concept path of the same pair, and the item rank was unchanged. The selected legacy score was lower than S0 in all 5,872 changed rows, as expected.

## Interpretation limits

- Policy trained for 2 epochs. Path mix (about 90% `enroll_user`, almost no teacher/school) is probably not representative of a converged agent.
- S1 and S2 did not change because the beam has almost no such paths. This is an availability bottleneck of the beam. Reward shaping could find paths outside the beam, so this does not show shaping is useless.
- S3 headroom (+9 pp) on a weak baseline is not evidence about reward value either way. If headroom stays high on a full-epoch beam, that means a strong post-hoc baseline, not that the reward has no value.
- SL is degenerate here (all effective lengths equal).

## Gaps

- `id_map_files` hashes are null in the manifest (runtime had no `data/mooc/processed_files`).
- Manifest `commit` is the upstream clone `7348dc2`, not the tool commit `66ca79c`. The tool is identified by sha256 `654f8e286e3642fcf06952ef08808c4f1010da8412ad49c71f74437ee7ebcae4`.
- Not done: list-level diversity selector (optional), full-epoch beam, local copy of the output files, push.

## Reproduce

Tool: `src/UPGPR/spike_b_posthoc.py`. Tests: `tests/test_spike_b_posthoc.py` (10 pass, includes comparison against the original `evaluate_paths` source extracted by AST).

```
python src/UPGPR/spike_b_posthoc.py --config config/UPGPR/mooc.json \
  --path_file tmp/mooc/train_agent/policy_paths_epoch_2.pkl \
  --out_dir tmp/mooc/spike_b_smoke_e2 \
  --logs_json tmp/mooc/evaluation/UPGPR_max_len_3/run_0.json
```

Outputs (on Drive `MyDrive/courserec/tmp/mooc/spike_b_smoke_e2/`): `baseline_reconstruction.json`, `selected_paths.jsonl`, `per_user.csv`, `summary.json`, `report.md`, `manifest.json`.

A run with epoch below the config value is labeled SMOKE automatically. Rerunning on a 50-epoch beam removes the label.
