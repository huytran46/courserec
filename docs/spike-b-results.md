# Spike B: same-item post-hoc path selection

**Status: two full-epoch runs, both TransE 30 (same embeddings), agent 50 epochs, seed 23, one training run each.** Primary protocol: train path length 4 / eval path length 3 ("4/3", next section). Sensitivity baseline: train 3 / eval 3 ("3/3", the rest of this document, unless a section says 4/3). An earlier 2-epoch smoke run is kept in the appendix as a feasibility check only.

## Primary protocol 4/3: results

Config: `effective_config.json` of the 3/3 run with `TRAIN_AGENT.max_path_len = 4`, `TEST_AGENT.max_path_len = 3`, `topk` 25/5/1, `tmp_dir = tmp/mooc_train4_eval3`. TransE was not retrained: `dataset.pkl`, `kg.pkl`, labels and `transe_embed.pkl` were copied byte for byte from the 3/3 run and their sha256 verified before training. Pipeline: upstream clone at `7348dc2`, `train_agent` then `test_agent` on a Colab T4 (train 15:17 to 16:07 UTC, test to 16:15 UTC). Output on Drive: `MyDrive/courserec/tmp/mooc_train4_eval3/` (`train_agent/`, `evaluation/UPGPR_train4_eval3/run_0.json`, `effective_config.json`, `spike_b_t4e3/`).

sha256: beam `5e6bae1ef937d33ca38b96427e75b832add7f5d1a7c29911b6cbf3a0716e990f` (393,920 paths), agent checkpoint epoch 50 `1dcc2f85d99bc5117c6c51ec7055c39f0ea7c475f4e9b35d14fac1ed46e3e460`, config `bb18f328c10b1b3e3ec39b73f9ae49b641d0025dc556f3d0dfac3a6e5b1b42e9`. Embeddings, kg, train_label and test_label hashes are identical to the 3/3 run (embeddings `5769c715...877e787`).

Checks: reconstructed metrics match `run_0.json` exactly (abs diff 0.0); ordered item IDs and all ranking metrics unchanged for all 5 selectors (diff 0.0); all 322,295 selected rows (64,459 pairs x 5) re-verified from the output file against the beam pickle as paths of the same (user, item) pair, 0 violations; probs shifted (min 1.0, max 2.0), 0 type or id errors.

| | 3/3 | 4/3 (primary) |
|---|---|---|
| NDCG all-user (%) | 18.4063 | 19.1470 |
| Recall / HR / Precision (%) | 24.7196 / 41.3862 / 4.7973 | 26.1315 / 43.1228 / 5.0646 |
| NDCG valid-only (%) | 18.0772 (6,230 users) | 18.7782 (6,249 users) |
| Users with fewer than 10 items | 277 | 258 |
| Top-K pairs | 64,461 | 64,459 |
| S0 pattern mix: enroll_user / concept / teacher / school | 80.86% / 18.59% / 0.40% / 0.14% | 91.87% / 8.01% / 0.12% / 0.01% |
| S0 counts: concept / teacher / school | 11,986 / 261 / 93 | 5,160 / 77 / 5 |
| S3 concept: S0 already, ceiling = achieved | 18.59%, 26.08% | 8.01%, 13.08% (8,434 pairs) |
| S3 pairs gained (pair-level) | 4,825 (7.49 pp) | 3,274 (5.08 pp) |
| S3 user-mean paired change, 95% CI | +7.45 pp [7.24, 7.66] | +5.04 pp [4.86, 5.23] |
| Pattern entropy S0 to S3 (bits, base 2) | 0.745 to 0.879 | 0.417 to 0.574 |
| Candidate pairs with any concept path (beam-wide) | 29.3% | 15.9% |
| Candidate pairs with any teacher / school path | 0.36% / 0.16% | 0.12% / 0.011% |

4/3 details: S1 teacher 8 pairs changed (5 enroll_user, 2 concept, 1 school to teacher; teacher 77 to 85 [derived]); S2 school 1 pair changed (concept to school; school 5 to 6 [derived]); S3 3,274 pairs, all enroll_user to concept, teacher (77) and school (5) unchanged; SL 0 changed. Every selector reaches its ceiling. Effective length is 3.0 for every path and there are no self-loops, even though training used 4 hops, so SL is still degenerate. Historical-item diversity S0 0.6844, S3 0.6861; shared-entity diversity S0 0.9898, S3 0.9903. Selector runtime S3 0.88 s. Normalized entropy uses the same denominator log2(4) = 2 for all selectors (4 observed patterns).

Reading: the 4/3 policy ranks better (NDCG +0.74 over 3/3 in this single seed, not tested for significance) but its 3-hop beam is more dominated by co-learner paths, so concept headroom for post-hoc selection roughly halves (ceiling 13.1% vs 26.1%) and teacher and school are almost absent. The post-hoc concept lift stays clearly above zero (+5.0 pp) with unchanged ranking, so S3 remains the concept baseline for shaping comparisons. Teacher and school are still an availability bottleneck of the beam, which says nothing about whether shaping can find them. The paper's 4-hop NDCG refers to eval at 4 hops, so it is not comparable with this 4/3 number. The top-K cohorts of 3/3 and 4/3 differ (64,461 vs 64,459 pairs, different items), so counts between the two runs are not pair-matched.

Reproduce 4/3: same tool and command as below with `--config tmp/mooc_train4_eval3/effective_config.json`, `--path_file tmp/mooc_train4_eval3/train_agent/policy_paths_epoch_50.pkl`, `--out_dir tmp/mooc_train4_eval3/spike_b_t4e3`, `--logs_json tmp/mooc_train4_eval3/evaluation/UPGPR_train4_eval3/run_0.json`. Same limits as below: one seed, `id_map_files` null and manifest `commit` null in the manifest.

## Question

For the exact same user / item / topK as the baseline UPGPR, does choosing a different explanation path from the saved beam of that same pair satisfy a target relation property more often? This is path-only selection on a frozen item list. It is not item reranking.

Three stages are kept separate: candidate generation (saved beam, untouched), item ranking (reconstructed once, then frozen), path selection (S0..SL, never re-sorts items).

## Legacy evaluator trace (`test_agent.py`, `sum_prob=false`)

- Saved probs are already shifted by +1 (`probs + actmask` in `batch_beam_search`). The tool uses them as saved and never adds 1.
- Legacy path score = product of saved probs (`path_prob`). Best path per pair = max `path_prob` (stable, first in beam on ties).
- Train items are dropped. Users with no train labels are skipped. Items sort by `(dot(user+enroll, item), path_prob)` descending, stable, top 10.
- Real beam start node is `(self_loop, user, uid)`.

## Inputs (full run)

| Item | Value |
|---|---|
| Beam | `tmp/mooc_full/train_agent/policy_paths_epoch_50.pkl` (UPGPR, Xuetang), 393,917 paths |
| Training | TransE 30 epochs, agent 50 epochs, seed 23, Colab T4 (`torch 2.14.1+cu130`, cuda True) |
| Config | `config/UPGPR/mooc.json` with `tmp_dir` changed to `tmp/mooc_full` (saved as `effective_config.json`); `topk` 25/5/1, `max_path_len` 3, `reward` binary_train |
| Pipeline | upstream clone `7348dc2`, `preprocess_mooc -> make_dataset -> train_transe_model -> train_agent -> test_agent` |
| Location | Drive `MyDrive/courserec/tmp/mooc_full` |
| Test users | 6,507 |

sha256: beam `7fe3509a...077232b06`, embeddings `5769c715...ab877e787`, checkpoint `29bf511f...0375b6`, config `996b9d1a...eb28a6`, train_label `2e211fd4...bf006b`, test_label `6c9ac4ec...f53a66`, kg `193776f1...0bb97f`. Train/test labels and kg hashes equal the smoke run, so the split is identical. Full values in `manifest.json` on Drive.

## Baseline reconstruction

- Reconstructed metrics match `run_0.json` exactly (abs diff 0.0 on all four).
- All-user (%): NDCG 18.4063, Recall 24.7196, HR 41.3862, Precision 4.7973.
- Valid-only (6,230 users with at least 10 items, %): NDCG 18.0772, Recall 24.5346, HR 41.3804, Precision 4.7448.
- 277 users have fewer than 10 items. They stay in the all-user report.
- Probs audit: shifted, min 1.0, max 2.0. ID audit: 0 type inconsistencies, 0 out-of-range ids over 393,917 paths.
- Paper reference (Xuetang, UPGPR 3 hop): NDCG 18.43. This run is 18.41 (all-user). Read this as a result near paper level for one seed. It is not a conclusion that the paper is fully reproduced (single seed, no variance estimate, protocol 3/3 only).

## Ranking invariants

For all five selectors, ordered item IDs equal the baseline for every user, and NDCG/Recall/HR/Precision differ by 0.0 (tolerance 1e-12) in both all-user and valid-only. Every selected path was asserted to belong to the original beam of the same (user, item) pair.

## Pre-registered targets (fixed before reading any real beam)

| Selector | Preferred path |
|---|---|
| S0 | legacy max score (baseline) |
| S1 | contains `item_teacher` |
| S2 | contains `item_school` |
| S3 | contains `item_concept` |
| SL | minimal effective length (non self-loop hops) in own beam |

Tie-break: higher legacy score, then earlier beam index. S0 is kept if it already meets the target. If no preferred path exists, S0 is kept.

## Results on topK final explanations (64,461 pairs)

| Selector | S0 already has | Pairs with preferred alt (ceiling) | Achieved | Pairs improved | Pairs worsened |
|---|---|---|---|---|---|
| S1 teacher | 0.40% | 0.42% | 0.42% | 7 | 0 |
| S2 school | 0.14% | 0.18% | 0.18% | 24 | 0 |
| S3 concept | 18.59% | 26.08% | 26.08% | 4,825 (7.49%) | 0 |
| SL shortest | 100% | 100% | 100% | 0 | 0 |

- All selectors reach the binary pairwise ceiling of the beam. The ceiling column counts every pair with at least one path having the property, including pairs where S0 already chose it (S3: 18.59% already + 7.49% newly gained = 26.08%), not only new alternatives. The 7.49% is a pair-level rate; the user-mean paired change below is +7.45 pp.
- Paired change (per-user mean, selector minus S0, own target), user-level bootstrap over test users, seed 20240131, 2000 resamples, not training-seed variance: S1 +0.0001 [0.0000, 0.0002], S2 +0.0004 [0.0002, 0.0005], S3 +0.0745 [0.0724, 0.0766].
- S0 pattern mix: `enroll_user` 80.86%, concept 18.59%, teacher 0.40%, school 0.14%.
- S3 pattern mix: `enroll_user` 73.38%, concept 26.08%, teacher 0.40%, school 0.14%. Counts (64,461 pairs): S0 enroll_user 52,121 / concept 11,986 / teacher 261 / school 93; S3 enroll_user 47,301 / concept 16,811 / teacher 256 / school 93. The teacher rate rounds to 0.40% in both, but it is not unchanged: 261 to 256.
- Path transitions (S0 kind to selector kind, changed pairs only; checked from `selected_paths.jsonl`): S1 7 changed (5 enroll_user to teacher, 2 concept to teacher). S2 24 changed (17 enroll_user, 6 concept, 1 teacher, all to school). S3 4,825 changed (4,820 enroll_user to concept, 5 teacher to concept). SL 0 changed. So each selector keeps its own target, but moves paths that carried another relation.
- Pattern entropy (base 2, pattern = per-node `<relation>_<node_type>` sequence including self_loop): S0 0.745 bits (normalized 0.373, 4 patterns), S3 0.879 bits (normalized 0.439). S1 0.746, S2 0.748, SL 0.745. Normalized entropy divides by log2 of the number of observed patterns. Every selector has 4 observed patterns, so all use the same denominator log2(4) = 2 and the numbers are comparable. This would not hold if a selector had a different pattern count.
- Effective length is 3.0 for every path in every selector. No self-loops in this beam, so SL is degenerate.
- Historical-item diversity (distinct per list / list size): S0 0.6855, S3 0.6848. Shared-entity diversity: S0 0.9981, S3 0.9984.
- Selector runtime: S0 0.22 s, S1 0.61 s, S2 0.63 s, S3 1.20 s, SL 0.99 s.

Beam-wide (secondary, all candidate pairs, not topK explanation metrics): 175,538 pairs. Share of pairs with at least one path containing the relation: teacher 0.36%, school 0.16%, concept 29.3%. Share of all beam paths: teacher 0.19%, school 0.08%, concept 18.9%.

## Interpretation

- Concept is the only property with real headroom in the beam. Post-hoc selection lifts concept presence from 18.6% to 26.1% (ceiling) with zero ranking change and no pair losing the binary concept target. This is a post-hoc baseline that needs no retraining and does not change the ranking (the selector itself costs about 1.2 s). It does not preserve every property: historical-item diversity drops slightly (0.6855 to 0.6848), and S3 turns 5 teacher paths into concept paths (teacher 261 to 256). The correct statement is "the target concept presence never drops", not "all properties are kept".
- Teacher and school are an availability bottleneck: almost no beam path contains them (0.36% and 0.16% of pairs). Nothing to select, so post-hoc selection cannot move them. Reward shaping could find paths outside the beam, so this does not show shaping is useless.
- High post-hoc headroom on concept means the baseline for that property is strong. It does not show the reward has no value, because 73.92% of pairs have no concept path at all in the beam of the same pair, and selection cannot reach them.
- Effective length and self-loop rate carry no signal on this beam (all paths have length 3).

## Relation to Spike A (from review note, not re-verified in this spike)

- This spike uses 30-epoch TransE embeddings (`embeddings` sha256 above). Per the review note, Spike A used older embeddings of unclear epoch.
- Pruning depends on embeddings, so Spike A's pruned-reachability numbers should be rerun with these full embeddings before the graph-vs-beam gap is attributed to policy or search.
- Spike A's full-graph results can be reused once its KG and label hashes are confirmed to match this run's (kg, train_label, test_label hashes above; not yet compared against Spike A's own hashes). Its pruned results must be rerun with the full embeddings.

## Next steps (proposed, not started)

1. Rerun A-pruned with the full embeddings, and measure alternatives for exactly the same top-K pairs as this spike so both share one denominator.
2. Primary protocol (train 4 / eval 3): done, see the 4/3 section at the top. Decide the protocol for shaping variants from it.
3. Then a small reward-only spike E: does shaping find paths outside the beam, and does it give a benefit beyond selector S3?
4. Teacher and school are worthwhile stress-test candidates, but recheck their reachability after full-embedding pruning first. Keep S3 post-hoc as the concept baseline.

## Limits

- Verification scope. Independently rechecked from the output files and the beam pickle (not the tool's in-run asserts): all 322,305 selected rows (64,461 pairs x 5 selectors) are paths of the same (user, item) pair in the original beam, 0 violations; the S0/S1/S2/S3 transition counts above. Not independently verified: the manifest contains only the TEST_AGENT config, with no train or TransE config and no copies of the original inputs, so hashes cannot be rechecked from the manifest alone. The train horizon (3) and TransE 30 epochs rest on `effective_config.json` (saved on Drive at `tmp/mooc_full/`, TRAIN_EMBEDS epochs 30, TRAIN_AGENT epochs 50, max_path_len 3, TEST max_path_len 3, topk 25/5/1) and on the checkpoint files seen on Drive (`transe_model_sd_epoch_30.ckpt`, `tmp_policy_model_epoch_50.ckpt`). The train log was only partly read.
- One training seed. User-level bootstrap does not cover training variance.
- `id_map_files` hashes are null in the manifest (no `data/mooc/processed_files` in the spike working directory). Beam, embeddings, labels, kg, checkpoint and config hashes are present.
- Manifest `commit` is null (spike ran outside a git checkout). The pipeline ran on upstream `7348dc2`. The tool is identified by sha256 `654f8e286e3642fcf06952ef08808c4f1010da8412ad49c71f74437ee7ebcae4`.
- The pipeline `run.log` was only seen in part (Drive syncs appended logs slowly). Completion is inferred from the output files and `run_0.json`, which exist.
- Not done: list-level diversity selector (optional), push (origin is upstream `epfl-ml4ed/courserec`, not a fork), local copy of output files.

## Reproduce

Tool: `src/UPGPR/spike_b_posthoc.py`. Tests: `tests/test_spike_b_posthoc.py` (10 pass, includes comparison against the original `evaluate_paths` source extracted by AST).

```
python src/UPGPR/spike_b_posthoc.py --config tmp/mooc_full/effective_config.json \
  --path_file tmp/mooc_full/train_agent/policy_paths_epoch_50.pkl \
  --out_dir tmp/mooc_full/spike_b_full_e50 \
  --logs_json tmp/mooc_full/evaluation/UPGPR_max_len_3/run_0.json
```

Outputs on Drive `MyDrive/courserec/tmp/mooc_full/spike_b_full_e50/`: `baseline_reconstruction.json`, `selected_paths.jsonl`, `per_user.csv`, `summary.json`, `report.md`, `manifest.json`.

## Appendix: smoke run (2 epochs, feasibility only)

Beam `tmp/mooc/train_agent/policy_paths_epoch_2.pkl` (TransE 2, agent 2). Baseline NDCG 15.10, not comparable to the paper. Reconstruction matched its log exactly and all invariants held. Findings there: 89.6% of S0 paths were `enroll_user`, concept presence 10.4% rising to 19.5% under S3 (ceiling), teacher/school essentially absent (0.13% of pairs had any teacher path). Outputs in `tmp/mooc/spike_b_smoke_e2/` on Drive. The full run shows the same qualitative picture with larger absolute availability, so the smoke numbers should not be cited.
