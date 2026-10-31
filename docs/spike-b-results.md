# Spike B: same-item post-hoc path selection

**Status: full-epoch run, train path length 3 / eval path length 3 ("3/3"). TransE 30, agent 50, default config values except `tmp_dir`, seed 23, single training run.** This is the predefined sensitivity baseline. The primary protocol (train 4 / eval 3) has NOT been run. An earlier 2-epoch smoke run is kept in the appendix as a feasibility check only.

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
2. Run the primary protocol (train 4 / eval 3) before choosing the protocol for shaping variants.
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
