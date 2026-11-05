# Findings log (Spike A, B, E, M5) - MOOC / Xuetang, train 4 / eval 3

Only numbers computed from saved outputs. Units: fractions as % / pp. n=3 seeds means sample SD (ddof=1), no significance claims. Detailed sources: `docs/m5/REPORT.md`, `docs/m5/S0_PATH_TYPES.md`, `docs/m5/preflight.md`; Drive `courserec/m5/tables/`, `courserec/spike_a/`.

## 1. What the displayed explanation is (M5, selector S0, 6,507 test users, 65,070 top-10 slots per seed)
- Control (no bonus): co-learner paths are 84.29% of explained slots (88.17 / 80.77 / 83.94% for seeds 23/24/25), concept 15.54%, teacher 0.15%, school 0.02%. Co-learner is above 50% in all 3 seeds.
- Shaped (lambda 5): co-learner 52.45% (49.89 / 51.69 / 55.76), concept 45.73%, school 1.00%, teacher 0.83%.
- The KG has no category entity (entities: user, item, school, teacher, concept). Hybrid and other paths are 0 because eval horizon 3 only allows user -> item -> X -> item. Each arm has only 4 distinct patterns.

## 2. Reward-only concept bonus works on the metric it targets, without an accuracy loss in these runs
- Test concept@10 (selected concept explanations / 10, macro): control S0 15.42, control S3 21.41, shaped S0 45.20, shaped S3 50.98. NDCG@10 19.05 vs 19.39 (+0.35 +- 0.35 pp), recall and HR also not lower.
- Lambda chosen on seed-23 validation only: {0,1,5}, lambda 5 (rule: NDCG >= 0.95 x control, concept gain >= 2 pp, highest concept@10).
- Bonus is about 60% of the training return in shaped runs; base return is 2.6 / 10.5 / 7.5% lower than control. Reward domination is present but did not reduce test accuracy here.

## 3. Where the gain comes from (arithmetic decomposition, not a causal share)
- Along control+S0 -> control+S3 -> shaped+S3: selector +6.0 pp, reward-driven beam availability +29.6 pp, total +35.6 pp (about 17% / 83%). The other order gives about 16% / 84%. Interaction -0.2 +- 0.6 pp: the S3 gain (about +6 pp) does not visibly shrink under the shaped policy.
- S3 concept@10 equals beam availability on top-10 slots, so the selector can only choose a concept path where the beam has one. Shaped S0 vs control S3 is +23.8 pp: the reward moves something the selector alone does not.

## 4. Funnel: the bottleneck is the beam, not the graph or pruning (common covered cohort only)
- Spike A cache (matched, hashes of KG / train / validation / test / embeddings / effective config equal to M5) covers 54-59% of each arm's top-10 pairs and 74% of common pairs; the rest is "uncovered" (unknown), never "no path". Nothing is claimed for uncovered pairs or one-arm-only items.
- Common covered pairs (control / shaped): concept path in full graph 99.5 / 99.5%, after pruning (max_acts 250) 97.9 / 97.9%, in the M5 beam 24.3 / 52.2%, S0 selects concept 14.0 / 42.6%. Beam miss 73.6 vs 45.7%; selector miss about 10% in both.
- Consistency: 0 violations at pair level (selected concept always in beam) and at path level (every beam path and every S0/S3 selected path accepted by Spike A `path_accepted`, full and pruned; about 130-150k paths per arm).
- Earlier Spike A result worth keeping (old Spike B beam, 64,459 top-10 pairs): concept paths exist for 98.86% in the full graph, 95.60% after pruning, 13.08% in the beam. Teacher paths are cut hard by pruning (16.29% full -> 2.70% pruned -> 0.13% beam); school is 86.9% full, 77.5% pruned, 0.01% in beam.

## 5. Shaped lists are slightly shorter
- Missing slots (65,070 per seed): shaped 761 / 739 / 743, control 559 / 463 / 490. Users with a list shorter than 10: shaped 304 / 303 / 305, control 247 / 190 / 204; no user has an empty list. Slots and users are different units (asserted: missing slots = sum of 10 - n_items over short users).
- 49.0% of shaped S3 slots have no concept explanation: 1.15% missing slots + 47.87% returned non-concept slots (control 78.6% = 0.77% + 77.82%). "No concept path in the beam" applies to returned slots only.
- Candidate recall of test positives 32.4 -> 31.5%. Common (user, item) pairs between arms are only about 40-45% of each arm's beam pairs; shaped changes which courses reach the beam, so the all-output availability gain (+29.6 pp) includes item mix (common pairs: 24.3 -> 52.2%, +27.9 pp). The 1.7 pp difference is not attributed.

## 6. Unexplained observations (not investigated)
- Shaped policy shows school and teacher paths without any reward for them (school 0.99 +- 1.16%, driven by seed 23: 1,514 slots vs 206 / 203). Cause unknown.
- Control seed-to-seed spread of the concept share is large (11.8 / 19.1 / 15.8% of explained under S0) versus shaped (46.7 / 47.3 / 43.2%).
- Training paths are 98-99% action self-loop padding at train horizon 4 (artifact; eval paths at horizon 3 have none).

## 7. Limits to keep with any claim
- n=3 seeds, one lambda grid {0,1,5}, lambda selected on seed 23 validation only, no user bootstrap.
- Embeddings (TransE, sha 5769c715) shared by all seeds: SDs exclude embedding-training variance.
- The Spike E pilot test results were seen before M5 (selection itself used validation only; test evaluated once per final policy).
- New-contract metrics (P@10 with /10, missing slots = 0) are not comparable to legacy numbers such as 19.147% NDCG.
- Funnel only covers pairs in the Spike A cache; uncovered pairs not enumerated by choice.
- Nothing here speaks to trust, usefulness or learning benefit for learners.

## 8. Process notes
- Spike A manifest row "embeddings mismatch" compares old Spike A input (7ae25a95) with B, not the matched train4_eval3 run; its pruned stage is policy independent and was reused, its beam/S0 columns were not (recomputed from M5 beams).
- Funnel, slot reconciliation, path validity and S0 path types are reproducible by CLI (`src/UPGPR/m5_funnel_run.py`, `src/UPGPR/m5_paths.py`; commands in `docs/m5/REPORT.md` and `docs/m5/S0_PATH_TYPES.md`). Code for funnel and path types also lives on Drive `courserec/m5/code/`.
- Colab MCP was unreliable (hangs, disconnects); results written to Drive were verified by file contents, not by cell output.

## 9. Open points
- Why shaped lists are shorter and why school / teacher paths appear.
- Whether the S3 gain (~6 pp) holds with more seeds or different embeddings.
- Funnel for uncovered pairs and for one-arm-only items (would need running Spike A's tool on them).
