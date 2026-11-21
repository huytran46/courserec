# M5 - pattern types of displayed explanations (S0), MOOC / Xuetang

Read-only analysis of the saved M5 test beams. No training, reward, split, beam or baseline change; no old result overwritten (new files only).
Scope: M5 train 4 / eval 3, seeds 23/24/25, arms control (lambda 0) and shaped (lambda* = 5), test cohort 6,507 users, top-10 = 65,070 slots per seed. Shares are fractions of slots; mean +- sample SD (ddof=1) over 3 seeds. No inference on causes or learner benefit.

## Classification rule
Full pattern = sequence of (relation, entity type reached) over the non-self-loop edges after the start user (self-loop padding ignored).
Group = entity types of the intermediate nodes (every node after the start user except the final item; `item` nodes ignored since every path first passes through an enrolled item):
- `user` -> co_learner, `concept` -> concept, `teacher` -> teacher, `school` -> school, `category` -> category
- two or more different entity types -> **hybrid** (reported separately, never merged)
- no intermediate entity, or path not ending at an item -> other

KG check (`kg_relations`/`kg_entities` in `s0_path_types.json`): entity types are user, item, school, teacher, concept; relations enroll, item_concept, item_teacher, item_school. **There is no category entity in this KG**, so the category group is 0 by schema, not by measurement. Hybrid = 0 and other = 0 in all 12 seed/arm cells (eval horizon 3 only allows user -> item -> X -> item).
Slots: "of all slots" uses 65,070 per seed (missing slots are the remainder, reported separately); "of explained" uses the returned slots.

## Result: selector S0 (displayed explanation)
Missing slots: control 559 / 463 / 490, shaped 761 / 739 / 743 (seeds 23/24/25). Explained slots: control 64,511 / 64,607 / 64,580, shaped 64,309 / 64,331 / 64,327.

| arm | group | count 23 / 24 / 25 | % of all 65,070 slots (mean +- SD) | % of explained (mean +- SD) | % of explained per seed |
|---|---|---|---|---|---|
| control | co_learner | 56,877 / 52,180 / 54,207 | 83.64 +- 3.62 | 84.29 +- 3.71 | 88.17 / 80.77 / 83.94 |
| control | concept | 7,603 / 12,326 / 10,179 | 15.42 +- 3.63 | 15.54 +- 3.65 | 11.79 / 19.08 / 15.76 |
| control | teacher | 27 / 79 / 176 | 0.14 +- 0.12 | 0.15 +- 0.12 | 0.04 / 0.12 / 0.27 |
| control | school | 4 / 22 / 18 | 0.02 +- 0.01 | 0.02 +- 0.01 | 0.01 / 0.03 / 0.03 |
| control | category, hybrid, other | 0 / 0 / 0 | 0 | 0 | 0 |
| shaped | co_learner | 32,081 / 33,255 / 35,870 | 51.84 +- 2.98 | 52.45 +- 3.01 | 49.89 / 51.69 / 55.76 |
| shaped | concept | 30,057 / 30,420 / 27,767 | 45.20 +- 2.21 | 45.73 +- 2.24 | 46.74 / 47.29 / 43.17 |
| shaped | teacher | 657 / 450 / 487 | 0.82 +- 0.17 | 0.83 +- 0.17 | 1.02 / 0.70 / 0.76 |
| shaped | school | 1,514 / 206 / 203 | 0.99 +- 1.16 | 1.00 +- 1.18 | 2.35 / 0.32 / 0.32 |
| shaped | category, hybrid, other | 0 / 0 / 0 | 0 | 0 | 0 |

Share of all slots does not sum to 100%: the missing slots (control 0.77%, shaped 1.15% on average) are the remainder.

## Top patterns per arm (S0, pooled over seeds; each arm has only 4 distinct patterns, so a top-10 does not exist)
| arm | pattern | % of explained (pooled) | per seed 23 / 24 / 25 |
|---|---|---|---|
| control | user -enroll-> item -enroll-> user -enroll-> item | 84.29 | 88.17 / 80.77 / 83.94 |
| control | user -enroll-> item -item_concept-> concept -item_concept-> item | 15.54 | 11.79 / 19.08 / 15.76 |
| control | user -enroll-> item -item_teacher-> teacher -item_teacher-> item | 0.15 | 0.04 / 0.12 / 0.27 |
| control | user -enroll-> item -item_school-> school -item_school-> item | 0.02 | 0.01 / 0.03 / 0.03 |
| shaped | user -enroll-> item -enroll-> user -enroll-> item | 52.45 | 49.89 / 51.69 / 55.76 |
| shaped | user -enroll-> item -item_concept-> concept -item_concept-> item | 45.73 | 46.74 / 47.29 / 43.17 |
| shaped | user -enroll-> item -item_school-> school -item_school-> item | 1.00 | 2.35 / 0.32 / 0.32 |
| shaped | user -enroll-> item -item_teacher-> teacher -item_teacher-> item | 0.83 | 1.02 / 0.70 / 0.76 |

## Reference: selector S3 (same counting, secondary)
| arm | group | count 23 / 24 / 25 | % of all slots | % of explained |
|---|---|---|---|---|
| control | co_learner | 53,052 / 48,678 / 49,853 | 77.65 +- 3.48 | 78.26 +- 3.57 |
| control | concept | 11,428 / 15,829 / 14,534 | 21.41 +- 3.48 | 21.57 +- 3.49 |
| control | teacher / school | 27, 78, 175 / 4, 22, 18 | 0.14 / 0.02 | 0.14 / 0.02 |
| shaped | co_learner | 28,411 / 30,067 / 32,032 | 46.37 +- 2.79 | 46.90 +- 2.81 |
| shaped | concept | 34,148 / 33,692 / 31,682 | 50.98 +- 2.02 | 51.57 +- 2.04 |
| shaped | teacher / school | 604, 376, 414 / 1,146, 196, 199 | 0.71 / 0.79 | 0.72 / 0.80 |
Under S3 the group order of control is the same as under S0 (co_learner first); shaped S3 has concept as the largest group (51.57% of explained), shaped S0 has co_learner (52.45%).

## Consistency check vs M5 Concept@10 (passed, no discrepancy to explain)
Concept group share of all slots equals the `concept_at10` stored in each `eval_test.json` for every seed/arm/selector (max abs difference 1.1e-16): control S0 15.42% (11.68 / 18.94 / 15.64), control S3 21.41%, shaped S0 45.20%, shaped S3 50.98%, matching 15.4 / 21.4 / 45.2 / 51.0 in `REPORT.md`. The count of paths containing `item_concept` anywhere equals the concept group (hybrid = 0).

## Short answer (Q6)
Under S0 of the control arm, co_learner is above 50% of explained slots in all 3 seeds: 88.17% (seed 23), 80.77% (seed 24), 83.94% (seed 25); mean 84.29%. Not seed dependent for control. For information, shaped S0 co_learner is 49.89% / 51.69% / 55.76%: below 50% in seed 23 only, above in seeds 24 and 25.

## Files and provenance
- Code: `src/UPGPR/m5_paths.py`, `tests/test_m5_paths.py`, repo commit `7110987` (sha256 of m5_paths.py `966080a54ae10745c4a77ce768a411c00d7f1a58a92b51429305e13fd7d28d65`; identical file used on Colab from Drive `m5/code/src/UPGPR/m5_paths.py`, hash verified). M5 inputs and beams unchanged: embeddings 5769c715, KG 193776f1, test labels 6c9ac4ec.
- Outputs (Drive `courserec/m5/`): `tables/s0_path_types_counts.json` (raw counts incl. all patterns), `tables/s0_path_types.json` (shares, mean, SD, top patterns, consistency), `s0_path_types.log`.
- Reproduce (no training or beam): `python src/UPGPR/m5_paths.py counts --root m5 --tmp_dir tmp/mooc_train4_eval3 && python src/UPGPR/m5_paths.py report --root m5`.
