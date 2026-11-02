# M5 report - reward-only concept shaping vs same-item selector S3

**Status: COMPLETE, funnel covers the Spike A cache cohort only.** The full matrix ran (lambda selection locked on validation, 3 paired seeds x 2 arms x 50 epochs, raw beams, per-user outputs, S0/S3 invariants). The funnel (full graph -> after pruning -> beam -> S0/S3) was added afterwards from the Spike A matched cache (see "Funnel"): full/pruned stages are known for 54-59% of each arm's top-10 pairs (74% of common pairs); the rest are `uncovered` (unknown, not "no path"). Beam/S0/S3 are known for all pairs. Units: fractions in [0,1] in files, shown as % / pp here. n=3 seeds: SD is sample SD (ddof=1), no significance or non-inferiority claim is made.

## Question and scope
Does a terminal concept bonus (reward-only) raise how often explanations pass through a concept, and by how much beyond re-selecting the path for the same item (S3), under matched embeddings, pruning and beam budget? Nothing here says anything about trust, usefulness or learning benefit. Novelty over P006/P007 is not claimed.

## Design actually run (evidence)
- Train horizon 4 / eval 3, 50 RL epochs, LR schedule defined for 50 epochs from epoch 1 (no resume of the 5-epoch pilot). Two arms per seed: lambda=0 (control) and lambda*=5 (shaped). Seeds 23, 24, 25.
- Lambda selection (validation, seed 23, epoch 50, locked in `selection.json` before any test beam): control S3 concept@10 17.4%, NDCG 16.1%; lambda=1: 49.3% / 16.8% (ratio 1.04); lambda=5: 52.7% / 16.6% (ratio 1.03). Both passed (NDCG >= 0.95 x control, gain >= 2 pp); lambda=5 chosen on higher concept@10 (gap to lambda=1 3.45 pp > 0.1 pp tie).
- Seed 23 test reuses the validation-phase checkpoints (7 full RL runs total: 3 dev + 4 new). Seeds 24/25 use fresh policy inits; **embeddings (TransE, sha 5769c715...e787) are shared by all seeds**, so embedding-training variance is not in the SDs.
- Inputs match the spec hashes (KG, train/validation/test labels, embeddings). Embeddings were hashed before/after every run (unchanged). Reward: base reward at every step, bonus lambda*B only at the terminal step; 0 early-bonus violations and 0 non-finite losses in all runs.
- Selectors S0/S3 run on the same raw beam; asserted equal item lists, order and metrics per user (S3 only changes the explanation path).

## Main result: 2 arms x 2 selectors (test, 6507 eligible users, macro, mean +- SD over 3 seeds)
| | NDCG@10 | Recall@10 | HR@10 | concept@10 |
|---|---|---|---|---|
| Control + S0 | 19.05 +- 0.25 | 25.86 +- 0.34 | 42.91 +- 0.44 | 15.42 +- 3.63 |
| Control + S3 | 19.05 | 25.86 | 42.91 | 21.41 +- 3.48 |
| Shaped + S0 | 19.39 +- 0.28 | 26.24 +- 0.51 | 43.42 +- 0.75 | 45.20 +- 2.21 |
| Shaped + S3 | 19.39 | 26.24 | 43.42 | 50.98 +- 2.02 |

concept@10 = selected concept explanations / 10, missing slots count 0. Per-seed values are in `tables/arms_x_selectors.csv` and `summary.json`.

Paired differences (per seed 23/24/25, then mean +- SD):
- Shaped - control, concept@10 under S3: +34.9, +27.5, +26.4 pp -> +29.6 +- 4.7 pp. Under S0: +34.5, +27.8, +27.0 -> +29.8 +- 4.1 pp.
- S3 - S0, control: +5.9, +5.4, +6.7 -> +6.0 +- 0.7 pp. S3 - S0, shaped: +6.3, +5.0, +6.0 -> +5.8 +- 0.7 pp.
- Interaction ((shaped S3 - S0) - (control S3 - S0)): +0.4, -0.4, -0.7 -> -0.2 +- 0.6 pp. S3's gain does not visibly shrink under the shaped policy; with n=3 this is not evidence of equality.
- Shaped S0 vs control S3: +23.8 pp, i.e. the reward moves something the selector alone does not.
- NDCG@10 shaped - control: +0.15, +0.14, +0.75 pp -> +0.35 +- 0.35 pp. Point estimates, no significance claim; the pre-set 5% loss threshold was not approached.
- Concept availability in the beam (pairs with >=1 concept path / pairs with a path, pooled): 24.3 +- 3.9% -> 56.8 +- 1.6% (+32.6 +- 4.9 pp). This is the S3 ceiling mechanism: S3 can only pick a concept path where the beam has one.
- Candidate recall of test positives: 32.4 +- 0.7% -> 31.5 +- 1.1%. Users with <10 returned items: control 247/190/204, shaped 304/303/305 (shaped returns shorter lists for roughly 100 more users).
- Valid explanations per 10 slots: 0.99 in both arms.

## Funnel (Spike A cache, exact M5 cohorts) and where the gain comes from
**Cache compatibility** (`spike_a/mooc_T3_full_embeds_matched_B_train4_eval3`): KG 193776f1, train 2e211fd4, validation db13246c, test 6c9ac4ec, embeddings 5769c715 all equal M5 inputs (recomputed on Drive); effective config sha bb18f328 same; eval horizon 3, max_acts 250, topk [25,5,1]; concept = path with an `item_concept` relation, same as `m5_metrics.has_concept` (A checked schema equivalence). A tool files (core, matched, artifacts) have the sha256 recorded in its manifest. The A manifest row "embeddings mismatch" compares the OLD Spike A input (7ae25a95) with B, not this matched run. A's pruned stage depends only on KG, embeddings, horizon, max_acts, so it is policy independent and valid for M5 policies. Not reusable: A's beam/S0 columns (they describe the old Spike B beam); M5 beam/S0/S3 are recomputed from the M5 raw beams.
**Coverage:** the cache holds 74,072 pairs (64,459 old top-10 pairs + 12,860 test positives, union). Covered share of M5 top-10 pairs: control 58.1 / 58.8 / 58.2%, shaped 53.7 / 54.0 / 54.3%; common pairs 74.0 / 74.3 / 73.5%; only-one-arm pairs 31-42%. Uncovered pairs have unknown full/pruned status (cache does not cover them, no claim of no path). Among covered pairs `no path in graph` = 0.
**Funnel on common covered pairs (same items in both arms; 25,094 / 25,253 / 25,840 per seed; share of pairs, mean over 3 seeds):**
| stage | control | shaped |
|---|---|---|
| concept path in full graph | 99.5% | 99.5% |
| after pruning (250) | 97.9% | 97.9% |
| in M5 beam (= S3 selects concept) | 24.3% | 52.2% |
| S0 selected concept | 14.0% | 42.6% |
Loss buckets (S0, share of common covered): no concept path in graph 0.5%; pruning loss 1.6%; beam miss 73.6% (control) vs 45.7% (shaped); selector miss (concept in beam, S0 not) 10.3% vs 9.6%. Graph and pruning are identical across arms by construction; the whole arm difference sits in the beam stage. Per-seed values and counts: `tables/funnel.json` (all / common / only-one-arm cohorts, per arm).
**Only-one-arm pairs (covered part), concept in beam:** control-only 13.0 / 17.6 / 16.7%, shaped-only 49.6 / 52.3 / 46.5%. Items unique to the shaped arm are mostly concept-reachable in the beam, items unique to control are not, so part of the all-pair gain is item mix; on common pairs the beam availability gain is +27.9 pp (covered) vs +29.6 pp on all top-10 slots.
**Consistency checks (0 violations everywhere):**
- Pair level (3 seeds x 2 arms, all top-10 pairs): selected S0 concept in beam 0 violations; S3 concept == beam concept 0 violations; beam concept pair with pruned_concept=0 or full_concept=0: 0 (covered pairs).
- Path level, using A's `path_accepted` (full and pruned `PrunedActions` from the real `_get_actions`) on every beam path of every top-10 pair, deduplicated per user (141k / 148k, 136k / 149k, 144k / 141k paths for control / shaped seeds 23/24/25) and on all S0/S3 selected paths (about 129k per arm): 0 rejected by full, 0 rejected by pruned, 0 selected paths absent from the pair's beam candidates. File `tables/funnel_path_validity.json` (also `docs/m5/funnel_path_validity.json`).
- Slot-level S0/S3 concept@10 recomputed from the funnel equal the M5 table values to <1e-9 (assert in the run).
**Attribution of concept@10 gain (test, macro, 3 seeds; S3 concept@10 equals beam availability on top-10 slots, so S3 minus S0 is exactly the selector contribution):**
| component | pp, mean +- SD | per seed 23/24/25 |
|---|---|---|
| Beam availability from reward (shaped - control, S3) | +29.6 +- 4.7 | +34.9 / +27.5 / +26.4 |
| Selector at control (S3 - S0, control) | +6.0 +- 0.7 | +5.9 / +5.4 / +6.7 |
| Total shaped+S3 vs control+S0 | +35.6 | +40.8 / +32.9 / +33.1 |
| Selector slack remaining under shaped (S3 - S0) | 5.8 +- 0.7 | 6.3 / 5.0 / 6.0 |
| Interaction (shaped slack - control slack) | -0.2 +- 0.6 | +0.4 / -0.4 / -0.7 |
Reading: about 83% of the total gain is beam availability (reward moves concept paths into the beam, mostly where control's beam misses them; graph and pruning are not the bottleneck), about 17% is the selector (re-picking the path for the same item, independent of reward, same size under both policies). The shaped S0 vs control S0 gain (+29.8 pp) equals availability gain minus the interaction, i.e. the shaped policy also ranks its concept paths first, but the S3 ceiling is the beam. The selector cannot create availability: even under shaped, 49.0% of top-10 slots (100 - 51.0%) have no concept path in the beam although the pruned graph has one for ~98% of covered pairs. n=3, no bootstrap, shared embeddings (limits below).

## Matched cohorts (common (user,item) pairs; conditional on post-treatment beams, not a causal effect on the catalogue)
Common pairs per seed (23/24/25): 65,656 / 68,134 / 71,533 (each arm has about 145k-165k beam pairs, so only ~40-45% of pairs are common). Among common pairs, shaped vs control concept availability:
- new (control none, shaped has): 38.6% / 28.8% / 30.8%
- lost (control has, shaped none): 4.2% / 6.3% / 6.1%
- both have: 18.9% / 27.3% / 22.0%; neither: 38.3% / 37.6% / 41.2%
Pairs present in only one arm are reported separately: only-control 90.6k / 95.8k / 83.9k (concept rate 17.6 / 23.1 / 23.1%), only-shaped 77.5k / 76.5k / 85.1k (58.6 / 58.3 / 57.0%). So the shaped policy both finds concept paths for courses control already reached and changes which courses reach the beam. Top-10 intersection (33.9k / 34.0k / 35.2k pairs): new 38.7 / 29.3 / 30.8%, lost 4.1 / 6.2 / 6.3%. File: `tables/common_pair_cohorts.json`.

## Relation composition (selected paths, S0; shares of effective-path families)
Concept share control 11.8 / 19.1 / 15.8% -> shaped 46.7 / 47.3 / 43.2%; co-learner 88.2 / 80.8 / 83.9% -> 49.9 / 51.7 / 55.8%; teacher 0-0.3% -> 0.7-1.0%; school ~0% -> 0.3-2.4%. Normalized family entropy (pooled, M=4): 0.27 / 0.36 / 0.33 -> 0.60 / 0.54 / 0.54. School/teacher rise without being rewarded; cause not investigated.

## Training diagnostics (epoch 50, training trajectories; eval-3 loop counts are not used here)
- Bonus share of total return, shaped: 58.1 / 61.0 / 60.5%. Base return shaped vs control: 0.456 vs 0.468, 0.425 vs 0.475, 0.417 vs 0.451 (-2.6% / -10.5% / -7.5%). Positive-endpoint rate shaped vs control: 23.4 vs 23.4, 21.8 vs 23.8, 20.9 vs 22.4%. B=1 rate 12.7 / 13.3 / 12.8% (control 1.0 / 3.0 / 1.4%). Early in training the shaped base return was much lower and recovered; the reward-domination pattern is present (bonus ~60% of return) but did not reduce test accuracy here.
- Action self-loop padding is 98-99% of training paths in both arms: a training-horizon-4 artifact (eval paths have no self-loops at horizon 3).
- Runtime: Colab CPU 2 cores, 3-4 runs concurrent; train 9.6-14.4 ks per run; test beam about 60 min for 6 concurrent beams.

## Limits and caveats
- n=3 seeds, one lambda grid {0,1,5}, selection on seed 23 validation only; embeddings shared across seeds; no user-bootstrap. SDs do not capture embedding-training variance.
- The Spike E pilot test results were already seen before M5; the selection itself used validation only, test was evaluated once per final policy.
- New-contract metrics (P@10 with /10 on short lists, etc.) are not comparable with legacy numbers (e.g. 19.147% NDCG) to claim gain.
- **Funnel coverage.** Full/pruned stages only for pairs in the A cache (54-59% of top-10 pairs); uncovered pairs are unknown, not "no path", and the covered subset is biased toward items the old Spike B beam also ranked top-10 (the 74% common-pair cohort is the primary comparison). Not run: A's tool on uncovered pairs (not required). No bootstrap; embeddings shared across seeds (SDs exclude embedding variance); both stay as stated limits, no extra runs added.
- Colab runtime recycled once after Phase F; all outputs were on Drive. Python 3.13 / torch 2.11 CPU on Colab; tests ran locally (37 pass).

## Artifact index (Drive `courserec/m5/`, code in repo `486088c`, `b498a7e`, `m5/code/`)
- `selection.json` (locked, read-only), `run_plan.json`, `manifests/`, `effective_config.json`, `inits/seed{23,24,25}.pt`
- `development/seed23/lambda{0,1,5}/` and `final/seed{24,25}/lambda{0,5}/`: `ckpt/` (every 5 epochs + 50 + `state_latest.pt`), `train_log.json`, `train_meta.json`, `checkpoint_hashes.json`, `beam_validation/` (dev only), `beam_test/beam.pkl`, `eval_test.json`, `per_user_test_{S0,S3}.csv`
- `summary.json`, `tables/arms_x_selectors.csv`, `tables/common_pair_cohorts.json`, `tables/funnel.json`, `tables/funnel_path_validity.json`, `phaseD.log`, `phaseF.log`
- Funnel code: `m5_funnel.py` (+ `tests/test_m5_funnel.py`); path validity used Spike A's `core.py`/`matched.py` (`path_accepted`). The Drive `m5/code/src/UPGPR/m5_funnel.py` is the first version (no `summarize`); repo has the final one.
- Code: `m5_metrics.py`, `m5_run.py`, `spike_e_train.py`, `kg_env.py` (bonus), tests `tests/test_m5_metrics.py`, `tests/test_m5_run.py`, `tests/test_spike_e_reward.py`

## Reproduce
```
python src/UPGPR/m5_run.py D --root m5 --config m5/effective_config.json --spike_e_init tmp/spike_e_mooc_t4e3/init_state_dict.pt
python src/UPGPR/m5_run.py F --root m5 --config m5/effective_config.json
python src/UPGPR/m5_run.py aggregate --root m5
pytest tests -q
```

## M6 handoff
Open points: why shaped lists are shorter for ~100 more users and why school/teacher paths appear; whether the S3 gain staying ~6 pp under shaping holds with more seeds or varied embeddings.
