# Spike E report - reward-only concept controllability

Status: **DONE (feasibility, one seed)**. Recommendation: **GO** (verify before building on it). Scope: seed 23, 5 epochs per branch, train4/eval3, CPU Colab. Says nothing about learner usefulness or reproduction of the paper. Spec: `experiment_spec.md`. Artifacts (beams, checkpoints, per-epoch logs, `summary.json`, `same_pair_comparison.csv`, `manifest.json` with all hashes, Spike B outputs per branch): Drive `courserec/tmp/spike_e_mooc_t4e3/`. This report and the spec live in the repo at `docs/spike-e/` (not copied to Drive).

## Plumbing: PASS
- lambda=0 reward equals the baseline commit `5334bf9` step by step (random walks), and the driver at lambda=0 reproduces `train_agent.train` weights exactly. 12 Spike E tests pass (hand-computed rewards, timing, gates, self-loop padding, no changes to embeddings/pruning/search); Spike B tests still pass (22 total locally).
- Real-data support: 60 sampled users all have a 3-effective-hop concept path padded with an action self-loop to 4 actions that gets B=1 (107,643 such paths; witness in `preflight/witness.json`: u0, enroll item0, item_concept c12, item_concept item3, self_loop, rewards [0,0,1,6]).
- Training: bonus never before the terminal step (0 violations), 0 non-finite losses, identical init hash for both branches (`89289bc9...55f2`). Eval vs logs abs diff 0.0, selector invariants (order + metrics) within 1e-12 for all branches.
- Inputs: embeddings hash `5769c715...e787` matches A/B; KG and train/validation/test labels are byte-identical to `mooc_full`.

## Training (epoch 5, training trajectories, not eval)
| | control (l=0) | shaped (l=5) |
|---|---|---|
| base return / episode | 0.421 | 0.296 |
| raw bonus / episode (B) | 0.012 | 0.099 |
| bonus x lambda | 0 | 0.497 |
| total return | 0.421 | 0.793 |
| terminal B=1 | 1.2% | 9.9% |
| positive endpoint | 20.7% | 10.3% |
| any concept in path | 11.9% | 64.2% |
| mean effective length | 3.04 | 3.45 |
| paths with an action self-loop | 93.7% | 53.8% |

B=1 fraction in shaped by epoch: 0.23, 1.8, 7.8, 10.1, 9.9%. Flag: in training the bonus is 63% of the shaped return and base return is 30% below control, with the positive-endpoint rate halved. Test NDCG did not drop (below), but this is the reward-domination pattern to watch.

## Evaluation (test, all users, % ; epoch fixed at 5, evaluated once)
| | NDCG | Recall | HR | Precision | users <10 preds |
|---|---|---|---|---|---|
| init (shared, untrained) | 19.04 | 26.63 | 42.92 | 5.08 | 0 |
| control | 18.56 | 25.04 | 41.59 | 4.85 | 234 |
| shaped | 19.58 | 26.64 | 44.00 | 5.12 | 164 |

NDCG shaped vs control +5.5% relative (guardrail of -10% not hit). One seed: no claim this is a real accuracy gain. Baseline50 S0 NDCG 19.147 is a descriptive reference only.

## Four comparisons (Spike B, own top-K pairs; denominators ~64.5k pairs, 6507 users)
| | concept in selected explanation | S3 gain over S0 | beam availability of concept (= S3 ceiling) |
|---|---|---|---|
| Control + S0 | 10.60% | | |
| Control + S3 | 15.83% | +5.23 pp | 15.83% |
| Shaped + S0 | 43.55% | | |
| Shaped + S3 | 48.52% | +4.98 pp | 48.52% |

Shaped beats Control+S0 by +32.9 pp, Control+S3 by +32.7 pp, and Shaped+S0 beats Control+S3 by +27.7 pp. The ceiling rises +32.7 pp. All far above the 2 pp practical threshold (not statistical significance). S3 still adds about 5 pp on top of the shaped policy.
Own-top-K differences mix a changed item set with changed paths, so the same-pair table below is the cleaner path evidence.

## Same-pair analysis (`same_pair_comparison.csv`; pair-weighted, user-mean in CSV)
"Outside the beam" here means outside the saved control beam only, not outside the KG or pruning.
| cohort | pairs | control no concept, shaped has | control has, shaped lost | both have | both lack |
|---|---|---|---|---|---|
| common candidate pairs | 69,220 | 23,174 (33.5%) | 2,322 (3.4%) | 14,574 (21.1%) | 29,150 (42.1%) |
| top-K intersection | 34,620 | 12,536 (36.2%) | 955 (2.8%) | 5,842 (16.9%) | 15,287 (44.2%) |
| top-K union | 94,619 | 14,471 (15.3%) | 1,167 (1.2%) | 7,034 (7.4%) | 17,964 (19.0%) |

Top-K union also has 27,992 pairs (29.6%) absent from the control beam and 25,991 (27.5%) absent from the shaped beam: item/beam availability missing, kept separate from concept-path discovery for the same course. So the shaped policy finds new concept paths for courses control already reached, and also changes which courses reach the beam.

## Candidates and overlap (12,860 test-positive pairs, 6507 users)
- Candidate recall (pair-weighted / user-mean): init 37.6/36.9%, control 30.4/30.4%, shaped 32.2/32.3%. Five epochs of training shrinks the beam relative to init (mean candidates/user 39.1 -> 22.4 control, 24.3 shaped).
- Concept available in all candidate pairs of the beam: init 24.6%, control 18.7%, shaped 48.5%.
- Control vs shaped, user-mean: candidate-set Jaccard 0.29, top-K Jaccard 0.38, top-K overlap fraction 0.54.

## Composition and side effects (selected explanations, S0)
- Pattern entropy (same 4-pattern reference set): control 0.51 bits, shaped 1.35 bits. Co-enrollment pattern 89.2% -> 49.0%.
- School presence 0.07% -> 6.4%, teacher 0.09% -> 1.0%. Not rewarded; side effect of leaving the co-enrollment pattern. Worth a look before treating as benign.
- Selected explanations at eval have effective length 3.0 and no self-loops in both branches (eval horizon 3). The action self-loops are a training-horizon-4 artifact (control already pads 94%).

## Runtime (CPU, 2 cores, 3 beams concurrent so beam times are inflated)
Train 599 s control / 572 s shaped; beam 1458 s / 1505 s (init 1332 s); baseline evaluator 9 s; selector run 47 s / 33 s.

## Decision: GO (one seed)
Plumbing passes, bonus appears and is learned, concept availability and final concept presence rise by a wide margin at the beam, selector and same-pair stages, and test NDCG did not fall. Shaped beats Control+S3, so the reward moves something the selector cannot.

## Limits and caveats
- One seed, 5 epochs, lr decayed over those 5 epochs. The control is itself immature (94% self-loop padding, 234 users with <10 predictions); the init beam is stronger than both on candidate recall. The comparison is shaped vs a short control, not vs baseline50.
- User-bootstrap CIs and validation tracking were not computed. No training-seed variance, so the NDCG gap is not evidence of an accuracy gain.
- Training-time reward domination pattern (bonus 63% of return, base return -30%) not yet checked at other lambda; per spec no sweep was run.
- Side effects (school/teacher, shorter beam vs init) are unexplained.
- Provenance: `kg_env.py` on Colab was LF-converted by the patch step (AST identical to commit `3acd0f4`). Colab ran Python 3.10 / torch 2.14.1 on CPU; tests ran locally (Python 3.13). Spike E code lives in local branch `worktree-spike-b-posthoc-paths`, not pushed.

## Reproduce (repo root, Colab venv, `tmp/` linked to Drive)
```
python src/UPGPR/spike_e_witness.py --config tmp/mooc_train4_eval3/effective_config.json --out tmp/spike_e_mooc_t4e3/preflight --ref tmp/mooc_full --max_users 60
python src/UPGPR/spike_e_train.py make-init --config $C --out $E/init_state_dict.pt
python src/UPGPR/spike_e_train.py train --config $C --branch control --lam 0 --init $E/init_state_dict.pt --out_dir $E/control --epochs 5
python src/UPGPR/spike_e_train.py train --config $C --branch shaped --lam 5 --init $E/init_state_dict.pt --out_dir $E/shaped --epochs 5
for n in init control shaped:  spike_e_train.py beam --config $C --ckpt <ckpt> --path_file $E/eval_$n/beam.pkl ; spike_e_eval.py --config $C --path_file $E/eval_$n/beam.pkl --out_dir $E/eval_$n --epochs 5
python src/UPGPR/spike_e_analyze.py --config $C --root $E
pytest tests/test_spike_e_reward.py tests/test_spike_b_posthoc.py
```
`C=tmp/mooc_train4_eval3/effective_config.json`, `E=tmp/spike_e_mooc_t4e3`.
