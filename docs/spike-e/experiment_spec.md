# Spike E - reward-only concept controllability (spec fixed before the run)

Question: does a terminal concept bonus in the reward make the policy produce more concept paths at eval, and can the post-hoc selector (Spike B) exploit them? One seed, feasibility only.

## Reward
`r_shaped(t) = r_base(t) + lambda * I(t = T_train) * B(path)`, `r_base` = original `binary_train_reward` at every step (unchanged timing).
`B(path)=1` iff all hold: endpoint is an item; item in the starting learner's TRAIN history; path uses relation `item_concept`; effective length (non-self-loop edges, start tuple excluded) > 1; `binary_train_reward(path) > 0` (baseline validity gate). Else 0.
Name: terminal concept bonus conditioned on a positive train endpoint. Not potential-based; no optimal-policy preservation claim. Train labels only. Code: `kg_env.concept_bonus`, config key `TRAIN_AGENT.concept_bonus_lambda` (default 0 = original reward, tested identical step by step vs baseline commit `5334bf9`).
Note: the bonus enters earlier-step returns only through the discounted return (gamma=0.99); no reward is added before the terminal step.

## Runs (only)
| branch | lambda | epochs | seed | train/eval horizon |
|---|---|---|---|---|
| control | 0 | 5 | 23 | 4 / 3 |
| shaped | 5 | 5 | 23 | 4 / 3 |

Matched: identical fresh ActorCritic state_dict (`init_state_dict.pt`, hash in manifest), fresh Adam, seed reset (python/numpy/torch/cuda) before env+model construction, same ordering/batches (batch 32), lr schedule (linear decay with horizon = 5 epochs, as `train_agent` with epochs=5), dropout 0.5, ent 1e-3, state_history 2, hidden [512,256]. Only lambda differs. Trajectories and RNG consumption diverge once policies diverge (stated, not hidden). Baseline50 is a descriptive reference only. No sweep of lambda/target/horizon/seeds/epochs.
Eval: unchanged baseline beam search, topk [25,5,1], max_acts 250, use_pattern false, sum_prob false, no items added outside the beam, legacy path score prod(1+pi) unchanged, train-item filtering, double reverse. Epoch fixed at 5 (no selection by test). Test evaluated once per checkpoint. Init checkpoint is shared by both branches, so one init beam is evaluated.
Selector comparisons: Control+S0, Control+S3, Shaped+S0, Shaped+S3 via Spike B (S3 keeps S0 exact ordered item list; metrics invariant, asserted at 1e-12).

## Inputs
`tmp/mooc_train4_eval3/` (effective config, dataset, KG, train/validation/test labels, TransE 30-epoch embeddings sha256 `5769c715...e787`, not retrained). Audit in `preflight/input_audit.json`.

## Decision criteria
Plumbing pass: lambda0 == baseline reward, bonus at correct step and gate and actually appears, no NaN/reward hacking. Continue signal: shaped beats control on concept availability or final composition by >= 2 percentage points at a specific stage (practical threshold, not significance); if it beats only Control+S0 but not Control+S3, say so. Guardrail: relative NDCG drop > 10% vs control = trade-off flag. No signal with correct plumbing: INCONCLUSIVE/REVISE, not "unlearnable". Output: GO / NARROW / STOP / INCONCLUSIVE. User-bootstrap reflects user variation only, not training-seed variance.
