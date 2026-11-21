---
type: spike
status: history
---

# M5 preflight (before any full run)

## Code mapping (verified by reading code)
loaders (`utils.load_labels`, `load_embed`) -> embedding lookup (`BatchKGEnvironment.embeds`, numpy, outside optimizer) -> action pruning (`_get_actions`: top `max_acts=250` by embedding score, visited-node filter, self-loop action kept) -> reward (`binary_train_reward`, every step; M5 bonus `concept_bonus` only at terminal step via `_batch_get_reward_shaped`) -> beam (`test_agent.batch_beam_search`, topk [25,5,1], unchanged) -> item rank (`dot(user+enroll,item)` then legacy path prob, tie by item id; `m5_metrics.rank_key`) -> S0/S3 (`spike_b_posthoc.select_path`, same item list) -> metrics (`m5_metrics`, fractions in [0,1]).

## Code provenance
Colab code = upstream `7348dc251a000f0bea593f998d4b1e69d3c24728` + Spike E/M5 patch. Local branch `worktree-spike-b-posthoc-paths` (not pushed). AST hashes of kg_env, train_agent, spike_b_posthoc, m5_metrics, spike_e_train are identical between local and Colab (spike_e_train: identical after dropping the module docstring). `kg_env.py` diff vs upstream: constructor arg, `concept_bonus`, `_batch_get_reward_shaped`, one call in `batch_step`. `train_agent.py`: one pass-through line.

## Inputs (all sha256 equal to the spec table, in both `tmp/mooc_train4_eval3` and `tmp/mooc_full`)
kg 193776f1..., train 2e211fd4..., validation db13246c..., test 6c9ac4ec..., embeddings 5769c715...e787. Counts from loader: 6,507 eligible test users, 12,860 test positives.

## Config vs spec (effective_config.json, train max_path_len 4 / eval 3)
batch 32, lr 0.001, gamma 0.99, ent 0.001, dropout 0.5, state_history 2, hidden [512,256], binary_train, use_pattern false, sum_prob false, max_acts 250, topk [25,5,1]: all equal to spec. LR decay horizon = 50 epochs from epoch 1 (`--epochs 50`). Pilot 5-epoch policies are not resumed.

## Checks done
- Local: 37 tests pass (reward truth table/timing, lambda=0 == baseline, metric edge cases: empty, short, duplicate, ties, train exclusion; S3 keeps item list; lambda selection rule incl. no-feasible case).
- Real data: `m5_metrics.evaluate_beam` on the saved Spike E control test beam reproduces Spike E: S0 concept 10.60%, S3 15.83% over 64,536 pairs; candidate recall 30.4%; concept availability 18.7%; co-learner share 89.2%; NDCG 18.56% (legacy denominators coincide here). New-contract Precision@10 is 4.79% vs legacy 4.85% (short lists use /10). Do not compare new numbers with legacy 19.147% for gains.
- Init: seed-23 init reused from Spike E (state-dict sha256 `89289bc9...55f2`, verified). Seeds 24/25 get fresh inits. Embeddings frozen by construction (numpy, not in optimizer); the driver hashes them before/after each run and asserts equality.
- Smoke: 1 epoch, 640 users, lambda 5: 5.9 s/epoch, checkpoint 2.0 MB, resume state 6 MB.

## Deviations / decisions (stated, not hidden)
- Embeddings: seed-23 embeddings reused for seeds 24/25 (spec allows; no TransE retraining). Seeds therefore vary policy init, sampling and beam RNG only, not embeddings.
- Runtime differs from Spike E: Colab CPU 2 cores, Python 3.13, torch 2.11 CPU (Spike E: 3.10). Runs in M5 all use this one runtime.
- Checkpoints saved every 5 epochs plus epoch 50, `state_latest.pt` every epoch for resume.
- Validation beams use the unchanged baseline beam search with `data="validation"`.
- Spec pilot test results (Spike E) were already seen; they are listed as a limitation, not hidden.

## Time/disk estimate (from smoke and Spike E, not measured at full scale)
Train about 60-120 s/epoch -> 1-2 h per run at 3-4 concurrent runs; beams 10-25 min each. Disk < 200 MB. Total about 4-6 h wall.
