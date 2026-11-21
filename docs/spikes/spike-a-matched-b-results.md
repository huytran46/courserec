---
type: spike
status: history
superseded_by: spikes/spike-a-matched-b-train4-eval3-results.md
---

# Spike A matched to Spike B full 3/3: where path availability is lost

Status: COMPLETE (Colab CPU, 861 s, exact on both cohorts, 0 censored pairs). Availability and headroom only: no claim on reward shaping, causal attribution to the policy, or learner usefulness.

Outputs: Drive `MyDrive/courserec/spike_a/mooc_T3_full_embeds_matched_B/` (`manifest.json`, `provenance_comparison.json`, `positive_pairs.csv`, `topk_pairs.csv`, `summary.json`, `witnesses.json`, `report.md`, `diagnostics.json`). The 125 MB `checkpoint.jsonl` stayed on the Colab runtime.
Tool: `src/spike_a/matched_cli.py`, `matched.py`. Tests: `tests/alt_paths/` (56 pass).

## 1. Provenance (hashes recomputed from the files)

| artifact | hash A (manifest) | hash B (manifest) | task reference | recomputed B | recomputed A | status |
|---|---|---|---|---|---|---|
| kg | 193776f1c9e3 | 193776f1c9e3 | 193776f1c9e3 | 193776f1c9e3 | 193776f1c9e3 | match |
| train_label | 2e211fd4afe5 | 2e211fd4afe5 | 2e211fd4afe5 | 2e211fd4afe5 | 2e211fd4afe5 | match |
| test_label | 6c9ac4eca92c | 6c9ac4eca92c | 6c9ac4eca92c | 6c9ac4eca92c | 6c9ac4eca92c | match |
| validation_label | db13246c13ec | n/a | n/a | db13246c13ec | db13246c13ec | match |
| embeddings | 7ae25a958a1e | 5769c715786f | 5769c715786f | 5769c715786f | 7ae25a958a1e | **mismatch (A vs B)** |
| beam | b7b5ab92c954 (2-epoch) | 7fe3509a1ad4 | 7fe3509a1ad4 | 7fe3509a1ad4 | n/a | different runs |
| effective_config | n/a | 996b9d1a2994 | 996b9d1a2994 | 996b9d1a2994 | n/a | match |

- A and B share KG, train, test and validation hashes. Old A used other embeddings (the 2-epoch smoke run), so **old A pruned numbers are not reusable**; old A full-graph numbers were recomputed and agree with the new run for all 12,860 pairs (0 differences).
- Embeddings: `transe_embed.pkl` equals the `transe_model_sd_epoch_30.ckpt` arrays exactly, and no other epoch of the 30 checkpoints matches. Config `TRAIN_EMBEDS.epochs=30`, last log line is epoch 30. So "epoch 30" rests on config, log and checkpoint equality, not the file name.
- B eval config: horizon 3, `max_acts` 250 (agent train also 3 / 250). Config equals `config/UPGPR/mooc.json` apart from `tmp_dir`.
- Audit: 0 errors. KG enrolments equal train exactly, no test or validation edge, no split overlap, node types, ID bounds and embedding rows consistent. Not audited: ID-map files (no `processed_files` on Colab); vocab sizes were checked only against KG and embedding rows. A local rebuild from raw MOOCCube produced a byte-identical KG and labels (earlier session), which covers this indirectly.
- Environment commit: the B pipeline ran on upstream `7348dc2` (stated by Spike B); the code files are unchanged against the checkout used here. Tool commit: `495c16c` on local branch `worktree-spike-a-alt-path-reachability` (not pushed; `origin` is upstream). These are different things and are recorded separately in the manifest.
- Tool identity: the 7 tool files on Colab hash identically to the committed ones. The bundle hash in the manifest (`47968192...`) also covers macOS `._*.py` junk files created by tar extraction; the tool now ignores hidden files (fixed after the run, `bundle` hash will differ from `47968192...` on re-run, per-file hashes do not).

## 2. Cohorts (separate denominators)

| | positive (held-out) | top-K (S0) |
|---|---|---|
| pairs / users | 12,860 / 6,507 (as expected) | 64,461 / 6,507 (as expected) |
| exclusions | none | 0 duplicates, 0 train items, 277 users with <10 (matches B) |
| note | test pairs not in train | only items that appear in the saved beam; 3,082 pairs (4.8%) are test positives |

## 3. Reachability (T3, all eligible pairs as denominator)

| | positive full | positive pruned | positive beam | top-K full | top-K pruned | top-K beam |
|---|---|---|---|---|---|---|
| reachable pairs | 12,860 (100%) | 12,856 (99.97%) | 3,916 (30.45%) | 64,461 | 64,461 | 64,461 |
| with concept | 98.78% | 96.30% | 8.77% | 98.61% | 95.24% | 26.08% |
| with teacher | 22.60% | 4.72% | 0.12% | 15.01% | 2.92% | 0.42% |
| with school | 76.26% | 67.85% | 0.06% | 87.28% | 77.85% | 0.18% |
| with co-learner | 99.91% | 99.71% | 27.74% | 99.99% | 99.99% | 87.53% |
| >=2 families | 99.56% | 98.16% | 6.24% | 99.96% | 98.60% | 14.20% |
| >=2 distinct paths | 100% | 99.92% | 19.04% | 100% | 100% | 47.56% |
| paths median/p90/max (reachable) | 902/2,842/16,477 | 336/1,022/7,562 | 2/8/21 | 1,039/2,574/16,477 | 374/889/7,562 | 1/5/25 |

Pruning effect: positive cohort 4 pairs lose all reachability, 3,358 lose at least one family (concept 319, teacher 2,299, school 1,081, co-learner 26), 37.2% of paths retained. Top-K cohort 0 pairs lose all reachability, 14,852 lose at least one family (concept 2,171, teacher 7,794, school 6,079, co-learner 3), 35.6% retained. Reachable-only shares are in `summary.json`.

Pattern vs family: checked, not assumed. Every observed pattern satisfies "family = intermediate node type = the single non-enroll relation = terminal relation" (concept = `item_concept`, teacher = `item_teacher`, school = `item_school`, co-learner = `enroll`), so Spike B's "path contains relation" equals these families on this schema. Effective length is 3 for every path (items and non-items form a bipartite graph), so action sequences equal paths.

## 4. Top-K funnel (denominator = all 64,461 pairs)

| target | full | pruned | beam | S0 selected | pruning loss | beam miss | selector miss |
|---|---|---|---|---|---|---|---|
| concept | 63,565 (98.61%) | 61,394 (95.24%) | 16,811 (26.08%) | 11,986 (18.59%) | 2,171 (3.37%) | 44,583 (69.16%) | 4,825 (7.49%) |
| teacher | 9,674 (15.01%) | 1,880 (2.92%) | 268 (0.42%) | 261 (0.40%) | 7,794 (12.09%) | 1,612 (2.50%) | 7 (0.011%) |
| school | 56,260 (87.28%) | 50,181 (77.85%) | 117 (0.18%) | 93 (0.14%) | 6,079 (9.43%) | 50,064 (77.67%) | 24 (0.037%) |
| co-learner | 64,457 | 64,454 | 56,425 (87.53%) | 52,121 (80.86%) | 3 | 8,029 | 4,304 |

Definitions: pruning loss = full has target and pruned does not; beam miss = pruned has it and beam does not; selector miss = beam has it and S0 did not pick it.

Check against Spike B, exact counts from its own `selected_paths.jsonl` (no rounding): concept beam ceiling 16,811, S0 11,986, S3 gain 4,825; teacher 268 / 261 / 7; school 117 / 93 / 24. All nine match. The derived values Spike B reported in chat (268, 117) are now confirmed by the file.

Consistency: every beam path and every S0 path is accepted by both the full and the pruned enumeration (0 violations), and no pruned or beam family appears outside the full enumeration.

## 5. Where availability is lost

- Concept: the graph has it for 98.6% of pairs, pruning keeps 95.2%, the beam holds it for 26.1%. The largest loss is between pruned and beam (69.2% of pairs). S0 to S3 recovers only 7.5 points of that.
- Teacher: scarce already in the graph (15.0%), and pruning removes 80.6% of those pairs (7,794 of 9,674), leaving 2.9%. The beam keeps 268 of the 1,880 pruned-available pairs (14%).
- School: common in the graph (87.3%) and mostly kept by pruning (77.9%), but almost absent from the beam (0.18%); 77.7% of pairs have a pruned school path the beam never holds.
- Old A (2-epoch embeddings) gave pruned teacher 17.4% on the positive cohort; with the full embeddings it is 4.72%. Teacher reachability under pruning depends strongly on the embeddings, which is why the rerun was needed.

Not concluded: a beam miss is a location of lost availability, not a cause. Policy, the 25/5/1 branching budget and the search all contribute and are not separated here. Pruned availability is what the environment action set allows, not what a learned policy can reach.

## 6. Verification

- Unit and end-to-end tests: 56 pass (tiny hand-computed graphs, self-loop and visited cases, brute-force fuzz against the real `_get_actions`, top-K cohort cases including duplicate selectors, items outside test positives and users with fewer than 10 predictions, funnel counts, resume).
- Real pairs vs independent brute force that calls the environment: 66 pair-modes (full and pruned), 0 mismatches, none aborted.
- Witnesses: 36 (concept, teacher, school times pruning loss, beam miss, selector miss, available in pruned, 3 each), every edge verified in the KG and every pruned or beam edge against `env._get_actions`, 0 problems. Raw-file edge check was not done on Colab (no `processed_files`).
- No censoring, no sampling: `exact_full_cohorts` true.

## 7. Reproduce

```
python src/spike_a/matched_cli.py \
  --config <B>/effective_config.json --base-config config/UPGPR/mooc.json \
  --artifacts-dir <B> --beam-paths <B>/train_agent/policy_paths_epoch_50.pkl \
  --b-dir <B>/spike_b_full_e50 --a-dir <drive>/spike_a/mooc_T3 --a-artifacts-dir <drive>/tmp/mooc \
  --out-dir <new dir> --env-source-commit 7348dc251a000f0bea593f998d4b1e69d3c24728 \
  --tool-commit <git commit of this tool> \
  --ref-hash kg=193776f1c9e3181cf8e96335d5d4c6fc18783a7a5b2761dcf68d26959c0bb97f \
  --ref-hash train_label=2e211fd4afe5dbb8a85140accca4627fa7c7824d69d0d5b47719845bc2bf006b \
  --ref-hash test_label=6c9ac4eca92cad058b8784b0e97fcb67ce7f1bb5ab7a4ad13f96013539f53a66 \
  --ref-hash embeddings=5769c715786ffbd3e635078e047284b60c7e1593b2602b5aabee90eab877e787 [--resume]
```
Needs the repo venv (Python 3.10, numpy<2, torch, scipy 1.11.4, scikit-learn 1.3.2). The run used `<B>` = `MyDrive/courserec/tmp/mooc_full`.

## 8. Limits

- Single training seed (Spike B); no variance across seeds.
- Top-K cohort depends on the baseline ranking and on the beam-only candidate set, so it is not a random sample of pairs.
- Embedding epoch 30 is established by checkpoint equality, but the early-stopping logic in `train_transe_model.py` can change the epoch used; here it matches the configured 30.
- ID-map files not audited on Colab (see section 1).
- The tool was committed locally only and ran from a tarball copy on Colab (per-file hashes verified equal).
