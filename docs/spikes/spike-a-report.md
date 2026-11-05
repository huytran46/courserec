# Spike A report: alternative-path reachability (MOOCCube / Xuetang, T3)

Status: DONE. Availability and headroom only. No claim on reward shaping, causal attribution to the policy, or learner usefulness. Counts do not measure controllability or learnability (that is spike E).

Detail files: `docs/spikes/spike-a-matched-b-results.md` (full tables and provenance), `docs/spikes/spike-a-alt-path-reachability.md` (how to run). Raw outputs on Drive: `MyDrive/courserec/spike_a/mooc_T3_full_embeds_matched_B/` (final) and `.../mooc_T3/` (stage 1). Code: `src/spike_a/`, tests `tests/alt_paths/` (56 pass), local branch `worktree-spike-a-alt-path-reachability` at `3ae7c08` (not pushed; `origin` is upstream).

## Question

For each test (learner, positive course) pair, and for the baseline top-10 recommendation pairs of Spike B, how many distinct paths exist at horizon 3, across how many relation types, and where is that availability lost: in the graph, after environment action pruning, in the saved beam, or at the baseline path selector (S0)?

## Two stages, and which numbers to use

- **Stage 1 (superseded for pruned numbers).** Run on the only artifacts then identified (`tmp/mooc`, 2-epoch embeddings, 2-epoch beam). Full-graph results stay valid (same KG and labels as Spike B, reproduced exactly in stage 2). Its pruned and beam numbers used other embeddings and must not be cited.
- **Stage 2 (final).** Same KG and labels, embeddings of Spike B's full 3/3 run (sha256 `5769c715...`, equal to the epoch-30 TransE checkpoint), beam `policy_paths_epoch_50.pkl`, `max_acts` 250. Two separate cohorts: held-out positives (12,860 pairs) and Spike B's S0 top-K (64,461 pairs).

## Provenance (stage 2)

KG, train, test, validation and effective-config hashes recomputed from the files and equal to Spike B's manifest and to the reference hashes you gave. Embeddings differ from stage 1 (`7ae25a95...` vs `5769c715...`). Audit: 0 errors (KG enrolments equal train exactly, no test or validation leakage, ID bounds, node types and embedding rows consistent). Pipeline source commit `7348dc2` (stated by Spike B, code files unchanged against the checkout); tool commit `495c16c` for the run. Not audited on Colab: ID-map files (no `processed_files`); a local rebuild from raw data gave a byte-identical KG and labels.

## Results (stage 2, exact, 0 censored)

**Reachability, all eligible pairs as denominator**

| | positive full | positive pruned | positive beam | top-K full | top-K pruned | top-K beam |
|---|---|---|---|---|---|---|
| reachable pairs | 100% | 99.97% | 30.45% | 100% | 100% | 100% |
| with concept | 98.78% | 96.30% | 8.77% | 98.61% | 95.24% | 26.08% |
| with teacher | 22.60% | 4.72% | 0.12% | 15.01% | 2.92% | 0.42% |
| with school | 76.26% | 67.85% | 0.06% | 87.28% | 77.85% | 0.18% |
| with co-learner | 99.91% | 99.71% | 27.74% | 99.99% | 99.99% | 87.53% |
| >=2 families | 99.56% | 98.16% | 6.24% | 99.96% | 98.60% | 14.20% |
| paths per reachable pair (median / p90 / max) | 902 / 2,842 / 16,477 | 336 / 1,022 / 7,562 | 2 / 8 / 21 | 1,039 / 2,574 / 16,477 | 374 / 889 / 7,562 | 1 / 5 / 25 |

Pruning keeps about 37% (positive) and 36% (top-K) of paths.

**Top-K funnel, one denominator (64,461 pairs)**

| target | full | pruned | beam | S0 selected | pruning loss | beam miss | selector miss |
|---|---|---|---|---|---|---|---|
| concept | 63,565 (98.61%) | 61,394 (95.24%) | 16,811 (26.08%) | 11,986 (18.59%) | 2,171 (3.37%) | 44,583 (69.16%) | 4,825 (7.49%) |
| teacher | 9,674 (15.01%) | 1,880 (2.92%) | 268 (0.42%) | 261 (0.40%) | 7,794 (12.09%) | 1,612 (2.50%) | 7 |
| school | 56,260 (87.28%) | 50,181 (77.85%) | 117 (0.18%) | 93 (0.14%) | 6,079 (9.43%) | 50,064 (77.67%) | 24 |

All nine reference counts from Spike B's own `selected_paths.jsonl` (concept 16,811 / 11,986 / 4,825, teacher 268 / 261 / 7, school 117 / 93 / 24) match exactly. Every beam path and every S0 path is accepted by both the full and the pruned enumeration (0 violations).

## Where availability is lost

- **Concept:** present in the graph for 98.6% of pairs and 95.2% after pruning, but only 26.1% in the beam. The largest loss is between pruned and beam; the S0 to S3 selector recovers 7.5 points of it.
- **Teacher:** scarce in the graph (15.0%), and pruning removes 80.6% of those pairs, leaving 2.9%. Beam holds 268 of 1,880.
- **School:** common (87.3%) and mostly kept by pruning (77.9%), almost absent from the beam (0.18%).
- Pruned teacher availability depends strongly on the embeddings (stage 1: 17.4%, stage 2: 4.72% on the positive cohort), which is why the rerun with Spike B's embeddings was required.
- Definitions: pruning loss = full has the target, pruned does not; beam miss = pruned has it, beam does not; selector miss = beam has it, S0 did not choose it. A beam miss locates lost availability; policy, the 25/5/1 branching budget and search may all contribute and are not separated.
- Family, relation and terminal relation were checked to coincide on this schema (concept = `item_concept`, teacher = `item_teacher`, school = `item_school`, co-learner = `enroll`). Every path has effective length 3 (bipartite graph), so action sequences equal paths.

## Verification

56 tests pass (tiny hand-computed graphs, self-loop and visited cases, brute-force fuzz against the real `_get_actions`, top-K cohort cases, funnel, resume). Independent brute force through the environment on real pairs: 66 pair-modes, 0 mismatches. 36 witnesses (concept, teacher, school across loss categories), every edge and pruning action verified, 0 problems. Old stage-1 full counts equal the new ones for all 12,860 pairs.

## Limits

- One training seed (Spike B); the top-K cohort depends on the baseline ranking and on beam-only candidates.
- The tool bundle hash in the stage-2 manifest includes macOS `._*.py` junk from tar extraction; per-file hashes equal the committed files and the tool now ignores hidden files.
- Nothing is pushed: `origin` is upstream `epfl-ml4ed/courserec`.

## Next

Primary run (train 4 / eval 3) done: see `docs/spikes/spike-a-matched-b-train4-eval3-results.md` (all nine B counts match; beam concept 8,434, S0 5,160). Spike E (reward-only) is cancelled (ADR 0002). Next: search candidate paths beyond the beam and re-select evidence, compared with a wider beam at equal budget.
