---
type: spike
status: history
superseded_by: spikes/spike-a-matched-b-train4-eval3-results.md
---

# Spike A results: alternative-path reachability (MOOCCube, T3)

Status: COMPLETE (run on Colab, real embeddings, exact full cohort, 0 censored).
Raw outputs: Drive `MyDrive/courserec/spike_a/mooc_T3/` (`report.md`, `manifest.json`, `summary.json`, `pairs.csv`, `witnesses.json`, `excluded_pairs.csv`).
Tool and how to run: `docs/spikes/spike-a-alt-path-reachability.md`, code in `src/spike_a/` (merged into main).

## Question

For each test pair (learner, positive course not in train), how many distinct paths exist, and across how many relation types? Full-graph reachability, environment-pruned reachability and beam availability are reported separately.

## Inputs and provenance

- Artifacts: Drive `courserec/tmp/mooc`. All 6 Drive run folders have byte-identical `kg.pkl`, label files and `transe_embed.pkl`, so there is a single artifact set.
- KG and labels also regenerated locally from raw MOOCCube (preprocess only): sha256 identical to Drive.
- KG nodes: user 6507, item 687, school 150, concept 3180, teacher 1683 (12,207 total, matches the ADR smoke figure, no discrepancy).
- Splits: train 75,310 pairs, test 12,860, validation 9,053. No overlap, KG enrolments equal train exactly, no test or validation edge in the KG.
- Source commit `7348dc2`: **inferred** (latest upstream commit before `kg.pkl` mtime; code files unchanged since Mar 2024). Needs your confirmation.
- Embeddings: epochs unknown. The only agent run on Drive is a 2-epoch smoke run. Pruned and beam numbers depend on these embeddings.
- Config: `config/UPGPR/mooc.json`, TEST_AGENT `max_acts=250`, `max_path_len=3`, `use_pattern=false`. `KG_ARGS.path_pattern` is a reward whitelist, not applied as a search constraint.

## Definitions

- Counts are de-duplicated non-self-loop simple paths (no node revisited, matching the env's visited filter on `(type, id)`).
- Effective length = number of non-self-loop edges. Action sequences (with self-loops) = sum over k of n_k * C(horizon, k).
- Family = type of the intermediate node: co-learner (user via enrol), teacher, school, concept.
- Pruned = action sets from `BatchKGEnvironment._get_actions` (top-`max_acts`) with the same embeddings and path history.

## Results (T3, 12,860 eligible pairs, 6,507 users)

| | Full graph | Pruned (250) |
|---|---|---|
| Reachable pairs | 12,860 (100%) | 12,858 (100%) |
| Pairs with a path via concept | 98.8% | 98.6% |
| via co-learner (user) | 99.9% | 99.7% |
| via school | 76.3% | 74.2% |
| via teacher | 22.6% | 17.4% |
| >=2 patterns / >=2 families | 99.6% | 99.3% |
| >=2 paths | 100% | 100% |
| Users with any >=2-family pair | 99.9% | 99.8% |
| Paths per reachable pair (median / p90 / max) | 902 / 2,842 / 16,477 | 331 / 919 / 7,093 |
| Total paths | 16,417,075 | 5,740,952 |
| n_families distribution (1/2/3/4) | 56 / 3048 / 6911 / 2845 | 88 / 3278 / 7332 / 2160 |

Shares are over reachable pairs; over eligible pairs they are the same to one decimal except 99.6% and 99.3% for >=2 families.

- Every path has effective length 3 (items and non-items form a bipartite graph), so action sequences equal paths (inflation factor 1.0). No repeated-node paths found.
- Full vs pruned on the same pairs: 2 pairs reachable in full but not pruned; 917 pairs lose at least one family; mean fraction of paths retained 38%; 34 pairs go from >=2 families to 1.
- No metric is below the 5% review threshold at full graph. Teacher is the lowest (22.6%). This threshold is a review signal only.
- Beam (2-epoch policy, `policy_paths_epoch_2.pkl`): 4,180 pairs (32.5%) have at least one beam path to the positive; 816 pairs (6.3%) have >=2 beam patterns. 0 beam patterns fall outside the pruned enumeration, so beam and embeddings are consistent. Beam availability is not a reachability estimate.

## Verification

- 41 unit and end-to-end tests pass (tiny hand-computed graphs, self-loop and visited cases, fuzz against an independent brute force that calls the real `_get_actions`, with trimming, leakage and shape audits, safe pickle loader).
- Real pairs vs brute force: 3 pairs on Colab (full and pruned), 4 locally, 0 mismatches.
- Witnesses: 49 on Colab, every edge verified in the KG, pruned ones against `env._get_actions`. Raw-file cross-check was done in the local run (same KG); not on Colab (no `processed_files` there).
- Cost: full T3 58 s, pruned T3 281 s on Colab CPU. Horizon 4 is cheap but adds no new lengths; horizon 5 is very expensive (about 320 s for 10 users with the expansion cap hit), so it needs sampling and gives lower bounds.

## Interpretation and limits

- Headroom is large: almost every test positive is reachable through several relation types, and pruning removes about 62% of paths but keeps multi-family access for 99.3% of pairs.
- Teacher paths are the scarcest and the most hurt by pruning (22.6% to 17.4%).
- The beam shows far less diversity than the graph allows (6.3% of pairs with >=2 patterns vs 99.3% available), which is the gap that searching candidate paths beyond the beam and re-selecting evidence targets (ADR 0002).
- Counts only confirm headroom and constraints. They do not measure controllability or learnability; spike E (reward-only) is not pursued (ADR 0002; code and reports kept).
- Caveats to settle: confirm the source commit, and the embedding epochs.
