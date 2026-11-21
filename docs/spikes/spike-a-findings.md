---
type: spike
status: current
---

# Spike A: findings and lessons (session notes)

Scope: availability and headroom only. No claim on reward shaping, causal attribution, or learner usefulness. Details: `spike-a-report.md`, `spike-a-matched-b-results.md` (3/3), `spike-a-matched-b-train4-eval3-results.md` (4/3, primary).

## Results that matter

1. **Availability is lost mostly between pruned and beam.** Primary run (4/3, 64,459 top-K pairs): concept full 98.9% -> pruned 95.6% -> beam 13.1% -> S0 8.0%. School 86.9% -> 77.5% -> 0.009% -> 0.008%. Teacher 16.3% -> 2.7% -> 0.13% -> 0.12%.
2. **Teacher is scarce in the graph and pruning removes most of it** (83% of pairs that have a teacher path lose it). School is common and kept by pruning, but the beam almost never holds it.
3. **Selector S0 -> S3 recovers little**: concept 3,274, teacher 8, school 1 pairs (4/3). Post-hoc selection cannot create paths the beam lacks.
4. **Beam differs a lot between runs, graph and pruning do not.** 3/3 beam: concept 8.4k -> 16.8k vs 4/3. School 117 vs 6. Same KG, labels, embeddings. Not separated: horizon, seed, search.
5. **Pruned numbers depend strongly on the embeddings.** Stage 1 (2-epoch smoke embeddings) gave pruned teacher 17.4% on positives; full embeddings give 4.72%. Stage-1 pruned and beam numbers must not be cited. Stage-1 full-graph counts are valid (0 differences on 12,860 pairs).
6. **All nine reference counts match Spike B** (concept 8,434/5,160/3,274, teacher 85/77/8, school 6/5/1 on 4/3; 16,811/11,986/4,825, 268/261/7, 117/93/24 on 3/3). Every beam and S0 path is accepted by full and pruned enumeration (0 violations).
7. **Structure of the schema**: bipartite (items vs non-items), so every T3 path has effective length 3 and action sequences equal paths. Family = intermediate type = non-enroll relation = terminal relation (concept `item_concept`, teacher `item_teacher`, school `item_school`, co-learner `enroll`), so Spike B's "path contains relation" equals these families.

## Provenance facts

- KG `193776f1...`, train `2e211fd4...`, test `6c9ac4ec...`, validation `db13246c...`, embeddings `5769c715...` are identical in the 3/3 and 4/3 runs. Only config and beam differ (3/3: config `996b9d1a...`, beam `7fe3509a...`; 4/3: config `bb18f328...`, beam `5e6bae1e...`).
- Embeddings file = `transe_model_sd_epoch_30` checkpoint, no other epoch of 30 matches. Early stopping can change the saved epoch, so check equality, not the file name.
- Pipeline commit `7348dc2` (stated by Spike B). Tool commit for both matched runs `3aa263b` (local only). Spike B tool hash `654f8e28...` (unchanged; one message had it garbled).
- Not audited on Colab: ID-map files (no `processed_files`). A local rebuild from raw data gave a byte-identical KG and labels earlier.
- Cohorts: positives 12,860 pairs / 6,507 users (same in both runs); S0 top-K 64,461 (3/3) vs 64,459 (4/3), so not pair-identical. Compare funnels within a run.

## Limits

One seed per run. Top-K depends on the baseline ranking and the beam. Counts are headroom, not controllability (spike E). The tool bundle hash of the 3/3 run includes macOS `._*.py` junk; per-file hashes equal the committed files and the tool now ignores hidden files (the 4/3 bundle hash `221c1e95...` is clean).

## Operational lessons

- Colab runtime resets lose `/content` (venv, repo, checkpoint). Rebuild: Python 3.10 venv, `numpy<2`, torch CPU, `scipy==1.11.4`, `scikit-learn==1.3.2`, `wandb==0.16.3`, **`setuptools<81`** (wandb needs `pkg_resources`). Ship the tool as a base64 tarball and verify sha256; a single corrupted block was caught by per-chunk md5.
- The Colab MCP is flaky: long calls hang, cells get wiped, tools vanish until `/mcp` reconnect. Run long jobs in the background (`Popen`, log to file), poll with short cells, keep `--resume` on. The 4/3 run (1,198 s) finished while the MCP was down.
- A worktree-isolated session cannot run git against the main checkout. Landing work on `main` needs the main checkout or a separate session.

## Next

Spike E (reward-only) is **not pursued** (ADR 0002, reward shaping dropped; code and reports kept). The headroom measured here (many paths in the full graph, few in the beam) is the premise of the new method: search candidate paths beyond the beam, then re-select evidence. Next: implement the search and selector, define LIR/SEP/ETD, compare with a wider beam at equal budget. Optional: ID-map audit with `processed_files`; fork to push (`origin` is upstream `epfl-ml4ed`, do not push there).
