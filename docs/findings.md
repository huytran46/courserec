---
type: fact
status: current
---

# Findings

Running log of verified facts about the repo, data, and thesis scope. Newest last.

## Data split and leakage (checked in code)

- Paper: 80/10/10 split per user (train/test/val), 3 seeds.
- `src/UPGPR/make_dataset.py`: `train_test_split` per learner, 0.8 train, remainder halved into val/test. Random, not temporal, and no seed set in code (reproducibility needs an explicit seed).
- KG enrolled edges come from `train.txt` only (`data_utils.py` loads `set_name + ".txt"`, `knowledge_graph.py::_load_enrolments` uses `dataset.interactions`). Matches the "train-only enrolled edges" design.
- Avoiding leakage is our own design rationale, not a claim made in the paper.

## Explanation-quality metrics: LIR, SEP, ETD

- Not in the repo. Only NDCG, Recall, HR, Precision exist (`test_agent.py`).
- Source (from memory, verify before citing): Balloccu et al., SIGIR 2022, "Post Processing Recommender Systems with Knowledge Graphs for Recency, Popularity, and Diversity of Explanations". LIR = Linking Interaction Recency (time-based, not "relevance to history"), SEP = Shared Entity Popularity, ETD = Explanation Type Diversity.
- Feasibility on Xuetang (MOOCCube):
  - ETD: from path patterns in `pred_paths`. Feasible.
  - SEP: from entity degree in the train KG. Feasible.
  - LIR: needs enrolment timestamps. `entities/user.json` has `course_order` + `enroll_time`. Verified: 100% of 682,753 pairs in `relations/user-course.json` match a timestamp, 0 users with length mismatch, 10/199,199 users with tied timestamps.
  - `preprocess_mooc.py` reads `user-course.json` (no time) and drops timestamps, so `enrolments.txt`/`train.txt` carry none. Needs a join with `user.json` and propagation through the split.
  - `data/mooccubex/user.json` also has `enroll_time` but uses numeric course ids, different from the pipeline's `C_course-v1:...` ids. Use the matching source.
- Risk: ETD may be near-flat. Paper reports ~77% L-C-L-C at length 3 on COCO; Xuetang likely similar. Random split also means LIR reflects train history only, not future behavior.

## Path coverage ("path exists on KG but not found")

- Can happen. Paper reports 5-7% invalid users on Xuetang at len 3/4, 0% at len 5. Causes: beam width (`topk`), `max_path_len`, action-space pruning (confirm in `kg_env.py`).
- Any post-hoc alternative-path search with node/length caps can miss paths too. Agent's own path is always a valid fallback.
- Check: full BFS (length <= L) on a sample of recommended pairs, report miss rate split by cause (node cap vs length).

## Scope check: post-hoc path selection + LLM proposal (superseded)

Update 2026-11-05: the supervisor approved this direction (LLM sentence and survey optional, demo with template sentences). Now the thesis direction, see ADR 0002. The conflicts below applied to the old reward-shaping thesis and are kept as history; the limits that remain (circular evaluation, post-hoc faithfulness, rec metrics unchanged by construction) are listed in `docs/thesis-context.md`.


- Conflicts with `docs/thesis-context.md` (reward shaping is the core claim). Top-10 unchanged means rec metrics equal baseline by construction.
- Human-perception claims ("easier to understand") violate the wording rule.
- Suggested use: post-hoc path re-selection as a baseline against reward shaping (RQ4). LLM text and web demo as future work. A pivot would need an ADR and an edit to `thesis-context.md`.

## Xuetang (MOOCCube) has no course category

- `data/mooc/MOOCCube/entities/course.json` keys: `id, name, prerequisites, about, core_id, video_order, display_name, chapter`. No category/subject field.
- Category (`first_level_category` / `second_level_category`) exists only for COCO (`preprocess_coco.py`).
- `config/UPGPR/mooc.json` KG relations: `enroll`, `item_concept`, `item_teacher`, `item_school`. No category entity.
- Only indirect source: `relations/course-concept.json` -> concept -> `relations/concept-field.json` (`K_T_<name>_<discipline>`), hierarchy in `relations/parent-son.json` (e.g. world history -> ancient/medieval -> NE Asia). Concept ids carry a discipline suffix (`K_活性炭_化学`).
- Derived course category = dominant field of its concepts (not implemented). Needs noise handling, a course can span several disciplines. `course-concept.json` covers 683 courses (check against the course count in `courses.txt` before relying on it).
- Relevance: paper notes say enriching the KG with category/concept would diversify paths (ETD). On Xuetang this needs the derived category above.

## Spikes A, B, E and M5 results (MOOC / Xuetang, train horizon 4 / eval 3, test 6,507 users, top-10)

Numbers from saved outputs. Full reports, code and tables are on branch `worktree-spike-b-posthoc-paths` (`docs/m5/REPORT.md`, `docs/m5/S0_PATH_TYPES.md`, `docs/spike-m5-findings.md`) and Drive `courserec/m5/`. n=3 seeds (23/24/25), SD is sample SD, no significance claims.

- **Displayed explanation (selector S0):** control (no bonus) is co-learner dominated: 84.29% of explained slots (88.17 / 80.77 / 83.94% per seed), concept 15.54%, teacher 0.15%, school 0.02%. Shaped (lambda 5): co-learner 52.45%, concept 45.73%, school 1.00%, teacher 0.83%. No category entity exists in this KG; hybrid and other paths are 0 (horizon 3 only allows user -> item -> X -> item).
- **Reward-only terminal concept bonus (M5):** concept@10 control S0 15.42 -> shaped S0 45.20; control S3 21.41 -> shaped S3 50.98. NDCG@10 19.05 vs 19.39 (no loss). Lambda picked on seed-23 validation from {0,1,5}.
- **Where the gain comes from:** arithmetic decomposition along control+S0 -> control+S3 -> shaped+S3 is selector +6.0 pp and beam availability from reward +29.6 pp (total +35.6, about 17/83; other order 16/84). Not a causal share. S3 gain stays about +6 pp under shaping (interaction -0.2 +- 0.6 pp).
- **Funnel (Spike A cache, common covered cohort only):** concept path in full graph 99.5%, after pruning 97.9%, in the M5 beam 24.3% (control) vs 52.2% (shaped), S0 selects concept 14.0% vs 42.6%. On this cohort the bottleneck is the beam, not graph or pruning. Cache covers 54-59% of top-10 pairs (74% of common pairs); uncovered pairs are unknown, not "no path". 0 violations: selected concept always in beam; every beam and selected path is accepted by full and pruned graph.
- **Pruning (older Spike A run):** concept paths 98.86% full -> 95.60% pruned -> 13.08% in beam; teacher paths 16.29% -> 2.70% -> 0.13%; school 86.9% -> 77.5% -> 0.01%.
- **Shorter lists under shaping:** missing slots (of 65,070 per seed) shaped 761 / 739 / 743 vs control 559 / 463 / 490; users with list < 10: 304 / 303 / 305 vs 247 / 190 / 204. 49.0% of shaped S3 slots have no concept explanation (1.15% missing + 47.87% returned non-concept).
- **Unexplained:** school and teacher paths appear under shaping without being rewarded (seed 23 school 2.35% vs 0.32% in other seeds); cause not investigated.
- **Limits:** n=3, one lambda grid, selection on seed 23 only, TransE embeddings shared by all seeds (SDs exclude embedding variance), no bootstrap, new-contract metrics not comparable to legacy numbers (e.g. 19.147% NDCG), funnel limited to cache coverage. Nothing here speaks to trust, usefulness or learning benefit.
