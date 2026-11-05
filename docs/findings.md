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

## Scope check: post-hoc path selection + LLM proposal

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
