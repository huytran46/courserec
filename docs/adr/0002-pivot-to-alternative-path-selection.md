---
type: decision
status: current
---

# 0002: Pivot from reward shaping to alternative-path search and re-selection

Status: Accepted (2026-11-05). Direction approved by the supervisor (thesis proposal dated 2026-10-29, "Khuyen nghi khoa hoc ca nhan hoa co kha nang giai thich").

## Context

The earlier direction was explanation-oriented reward shaping for UPGPR (`R = R_rec + l1 R_length + l2 R_relation + l3 R_diversity`, retrain the agent). Spike A (`docs/spikes/spike-a-report.md`) showed on Xuetang, schema T3:

- Many valid paths exist per (learner, course) pair: median ~1,000 in the full graph, but the beam holds 1-2.
- Availability is lost mostly between the pruned graph and the beam, not in the KG. Selecting among beam paths alone recovers little (S0 -> S3: concept +3,274 pairs, teacher 8, school 1 on 4/3).
- Beam composition is very skewed: co-learner 87.5%, concept 26%, teacher 0.4%, school 0.2%.
- Exhaustive enumeration at horizon 3 is cheap (861-1198 s on Colab CPU for 12-64k pairs).

## Decision

Thesis topic: **Explainable personalized course recommendation** (EN: Explainable Personalized Course Recommendation).

- Keep UPGPR unchanged as the recommender. The Top-10 list is not modified.
- Contribution: for each (learner, recommended course) pair, search additional candidate paths on the KG under a length and visited-node budget, rank them by relevance to the learner's history and relation-type diversity, keep the best as evidence. See `docs/thesis-context.md`.
- Compare against UPGPR's own path and against widening UPGPR's beam under the same compute budget.
- Evaluate with LIR, SEP, ETD (Balloccu et al., SIGIR 2022 and KBS 2023), plus cost (time, visited nodes). Recommendation metrics (NDCG@10, Recall@10, HR@10) are a sanity check that they are unchanged, not a result.
- Optional: LLM-written explanation constrained to path entities and relations; small survey; demo with template sentences.
- Reward shaping is dropped as the core. Spike E (reward-only) is cancelled. Spikes A and B stay as supporting evidence.

## Consequences

- No retraining of the agent for the method. Only the baseline UPGPR needs full training (3 seeds). Compute cost drops sharply.
- The selected evidence path may lie outside the agent's beam, so it is a post-hoc justification, not the path that drove the model's decision. State this limit in the thesis; avoid "the reason the model recommended it".
- Selecting by LIR/SEP/ETD and then evaluating with the same metrics is circular. Need at least one independent check (survey, or a measure not used for selection) and must say so as a limit otherwise.
- LIR needs enrolment timestamps. `entities/user.json` has them (100% of 682,753 pairs match) but `preprocess_mooc.py` drops them. Preprocess must carry them through the split.
- Xuetang has no course category. "Topic" means concept. Path families are concept, teacher, school, co-learner only, so ETD has a low ceiling.
- Wording rule: no strong claims about learner understanding or trust without the survey.
- Superseded docs: reward-shaping parts of `thesis-context.md` (rewritten), `thesis-feasibility.md` and `spikes/init_spikes.md` (kept for history, banner added).
