# Thesis context

Why this repo is being used. Read before touching evaluation, path selection, or experiment design.

Proposal: `Đề cương LVThS` (2026-10-29), supervisor-approved. Decision record: `docs/adr/0002-pivot-to-alternative-path-selection.md`. The earlier reward-shaping direction is dropped.

## Direction

MSc thesis, **ML/technical** framing: *Explainable Personalized Course Recommendation* (VI: Khuyến nghị khóa học cá nhân hóa có khả năng giải thích).

- Base: UPGPR (*Finding Paths for Explainable MOOC Recommendation: A Learner Perspective*). Summary in `docs/paper-notes.md`. UPGPR is the recommender and stays unchanged; Top-10 is kept.
- Problem: each recommended course ends a path found by the agent, but that path is not necessarily the best evidence. Many valid paths exist per (learner, course) pair, yet the beam keeps 1-2 and only shows one or a few. Some may tie better to the learner's history (e.g. via a recently enrolled course or a specific concept), and many go through very popular entities that tell the learner little. Spike A confirms the headroom (see `docs/spikes/spike-a-report.md`).
- Input: enrolment history, course info, KG relations from training data. Output: Top-K courses with a short explanation backed by real KG relations.

## Method

1. For each (learner, UPGPR-recommended course) pair, search additional candidate paths on the KG under limits on path length and visited nodes.
2. Rank candidates by relevance to the learner's history (LIR-style recency of the linking interaction) and relation-type diversity; avoid popular shared entities (SEP). Keep the best as evidence.
3. Top-10 is unchanged when evaluating explanations.
4. Baselines for the explanation side: UPGPR's own path; widening UPGPR's beam under the same compute budget.
5. Optional: LLM turns the evidence path into a sentence, restricted to entities and relations in the path; compare with template sentences (faithfulness, understandability). Optional small survey.
6. Demo web: Top-10 for anonymized learners, template-based sentence per course, open the evidence path, compare UPGPR vs proposed.

## Evaluation

- Recommendation (sanity check that nothing changed): NDCG@10, Recall@10, HR@10.
- Explanation path quality: LIR (linking interaction recency), SEP (shared entity popularity), ETD (explanation type diversity). Source: Balloccu et al., SIGIR 2022 and KBS 2023 (verify definitions against the papers before implementing).
- Cost: processing time, extra visited nodes per learner.
- Protocol: Xuetang, 80/10/10 split per learner, only train enrolments create enrol edges, 3 seeds, tune on validation.

## Research questions

- RQ1: How effective is UPGPR for course recommendation, and what kinds of explanations does it produce?
- RQ2: How can explanation paths be chosen to be more tied to learning history and more diverse?
- RQ3: How does the proposal fare on recommendation, explanation quality, and compute cost?

## Contribution (intended claim)

A search-and-reselect method on the KG that gives evidence paths with better LIR/SEP/ETD than UPGPR's own paths or a wider beam at the same budget, with recommendation quality unchanged and bounded extra cost.

## Known limits to state in the thesis

- Evidence paths may lie outside the agent's beam, so they justify the recommendation post hoc and are not the path that drove the model's choice.
- Selecting by LIR/SEP/ETD and evaluating by the same metrics is circular. Need an independent check (survey, or a measure not used for selection).
- Recommendation metrics are equal to the baseline by construction when Top-10 is kept.
- Xuetang has no course category; "topic" = concept. Path families: concept, teacher, school, co-learner. Teacher paths exist for only ~15% of top-K pairs, so ETD has a low ceiling.
- Wording rule: no strong human-perception claims ("more understandable", "more trustworthy") without the survey. Use "relevance to history", "popularity of shared entity", "type diversity".
- One dataset (Xuetang); COCO needs a request to the authors.

## Pipeline

```
UPGPR baseline (train, 3 seeds) -> Top-10 + agent paths
  -> enumerate / search candidate paths under budget
  -> rank + select evidence (LIR, SEP, ETD)
  -> evaluate: rec metrics unchanged, path quality, cost
  -> optional: LLM sentence, survey
  -> demo
```

## Baseline reference (Xuetang NDCG, from the paper)

| Pop | NeuMF | CFKG | PGPR(3) | UPGPR 3 / 4 / 5 |
|---|---|---|---|---|
| 3.06 | 14.05 | 14.20 | 18.16 | 18.43 / 17.96 / 20.85 |

The reproduced baseline must be close to these before anything built on it is trusted.

## Timeline (proposal)

12/2026-05/2027. Dec-Jan: survey, data, reproduce UPGPR. Jan-Mar: analyze explanations, develop method. Feb-Apr: experiments, survey, demo. Mar-May: thesis and paper draft.

## Open items

- Baseline protocol (train at L+1 as in the paper vs repo default 3/3), number of seeds, and whether the split seed varies. Needs an ADR.
- Full-epoch baseline not yet run; no paper number reproduced (only 2-epoch smoke runs).
- Carry enrolment timestamps through preprocess and split (needed for LIR).
- Define LIR/SEP/ETD precisely against Balloccu et al.; implement in a module.
- Independent evaluation of explanation quality (survey scope).
- COCO: not available locally.
