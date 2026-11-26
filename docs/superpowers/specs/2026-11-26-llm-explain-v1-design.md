---
type: spec
status: current
---

# LLM explanation + claim check, v1 (pilot)

## Intent
Measure how often an LLM adds information not supported by a UPGPR path. Secondary contribution (thesis-context step 5).
No faithfulness number is a finding until the checker is validated on hand labels.
Non-goals: change UPGPR, re-rank, fix odd concept links.

## Approaches
- A (picked): logic in `src/llm_explain/` with local tests, thin Colab runner.
- B: all logic in notebook cells (as v0). Not testable, so rejected.
- Recommendation-framing claims: A (picked by user) label `recommendation_framing`, excluded from rates;
  B count as `other` (inflates every explanation).

## Decisions
- Beam: M5 control (lambda0), seed 23, test. Train 4 / eval 3, 50 epochs (not the paper config).
- Top-10 and displayed path: `m5_metrics.build` + `spike_b_posthoc.select_path("S0")`, as in `m5_paths.py`
  (`m5_metrics.freeze` tie-break, not `spike_b_posthoc.freeze_ranking`).
  Gate: seed-23 slot group counts must equal S0_PATH_TYPES.md (56877 co_learner, 7603 concept, 27 teacher, 4 school).
- Labels and embeddings from M5 `tmp/mooc_train4_eval3`; embedding sha256 must match the REPORT (5769c715...e787).
- Sample: 100 co_learner, 100 concept, up to 25 teacher, up to 25 school. Seed 0.
- Entity codes `[L#] [C#] [K#] [T#] [S#]`, numbered by order in the path. Checker matches codes only.
- Writer and extractor in English; entity names kept as given. CJK outside entity names is counted.
- Extractor returns `{head, relation, tail, quote}`. Relations: enrolled_in (L->C), teaches_concept (C->K),
  shares_learner_with (symmetric; L-L via a course both enrolled in on the path, or C-C via a learner on the path),
  taught_by (C->T), offered_by (C->S), recommended_to (C->L target only), other.
- Labels: supported, entities_in_path_relation_mismatch, unknown_entity, unsupported_claim, recommendation_framing.
  `unsupported_claim`: relation other, or the quote has temporal/causal wording.
  `recommendation_framing`: recommended_to for the final course and the target learner, with no such wording in the quote.
  "because" alone is framing; "because it builds on" is not.
- Greedy decoding, fixed seeds; log model, revision, quantization, prompts, beam hash.
- Hand-label CSV: 40 examples stratified by group, one row per claim, `human_true` column empty, no checker label shown.
  A separate score step computes precision/recall of the checker.
- Odd-but-faithful paths: a `path_odd` column in the CSV; counted by the user.
- LLM judge: only if the checker proves weak on hand labels (not built now).
