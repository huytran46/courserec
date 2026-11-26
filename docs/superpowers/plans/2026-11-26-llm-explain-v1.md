---
type: plan
status: current
---

# LLM explain v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans (inline).

**Goal:** Claim-check pilot on displayed Top-10 UPGPR paths.
**Architecture:** Pure modules in `src/llm_explain/` (CPU, unit-tested); `run.py` CLI does I/O and the model; thin notebook calls the CLI.
**Tech Stack:** numpy, pytest, transformers + bitsandbytes (Colab only).
**Spec:** `docs/superpowers/specs/2026-11-26-llm-explain-v1-design.md`

## Global Constraints
No change to `src/UPGPR/`. Qwen2.5-7B-Instruct 4-bit, greedy. English prompts. No faithfulness number is a finding before hand-label validation.

## Review Focus
1. Extractor drops "then"/"finished" from its quote, or invents a quote -> wording still flagged.
2. Entity display name contains a trigger word or CJK -> not flagged, not counted as leakage.
3. Extractor returns bad JSON / non-dict items -> counted as extraction failure, no crash.
4. Wrong code or no code in a claim -> unknown_entity, never fuzzy-matched.
5. Unlabeled or typo `human_true` cell at scoring -> hard error, not silently false.

## Tasks
- [ ] 1. `core.py` + `tests/test_llm_explain_core.py`: `facts(path, name_of) -> dict(entities, triples, target, final)`,
  `parse_claims(text) -> list|None`, `label_claim(claim, facts, explanation) -> str`, `cjk_leak(text, facts) -> bool`,
  `code_name_mismatches(text, facts) -> int`. Test, fail, implement, pass, commit.
- [ ] 2. `report.py` + `tests/test_llm_explain_report.py`: `summarize(rows) -> dict`, `pick_for_labeling(rows, n=40, seed=0)`,
  `write_label_csv(rows, path)`, `score_labels(label_csv_rows, key) -> dict`. Same cycle.
- [ ] 3. `run.py` (stages `sample`, `generate`, `report`, `score`), `prompts.py`, `notebooks/llm_explain_v1.ipynb`, README.
  `sample` gates: embedding sha256, seed-23 S0 group counts, every KG edge of sampled paths matches processed_files.
  Verify: tests pass, CLI `--help`, `sample` logic smoke test on synthetic inputs.
- [ ] 4. Run on Colab: sample, 4-example smoke, full, report, CSV.
