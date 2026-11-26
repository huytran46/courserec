---
type: guide
status: current
---

# LLM explanation + claim check (v1 pilot)

Design: `docs/superpowers/specs/2026-11-26-llm-explain-v1-design.md`. Read-only on `src/UPGPR/`.

**All checker numbers are unvalidated until step 5. Do not report them as findings.**

## Run (Colab T4)
Use `notebooks/llm_explain_v1.ipynb`, or from the repo root with Drive mounted:

```
D=/content/drive/MyDrive/courserec; OUT=$D/llm_explain_v1
pip install -q transformers accelerate bitsandbytes
python src/llm_explain/run.py prepare  --zip $D/cache/MOOCCube.zip
python src/llm_explain/run.py sample   --root $D/m5 --tmp_dir $D/tmp/mooc_train4_eval3 --out $OUT
python src/llm_explain/run.py generate --out $OUT --smoke    # 1 example per group, printed
python src/llm_explain/run.py generate --out $OUT            # resumable
python src/llm_explain/run.py report   --out $OUT
```

`sample` stops if the embedding hash, the seed-23 Top-10 group counts, or any sampled KG edge does not match.

## Outputs in `$OUT`
- `sample.jsonl`, `meta_sample.json` (beam and embedding sha256, seed, counts)
- `results.jsonl` (per example: facts, explanation, raw extractor output, claims + labels), `meta_generate.json`
- `summary.json` (per group; rates exclude `recommendation_framing`)
- `hand_labels.csv` (40 examples, one row per claim, checker labels hidden). Never overwritten.

## 5. Hand labels
Fill in a copy of `hand_labels.csv`:
- `human_true` (every claim row): `true` if the claim says nothing beyond `path_facts`, else `false`.
  A plain "X is recommended to learner" is `true`; with an added reason the path does not show, `false`.
- `path_odd` (any text, one row per example is enough): path is hard to understand even if told faithfully.
- `missed_claim`: statement in `explanation` that no row covers.

Save as CSV UTF-8, then:

```
python src/llm_explain/run.py score --out $OUT --labels <filled.csv>
```

Positive class = claim not true on the path. Checker positives = `unsupported_claim`, `unknown_entity`,
`entities_in_path_relation_mismatch`. Precision/recall cover extracted claims only.

## Limits
- Wording check is a fixed English word list (`core.WORDING`): it misses paraphrases and can over-flag.
- `teacher`/`school` groups are tiny in the control beam (seed 23: 27 and 4 slots in total).
- A wrong display name next to a correct code does not change a claim label; it is counted as `code_name_mismatches`.
- LLM judge not built; add it only if the checker is weak on hand labels.

## Tests
`pytest tests/test_llm_explain_*.py` (numpy needed).
