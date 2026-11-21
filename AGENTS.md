# courserec

Code for paper "Finding Paths for Explainable MOOC Recommendation: A Learner Perspective" (LAK '24, arXiv 2312.10082). Summary: `docs/paper-notes.md` (read first). Full PDF `docs/paper.pdf` (untracked, 12 pages, use `pages` param) only if notes lack detail.

UPGPR = Unrestricted PGPR: RL agent walks a knowledge graph (learner -> ... -> course) to recommend courses with path explanations. Baselines via RecBole (Pop, NeuMF, CFKG).

Why we use it: MSc thesis on explainable personalized course recommendation. UPGPR stays unchanged as the recommender; the thesis searches extra candidate paths on the KG and re-selects evidence paths (reward shaping was dropped, ADR 0002). Read `docs/thesis-context.md` before touching evaluation, path selection, or experiment design. Decisions are in `docs/adr/`. Doc index: `docs/README.md` (every doc has `type`/`status` frontmatter; skip `status: superseded` or `history` docs unless asked for history). Verified facts (split/leakage, LIR/SEP/ETD feasibility, path coverage, scope checks) are in `docs/findings.md` - read before designing evaluation metrics or the data split. Experiments run on Colab via `notebooks/colab_upgpr_mooc.ipynb` (see ADR 0001).

## Layout
- `src/UPGPR/` - pipeline: `preprocess_*` -> `make_dataset` -> `train_transe_model` -> `train_agent` -> `test_agent`. Core: `kg_env.py`, `actor_critic.py`, `knowledge_graph.py`, `transe_model.py`.
- `src/baselines/` - RecBole formatting (`format_*`) and `baseline.py`.
- `config/UPGPR/*.json` (mooc, coco, example), `config/baselines/*.yaml`.
- `data/` - datasets (Xuetang/MOOCCube, COCO gitignored), `data/example/` = tiny custom-dataset format sample.
- Outputs go to `tmp/<dataset>` (gitignored).

## Setup
- Python 3.10, `pip install -r requirements.txt`; COCO skills need `python -m spacy download en_core_web_lg`.
- Run scripts from repo root with `--config config/UPGPR/<name>.json`. Full commands in `README.md`.

## Notes
- Thesis metrics: LIR, SEP, ETD (Balloccu et al.) plus cost; rec metrics must stay unchanged (Top-10 kept). LIR needs enrolment timestamps, which `preprocess_mooc.py` currently drops.
- PGPR vs UPGPR switched purely by config (`reward`, `use_pattern`, `max_path_len`, `topk` in TRAIN_AGENT/TEST_AGENT). See README "Config files".
- Custom dataset: define `KG_ARGS` in config (see `config/UPGPR/example.json`).
- Tests: `tests/` (pytest, spike A/B/E/M5 code only, pytest not in requirements.txt). No tests for the base UPGPR pipeline. Datasets are large and not in repo - don't assume they exist.
