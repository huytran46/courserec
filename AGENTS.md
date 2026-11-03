# courserec

Code for paper "Finding Paths for Explainable MOOC Recommendation: A Learner Perspective" (LAK '24, arXiv 2312.10082). Summary: `docs/paper-notes.md` (read first). Full PDF `paper.pdf` (repo root, untracked, 12 pages, use `pages` param) only if notes lack detail.

UPGPR = Unrestricted PGPR: RL agent walks a knowledge graph (learner -> ... -> course) to recommend courses with path explanations. Baselines via RecBole (Pop, NeuMF, CFKG).

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
- PGPR vs UPGPR switched purely by config (`reward`, `use_pattern`, `max_path_len`, `topk` in TRAIN_AGENT/TEST_AGENT). See README "Config files".
- Custom dataset: define `KG_ARGS` in config (see `config/UPGPR/example.json`).
- No test suite. Datasets are large and not in repo - don't assume they exist.
