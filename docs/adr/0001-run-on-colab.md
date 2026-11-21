---
type: decision
status: current
---

# 0001: Run experiments on Google Colab

Status: Accepted (2026-10-10)

## Context

Training (TransE, actor-critic agent) and testing need a GPU. The code selects `cuda` if available, else `cpu` (no MPS support). The dev machine is an Apple Silicon Mac with limited free disk, so it would run on CPU only.

Colab's default Python was 3.13 when checked. The repo pins Python 3.10 and old libraries (`recbole==1.2.0`, `wandb==0.16.3`, `skillner==1.0.3`).

## Decision

- Run the pipeline on a Colab T4 runtime via `notebooks/colab_upgpr_mooc.ipynb`.
- Install deps in a **Python 3.10 venv created with `uv`**, so Colab's own packages are untouched. All repo scripts run through the venv's python as CLI subprocesses. Pin `setuptools<81` (wandb 0.16.3 needs `pkg_resources`).
- Persist to Google Drive: MOOCCube zip cached in `MyDrive/courserec/cache`, repo `tmp/` symlinked to `MyDrive/courserec/tmp`. Only the 4 needed relation files are extracted to the runtime.
- Skip finished steps with **marker files that include the epoch count** (`tmp/mooc/.done_<step>[_e<epochs>]`), written after the step succeeds. Output-file checks were rejected: a 2-epoch smoke run leaves the same files as a full run and would make the full run skip.
- Clone from upstream (`epfl-ml4ed/courserec`) into the runtime. Local edits are not synced unless pushed to a remote the notebook clones.
- Drive the notebook from Claude Code through the Colab MCP server (`uvx git+https://github.com/googlecolab/colab-mcp`), added at **local** scope (this project only). Not pinned to a commit.

## Verified (smoke runs, Xuetang, T4)

- preprocess + make_dataset: 6,507 users, 687 courses, 12,207 KG nodes.
- train_transe: about 23 s per epoch. train_agent: about 26 s per epoch. test_agent: about 400 s for 6,507 users, mostly CPU-bound (GPU util 0-3%).
- 2-epoch smoke metrics are not meaningful for comparison with the paper.

## Consequences

- `train_agent` and `train_transe` are not resumable. If the runtime dies mid-step, that step restarts.
- `/content` (venv, repo, `data/mooc/processed_files`) is lost when the runtime ends. `tmp/` and the zip cache survive on Drive. Preprocess is fast to redo.
- Free tier may disconnect long sessions. Many-run experiments (path lengths x seeds x reward variants x 2 datasets) may justify Colab Pro.
- The Colab MCP server is new and unpinned, so its behavior can change. Reconnecting after a runtime change may need the browser tab reloaded.
- Smoke-run outputs currently sit in Drive `tmp/mooc`. Full runs use different markers (`_e30`, `_e50`) and overwrite embeddings, checkpoints, and `run_0.json`.

## Alternatives considered

- Run locally on the Mac: CPU only, and disk is tight. Rejected.
- Use Colab's default Python for the repo deps: not tested. Rejected as risky given the old pinned libraries.
