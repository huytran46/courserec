# Docs index

## Core (read in this order)
| File | Purpose | Read when |
|---|---|---|
| `paper-notes.md` | Summary of the base paper (UPGPR, LAK '24) | First |
| `thesis-context.md` | Thesis direction, method, RQs, wording rules | Before touching reward, evaluation, experiment design |
| `thesis-feasibility.md` | OLD reward-shaping feasibility summary (vi), superseded; some sections still useful (see banner) | History, Colab/protocol facts |
| `findings.md` | Verified facts about code, data, metrics, scope | Before designing metrics or the data split |
| `adr/` | Decisions: 0001 run on Colab, 0002 pivot to alternative-path selection (current thesis direction) | Before changing infra or making a design decision |
| `paper.pdf` | Full paper, 12 pages (untracked) | Only if `paper-notes.md` lacks detail |

## Spikes (`spikes/`)
Single-assumption smoke runs (2 epochs, Xuetang, Colab T4, seed 23).
- `init_spikes.md`: setup/timing/protocol smoke spikes (still valid for the baseline).
- Spike A (alternative-path reachability): `spike-a-report.md` (start here), `spike-a-findings.md` (lessons), `spike-a-alt-path-reachability.md` (how to run), `spike-a-results.md` (stage 1), `spike-a-matched-b-results.md` (3/3), `spike-a-matched-b-train4-eval3-results.md` (4/3, primary).

Other: `../README.md` (how to run the pipeline), `../notebooks/colab_upgpr_mooc.ipynb` (Colab runner, ADR 0001).

## Conventions
- `findings.md`: grouped by topic; state what was verified and how. Mark unverified claims "verify before citing".
- New decisions go in `adr/` with the next number. Update `thesis-context.md` if scope changes.
- Spike E (reward-only) is cancelled since ADR 0002; Spike A is the premise of the current method.
- New spike: `spikes/spike-<letter>-*.md`, add a line above.
