# Docs index

Every doc has frontmatter: `type` and `status`, plus `superseded_by` when replaced.

- `type`: `fact` (verified facts), `context` (thesis direction), `decision` (ADR), `spike` (smoke run result), `howto` (how to run), `history` (old, kept for reference).
- `status`: `current` (trust it), `superseded` (replaced, follow `superseded_by`), `history` (old run or detail, not the primary result).
- Agents: skip `superseded` and `history` docs unless asked for history. Filter fast: `grep -L 'status: superseded' docs/**/*.md`.

## Core (read in this order)
| File | Type | Status | Purpose | Read when |
|---|---|---|---|---|
| `paper-notes.md` | fact | current | Summary of base paper (UPGPR, LAK '24) | First |
| `thesis-context.md` | context | current | Thesis direction, method, RQs, wording rules | Before touching evaluation, path selection, experiment design |
| `findings.md` | fact | current | Verified facts about code, data, metrics, scope (topics below) | Before designing metrics or the data split |
| `adr/` | decision | current | Decisions (list below) | Before changing infra or making a design decision |
| `thesis-feasibility.md` | history | superseded | OLD reward-shaping feasibility (vi). Banner lists the still-useful sections | History, Colab/protocol facts only |
| `paper.pdf` | - | - | Full paper, 12 pages (untracked) | Only if `paper-notes.md` lacks detail |

## `findings.md` topics
| Section | Content | Status |
|---|---|---|
| Data split and leakage | 80/10/10 per user, random, no seed in code, train-only enrolled edges | current |
| Explanation-quality metrics: LIR, SEP, ETD | Feasibility of metrics on our data, LIR needs timestamps | current |
| Path coverage | "path exists on KG but not found" | current |
| Scope check: post-hoc selection + LLM proposal | Old scope check | superseded |
| Xuetang has no course category | Dataset limitation | current |

## ADRs (`adr/`)
| ADR | Decision | Status |
|---|---|---|
| `0001-run-on-colab.md` | Run experiments on Google Colab | current |
| `0002-pivot-to-alternative-path-selection.md` | Drop reward shaping, search and re-select evidence paths (current thesis direction) | current |

## Spikes (`spikes/`)
Single-assumption smoke runs (Xuetang, Colab, seed 23). Spike E (reward-only) is cancelled since ADR 0002.

| File | Type | Status | Purpose |
|---|---|---|---|
| `init_spikes.md` | spike | current | Setup/timing/protocol smoke spikes, valid for the baseline only |
| `spike-a-report.md` | spike | current | Spike A (alternative-path reachability) conclusion. Start here, it is the premise of the current method |
| `spike-a-findings.md` | spike | current | Spike A lessons |
| `spike-a-matched-b-train4-eval3-results.md` | spike | current | Spike A primary run (train 4 / eval 3), full tables |
| `spike-a-alt-path-reachability.md` | howto | current | How to run Spike A |
| `spike-a-matched-b-results.md` | spike | history | Spike A 3/3 run, secondary to the 4/3 run |
| `spike-a-results.md` | spike | history | Spike A stage 1, replaced by the matched runs |

New spike: `spikes/spike-<letter>-*.md`, add a row above.

Other: `../README.md` (how to run the pipeline), `../notebooks/colab_upgpr_mooc.ipynb` (Colab runner, ADR 0001).

## Conventions
- Every new doc starts with frontmatter (`type`, `status`, `superseded_by` if any). Add it to the index above.
- When a doc is replaced: set `status: superseded` + `superseded_by`, add a banner, update this index.
- `findings.md`: grouped by topic; state what was verified and how. Mark unverified claims "verify before citing". Add new topics to the table above.
- New decisions go in `adr/` with the next number. Update `thesis-context.md` if scope changes.
