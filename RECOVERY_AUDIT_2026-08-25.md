# Looped Transformers recovery audit — 2026-08-25

## Working baseline

- Clean working path: `C:\projects\LoopedTransformers`
- Source repository: `https://github.com/MoralityLabAI/HybridTRMLDT.git`
- Recovery branch: `recovery/looped-transformers-20260825`
- Verified baseline: `d3cc9380b7c0ce4b21a9f79b15a9b5db6a7d5395`
- Baseline time: 2026-07-22 21:31:35 -04:00
- Baseline subject: `Audit RLM pregate instability and costs`

The July 22 endpoint is the tip of a continuous feature-branch lineage. The public
`master` branch itself stops at July 17.

## Forensic source

The original recovered tree remains untouched at:

`E:\Recovered_C_projects\Spiral_target\Users\patri\OneDrive\Documents\ChatGPT\Looped Transformers`

Its Git metadata has no refs, reflogs, valid commit objects, or remote URL. The
recovery audit found 8,886 untracked files and 6,338 null-corrupt files. A focused
core-text audit found 660 text files, of which 633 were null/empty. Most of the 27
non-null files contained unrelated cache fragments. The damaged `.git` directory
cannot supply a later baseline.

## Exact post-July material restored here

The working tree contains twelve byte- or syntax-validated files recovered directly
from the forensic tree: VRPD registries/schema, the spiral-v4 planner, two SRT loop
tests, three PowerShell runners, two oracle configs, and `examples/__init__.py`.

Two late-stage files were reconstructed from exact surviving Codex tool-call text:

- `research_gym/benchmarks/arc_open_alphabet_spiral_v4/confirmation_result.md`
- `research_gym/benchmarks/arc_public_descriptive_v1/preflight.py`

The original task log records the v4 synthetic confirmation as 47/64 SRT solves
versus 35/64 for the strongest matched control (+18.75 points), with a positive
root-block interval. It also records a descriptive public ARC result of 0/24. The
underlying v4 rows, summary, audit, public rows, and final checkpoint are corrupt or
missing, so these claims are historical evidence and cannot yet be independently
recomputed from recovered artifacts.

## Surviving paper branch

The Morality Lab site repository contains a fully intact remote branch:

`codex/spiroscopic-recursive-transformers-paper`

Commit `a90615ab5ca9226b77947eb16e7833aa45577938` (2026-08-15) preserves the
Spiroscopic Recursive Transformers Markdown paper, PDF, build script, and site index
entry. The local `C:\projects\morality-lab-site` checkout is on that branch.

## Experiment archive audit

The preserved overlay ZIP is:

`C:\Users\patri\Documents\Codex\Recovered_Rebuild_2026-08-23\worktree_overlays\Looped-Transformers-f8c4711213.zip`

It retains names and bytes for 3,853 files, but it is quarantine evidence rather
than a working overlay. Among 290 ARC/MeTTa JSON or JSONL artifacts checked, only 26
parsed as JSON under a strict pass; core v3 row files contain non-UTF-8 contamination.
None of the 215 `.pt` entries has a valid PyTorch ZIP checkpoint header; 126 begin
with zero bytes and the remainder have unrelated/corrupt headers. Do not overlay
this ZIP wholesale onto the clean repository.

## Known recovery boundary

No accessible VS Code local history, Windows Search record, valid Python bytecode,
Git object, or completed alternate recovery-scan output was found for the missing
source. The saved Codex continuation briefs remain at:

`E:\Recovered_C_projects\Sensitive_state\current_2026-08-23\Codex_history_recovery\continuation_briefs`

They preserve the three project task goals and the latest recorded actions. Future
reconstruction should treat those briefs and this audit as evidence, implement
missing modules in small reviewed slices, and rerun every experiment rather than
trusting damaged outputs.

## Codex project handoff

Saved project ID `726a5142-ee8c-4df1-9ed8-87cdd382a102` was repointed to
`C:\projects\LoopedTransformers`. The pre-edit state is backed up at:

`C:\Users\patri\Documents\Codex\Recovered_Rebuild_2026-08-23\codex-global-state.before-looped-repoint.20260825.json`

The running Codex desktop process may retain the previous path until the app is
reopened.
