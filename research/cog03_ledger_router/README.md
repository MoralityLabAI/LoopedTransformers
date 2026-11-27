# COG-03: recurrent proof-ledger router

This gym trains a small transformer to choose a rule in a finite propositional proof search.
DataGoblin COG-02 exports each world and a ledger of accepted and rejected attempts.
The router reads the public graph and current facts, then submits one action to an independent Horn checker.

The model reuses `block` from `research/lean_cgrt_hrm400/ppt_gym/core.py`.
Its rule tokens carry ten allowlisted public features; a dependency mask routes messages backward from consumers to prerequisite rules.
Four passes share one cell. A gated state persists between checker calls.

## Run

```bash
python -m research.cog03_ledger_router.run \
  --corpus research/cog03_ledger_router/data/tasks.jsonl.gz \
  --output /tmp/cog03-run
python -m unittest discover -s research/cog03_ledger_router/tests -v
```

Install CPU PyTorch 2.5.1 to reproduce the recorded run. The implementation supports Python 3.10+ and PyTorch 2.2 through 2.x.
The default run fits seeds 17 and 29 for 180 updates each, at inner depths four and one, with two CPU threads.
It preserves checkpoints and all optimizer losses. Training teacher checks and evaluation checker calls have separate counts.

## Evaluation

Every policy receives 24 checker calls per root. Each rejected attempt costs a call, and a completed goal ends the episode.
A shared action filter skips known conclusions and retries a failed rule after the ledger changes.
The controls include uniform choice, first ready rule, and a symbolic goal-ancestry policy that can read the full public rule graph.
The latter supplies a structural ceiling for this task family.

The learned controls include a separately trained one-pass model with the same parameter count, clearing carried history, resetting every inner pass, and rolling history across rule positions.
The reset-inner intervention still executes four cell calls, which exposes the cost of removing information between passes.
Registration fixes source roots and all arms before training. Final checkpoints are used directly.

The evidence class is a public development pilot over synthetic Horn proof DAGs.
The distinct-root count remains separate from repeated checkpoint measurements.
The causal power gate described by EvalMath's mechanistic-power skill remains open; reports keep raw paired outcomes for that next gate.

The four training fits made 159,724 calls to the finite Python checker for labels and teacher transitions.
Evaluation made 40,968 checker calls across 3,072 episodes. These cost counters describe that checker backend.
DataGoblin separately compiled all 256 source proof certificates with Lean 4.35.0.
The local cross-check compared both Python interpreters on 81,046 rule and sentinel applications across 2,709 teacher states.

The ordinary test contains 64 roots measured at two training seeds. Both fitted architectures completed all 128 trials.
The four-pass model used 7.453 calls per trial and the one-pass model used 8.375.
The 64 deeper OOD roots yielded 125/128 completions for four passes and 122/128 for one pass.
Clearing carried state reduced the four-pass checkpoint's OOD completion count to 87/128.
All arms and both seeds remain in the report, including the symbolic policy's 128/128 OOD completions.

The merged EvalMath FSI-01 bridge assigns `inconclusive` to the primary completion comparison in both test splits.
The OOD cluster effect is 2.34375 percentage points, with an approximate cluster bootstrap interval from -3.125 to 8.59375 points and unadjusted cluster sign-flip p = 0.528.
The registered scientific effect floor is ten points, and this public development readout remains unsealed.
The paired files and statistical receipts live under `analysis/`; each checkpoint pair contributes repeated observations within its root.

## Files

- `environment.py` implements the public task contract and a second Horn interpreter.
- `model.py` implements the tied rule-token transformer.
- `run.py` trains models, executes all policies and writes costed attempts.
- `results/REPORT.md` describes the committed measurements.
- `data/SOURCE.json` pins the DataGoblin source and imported corpus bytes.

Replay every stored action with Python's standard library:

```bash
python -m research.cog03_ledger_router.replay \
  --corpus research/cog03_ledger_router/data/tasks.jsonl.gz \
  --results research/cog03_ledger_router/results
```

The input is pinned to [DataGoblin commit ae0dbec](https://github.com/MoralityLabAI/DataGoblin/tree/ae0dbecffc1d246778ee47d8f3d8ec59b5948f2b).

The source paper collection motivates this experiment: [Cogentic research papers, pinned source](https://github.com/google-research/google-research/tree/4a7bc8ab0bc7/cogentic_research_papers).
The corpus contains generated propositional exercises. Source provenance marks every row accordingly.
