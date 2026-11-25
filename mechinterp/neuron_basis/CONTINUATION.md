# NBR continuation ledger (2026-11-25)

PR #351; do not merge until engineering CI and receipt review. This is a
continuation of PROTOCOL.md, not a prospective scientific registration.
Original native toy checks passed Actions run 45514235253. Extended suite
and pretrained checkpoint pilot outcomes are pending at time of writing.

| ID | Implementation and outcome boundaries |
|---|---|
| NBR-01 | native.py: ACT3/6/12 same-ID stability, norm-stratified shuffled controls, static head residualization, board bootstrap. Real checkpoint pilot in CI. |
| NBR-02 | suite.py: 64 direction candidates per arm (or smaller on toy), random fixed subset of native columns, PCA/random basis, undercomplete unit-decoder SAE trained on calibration tokens; transported write, activation-only and static-head controls. Class-degenerate recovery cohorts report undefined AUROC. Full prospective comparison still needs larger rosters, an overcomplete SAE/budget curve, shuffled-label and supervised-probe baselines, PR curves and calibration diagnostics. |
| NBR-03 | L0 to L1 same-token Jacobian of selected activations and central finite-difference check; local edge validation only. Multi-cycle graph recovery and shuffled-edge predictive controls remain future gates. |
| NBR-04 | Calibration-selected directions, signed equal-L2 residual edits 0.02/0.25, same-forward clean and sham, local linear prediction, ACT16 verified schedule and signed outcome. Numerical stability across hardware remains untested. |
| NBR-05 | Original and alternate verified checkpoint at FP32; simulated symmetric per-row signed 4-bit weight rounding on original. This tests precision sensitivity and same-architecture checkpoint replication, not architectural scaling or compressed runtime performance. Sealed multi-size comparison needs additional checkpoints. |
| NBR-06 | ACT6 label-free features selected on calibration boards to predict ACT16 exact-board failure, evaluation AUROC and paired bootstrap; class coverage reported. |
| NBR-07 | Six synthetic call-local adapter cases with planted digit-target column and decoy columns; transported weight-informed, static and observed-output guesses; norm-matched random adapters as negative controls. Only a benign engineering analogue; actual hidden finetuned behavior and reward hacking require real labeled adapters and a separate blinded test. |
| NBR-08 | Top eight native neurons by calibration contribution magnitude, zeroed during ACT7..16; equal-count, approximately write-column-norm-matched random neuron ablation; record removed energy and deduction outcomes. This tests selective ablation, not physical pruning, resource savings, or retraining. |

## Run all engineering stages locally

    python -m mechinterp.neuron_basis.native --self-test
    python -m mechinterp.neuron_basis.roster --count 2 --out nbr_roster.json
    python -m mechinterp.neuron_basis.suite --roster nbr_roster.json --out nbr_toy --toy
    python -m mechinterp.neuron_basis.audit --roster nbr_roster.json --out nbr_toy
    python -m mechinterp.neuron_basis.native --roster nbr_roster.json --out nbr_native
    python -m mechinterp.neuron_basis.suite --roster nbr_roster.json --out nbr_fp32
    python -m mechinterp.neuron_basis.audit --roster nbr_roster.json --out nbr_fp32
    python -m mechinterp.neuron_basis.suite --roster nbr_roster.json --out nbr_w4 --precision w4
    python -m mechinterp.neuron_basis.audit --roster nbr_roster.json --out nbr_w4
    python -m mechinterp.neuron_basis.suite --roster nbr_roster.json --out nbr_rep --checkpoint yagizdevre
    python -m mechinterp.neuron_basis.audit --roster nbr_roster.json --out nbr_rep

CPU-only CI runs toy FP32/W4 first, then three real checkpoint jobs, each
on the same deterministic six-board engineering roster. The original
checkpoint job runs native.py before suite.py. Artifacts include the roster,
raw receipts, frozen basis, and report. CI is an engineering run; it does
not evaluate a scientific gate. Do not select parameters using its results
and then report those same boards as held-out.

## Remaining prospective gates, in order

1. Inspect CI and raw receipts; repair engineering problems without changing
   scientific claims. Keep failed receipts. Measure actual runtime, memory
   and class coverage. Add independent tests for source solver admission,
   invalid grids, derivative accounting and sham.
2. Source-only admission: choose >=48 boards per calibration/eval/rep
   family, with independent unique-solution verification, no previous
   exact input/solution overlap or symmetry overlap, hard-family and
   calibration-class coverage plan. Freeze exact hashes/roster before
   model access. The deterministic generator's distinct symmetry
   invariants exclude within-roster orbit matches; it has not been checked
   against historical sources. Generated puzzles may be easier than
   Champagne hard families; stratify by a model-blind difficulty measure.
3. Freeze code, model/platform hashes, one primary contrast per test,
   multiplicity policy and held-out analysis. Split calibration fitting
   and held-out execution into separate commands; seal directions and
   hyperparameters before opening eval. Existing engineering code collects
   eval traces in the same process but does not use them in fitting.
4. Run NBR-01/02 and numerical replays; apply the proposed registered gate
   only after a prospective registration commit. NBR-02 additionally
   needs the full comparison controls in PROTOCOL.md.
5. Causal and graph work: finite-difference tolerance and same-batch sham
   gates first; compare native effects to matched random and SAE, then
   verify multi-edge multi-step circuits. A large VDD can be destructive:
   report signed schedule shifts and final correctness alongside it.
6. NBR-05: obtain architecture/size-controlled checkpoints and freeze
   matching policy. NBR-07: obtain genuine independently labeled adapter
   family and blind its held-out behavior descriptions. NBR-08: implement
   physical pruning and wall-clock/accuracy evaluation if causal ablation
   supports it. Do not generalize from the toy adapters or W4 rounding.
7. Publish the engineering and scientific verdicts separately, preserving
   failures and class-degenerate results. No main merge without review.

## Continuation instruction

Inspect PR #351 head, Actions and receipts. Repair failing engineering
checks and independently recompute outcomes. Advance the remaining gates
in ledger order, preserving registration and failed results; mark each
gate PASS/FAIL/INCOMPLETE with a commit and receipt. Do not change thresholds
or reclassify the pilot as held-out. Stop at missing checkpoint, compute or
source admission and state the blocker. Update the ledger before ending.
