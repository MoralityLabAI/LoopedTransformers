# SAE causal transfer for looped transformers (LS-SAE01)

This module brings the **SMC-13 through SMC-17** feature-transfer campaign, and the **OLM-14 through OLM-17** causal geometry, into LoopedTransformers/mechinterp. It contains executable physical-SAE intervention planning, a source-indexed negative-result ledger, and an exact-rational loop-depth witness. The current execution is a portable observer and mathematical control gym. **No new pretrained LoopedTransformers/LoopScope/LoopedChase checkpoint result is claimed by this module.**

The key empirical update is [SMC-17 PR #179](https://github.com/MoralityLabAI/Compactification-Theory/pull/179): an independent hosted full/pilot workflow and stdlib audit passed, while its **registered scientific gate passed for 1/4 independently fitted receiving SAE comparisons** after improving all-92-inactive-feature displacement matching. The immediately preceding [SMC-16 PR #177](https://github.com/MoralityLabAI/Compactification-Theory/pull/177) passed 3/4 against simpler four-active-feature-preserving tangent shams. Those decisions and the failed pair conditions are retained in [source_evidence.json](source_evidence.json).

## What can now be run in this repo

- [observer.py](observer.py): accepts a frozen hidden vector and a physical 96×d top-four SAE dictionary. Performs a minimum-L2 target-feature kill while preserving three other active preactivations, then proposes a 4-seed original tangent sham and selects the best inactive-preactivation-matched **same-dose** sham from the original valid baseline, signed unconstrained least-squares tangents and 128 seeded Gaussian tangents. Selection uses encoder geometry and clean feature activation only. Full 96-dimensional displacement traces, rank, residual floor, effect dose, refusal reasons and finite candidate receipts are saved. A separate callback scores the clean and edited task margins after selection is frozen.
- [adapter.py](adapter.py): converts a normalized SAE to physical hidden-state covectors and decoder directions; also provides a deterministic array digest to accompany the original checkpoint-file SHA256.
- [loop_depth.py](loop_depth.py): exact Fraction witness on two weight-tied six-dimensional linear recurrent teachers with identical clean SAE codes and correct task classifications. Their per-loop matched-feature causal effects are **+5k/8** and **−5k/8** at recurrence depth k, including k=0,1,2,4,8. The 92-row inactive measurement system has full column rank and a strictly positive unavoidable matching residual. This is a mathematical construction, not a measured neural feature.
- [tests/test_observer.py](tests/test_observer.py): checks full-rank inactive encoder leakage, normalized coordinate conversion, frozen-control repeatability, active coefficient preservation, source-gate fidelity, opposite-sign loop-gradient effects, and fail-closed cases.

Run at the repository root:

    python -m pip install numpy==2.1.3
    python -m unittest discover -s mechinterp/sae_causal_transfer/tests -p 'test_*.py' -v
    python -m mechinterp.sae_causal_transfer.loop_depth --out runs/ls_sae01_exact_toy.json

The dedicated CI workflow runs the same checks and uploads the exact-rational JSON output.

## Interfacing with actual LoopScope checkpoints

First freeze a checkpoint and its original file-byte SHA-256, model architecture, loop step, source/receiving SAEs, top-four feature correspondence, and calibration feature-assignment hash. Preserve complete calibration and test puzzle manifests. The original SMC teacher used 48-dimensional hidden states, while **LoopScope-02's real TRM** is a different model; its SAE must be independently fitted and source-locked, with physically matching hidden dimension and loop site.

Convert normalized SAE weights into physical hidden coordinates. If z(h) is computed on x=(h−mu)/scale, then the physical encoder has rows W/scale and bias b−(W/scale)mu; physical decoder columns are diag(scale)D. Use:

    from mechinterp.sae_causal_transfer.adapter import to_physical_dictionary
    from mechinterp.sae_causal_transfer.observer import plan_edit, score_margin

    physical_W, physical_bias, physical_decoder = to_physical_dictionary(
        sae.encoder.weight, sae.encoder.bias,
        sae.decoder.weight, train_mean, train_scale)

    edit = plan_edit(hidden_state, physical_W, physical_bias,
                     physical_decoder, matched_target_feature,
                     case_key="frozen-calibration|test-family|puzzle-seed|goal-position")

    if edit["eligible"]:
        effects = score_margin(
            edit, hidden_state,
            continuation=lambda changed_site: frozen_model_goal_margin(
                changed_site, fixed_clean_rival=True))

The caller's continuation must substitute the changed vector into **one predetermined token at one predetermined recurrence step**, hold all other token states unchanged at injection, then replay the frozen model for a preregistered loop budget. It must fix the correct digit and strongest clean non-target rival for evaluation *after* the candidate search. The callback in the code example is schematic and must be implemented for the chosen LoopScope checkpoint. The current portable planner uses NumPy float64; it makes no claim of bit-identical FP32/topk or Torch RNG behavior with the original SMC runner.

For the next LoopScope run, preregister a loop-depth grid and score the same frozen interventions at each unroll depth. Let J_{t→T} be the Jacobian of the frozen continuation and c the clean target-minus-rival readout covector. The first-order margin response of a matched kill delta versus sham s is

\[
D_T\simeq c^\top J_{t\to T}(\delta-s).
\]

More recurrent steps can amplify, cancel or reverse the direction. Compare task-verified margins, flip rates and output gradients across depth and source/receiving checkpoints, preserving puzzle-ID clustering. **The exact linear witness verifies the formula, not the behavior of any trained checkpoint.**

## Scientific and control boundary

The historical sequence must be read with its stronger baselines:

| Experiment | Frozen question | Registered pair verdict |
|---|---|---|
| [SMC-13](https://github.com/MoralityLabAI/Compactification-Theory/pull/171) | Can activation overlap occur across independently trained teacher weights after visible clue/position conditioning? | Observational; Gaussian autoencoding alone yielded 95/96 and 94/96 self-selection cones |
| [SMC-14](https://github.com/MoralityLabAI/Compactification-Theory/pull/174) | Does a frozen cross-checkpoint SAE match predict finite decoder-direction causal sensitivity? | 4/4 |
| [SMC-15](https://github.com/MoralityLabAI/Compactification-Theory/pull/176) | Does erasing a naturally active matched native decoder contribution matter more than an equal-dose other-active erasure? | 4/4 |
| [SMC-16](https://github.com/MoralityLabAI/Compactification-Theory/pull/177) | Does minimum-norm selective encoder feature kill beat a tangent sham preserving the other active features? | 3/4; B501 SAE89 failed coverage |
| [SMC-17](https://github.com/MoralityLabAI/Compactification-Theory/pull/179) | Does the effect survive a 131-candidate sham search matching shifts in all 92 inactive encoder rows? | **1/4**; only B701 SAE89 passed both fresh cohorts |

The [OLM-17 mathematical result](https://github.com/patrickdugan/MathCan/pull/100) shows why perfect inactive displacement matching is generally impossible when the 92×d inactive matrix has full column rank. SMC-17 found rank **48/48 on all 267 selected goals**. The least-squares floor was close to the finite candidate best residual, and more sham refinement alone is unlikely to remove the remaining rank obstruction.

The next native LoopedTransformers study should measure how much of the effect is explained by **the actual task-gradient projection** along the matched edit, and the residual nonlinear response after each recurrent step. Require a task-only matched-budget comparator, **at least 100 rank-matched random observer bases**, corresponding normalised random intervention controls, visible-token/position/digit-conditioned shuffles, fresh solution-family separation and no heldout-label-assisted feature or sham selection. The existing 128 Gaussian sham directions are geometry controls and do not substitute for the independent random-observer-basis baseline.

Successful geometric editing can coexist with a feature that is causally irrelevant to the task, as OLM-16's counterexample shows. SMC-17 receiving teachers had substantially lower Sudoku accuracy than the original pretrained teacher. All measurements are conditional on active receiving features and retained full-SAE reencoding validity. A source audit replay verifies recorded numerical observations rather than independently recomputing trained-model forward passes.

## Related work in this repository

- [LoopScope-02](../loopscope02/README.md): learned TRM hidden-state roles, fiber controls, Sudoku propagation strata and model-specific causal ablations.
- [LoopScope-03](../loopscope03/README.md): three-ledger recurrence, path-independence and cached-state interpolation across frozen looped checkpoints.
- [Phase-gauge07](../phase_gauge07/PROTOCOL.md): label/phase-conditioned controls and source-locked output telemetry.

Use the same heldout split firewall and current LoopScope native checkpoint harness when upgrading this portable observer to trained-network evidence.


## Native LoopedTRM continuation: LS-SAE02

[Native TRM v1 protocol](native_trm_v1/PROTOCOL.md) freezes the first true-checkpoint experiment: [source-hash-locked LoopScope-02 pretrained TRM](../loopscope02/README.md), calibration-only PCA/SAE, 30/34-hole full-solution-disjoint Sudoku cohorts, four-arm minimum-norm feature kills versus 131 inactive-aware sham candidates, and actual recurrent Torch autograd at 0/1/2 subsequent ACT steps. It archives per-goal 48-dimensional gradient projections, all 96 encoder preactivation shifts, finite rank-based p against 128 rank-one random observers, and independent stdlib receipt replay. This experiment adds a *native within-checkpoint test*, with acceptance decided by preregistered fresh-puzzle cluster intervals.

The directory includes executable native checkpoint loader/runner, exact randomized Sudoku generator, SAE fitter, source-evidence audit and CPU tests. From repo root:

    python -m unittest discover -s mechinterp/sae_causal_transfer/native_trm_v1 -p 'test_*.py' -v
    python -m mechinterp.sae_causal_transfer.native_trm_v1.run_native --pilot --out runs/ls_sae02/pilot.json
    python -m mechinterp.sae_causal_transfer.native_trm_v1.audit --input runs/ls_sae02/pilot.json --out runs/ls_sae02/pilot-audit.json

Hosted CI has separate native pilot and full-run jobs. Source/checkpoint accuracy, neural gradients and scientific gate results are reported only after the respective actual run completes and is independently audited.


### LS-SAE02 completed source-locked outcome (2026-11-16)

The frozen native [Actions #43810069425](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43810069425) succeeded through CPU regressions, exact source SHA verification, native pilot/full recurrent gradient measurement and independent standard-library replay. The **predeclared scientific gate failed on both fresh noncanonical Sudoku families**: only 2/32 and 6/32 test goals passed the label-blind full-SAE reencoding guards, with no clean-correct-to-wrong flips. Of all 64 selected goals, 52 failed feature-kill reencoding and four failed the baseline sham. Among the eight admitted goals, every absolute kill-minus-best-sham margin effect fell by more than 99% from immediate readout to two further native ACT steps (42 L-level calls). The observation is descriptive on a small selected sample.

See the full [LS-SAE02 research report](results/LS_SAE02_RESULTS.md), [machine-readable summary](results/LS_SAE02_SUMMARY.json), and exact original raw evidence/audit archive on the linked Actions run. Source checksum, failed bootstrap gates, 128 rank-one random controls and the original SMC-17 comparison remain independently traceable.


## LS-SAE03: native 512-dimensional Jacobian-vector transport

The [prospective LS-SAE03 registration](native_jvp_v1/PROTOCOL.md) retains the source-locked LS-SAE02 pretrained 7M Sudoku TRM and exact 32-puzzle fit-only PCA+SAE, verified against original calibration SHA fingerprints before creating new heldout hidden states. It studies **24 entirely fresh noncanonical Sudoku solutions** (12 per 30/34-hole family) at one input-mask-selected goal each. Actual forward-mode 512D JVPs through 0/1/2 full native ACT updates, forced math attention for correct forward AD, track all 97 token-state tangent norms and original goal-token task-readout slopes. Matched physical source SAE direction, clean predicted-task head, head-null SAE, and one equal-norm seeded random direction are measured under actual ±0.2 native interventions. A separate 128-seed rank-one random observer test uses full-BPTT ground-truth margin gradients.

An independently written standard-library auditor reconstructs 97-token JVP norm partitions, target-token 512D readout inner products, inverse-mode versus forward-mode directional scores, finite symmetric secants, random rank-based empirical p, unique Sudoku rosters and the 2000 puzzle-cluster bootstrap gate. Registered primary event requires substantial task-slope attenuation with preserved overall tangent-state norm, with the source-random physical direction as paired comparator. This design deliberately **does not require full SAE reencoding stability for inclusion**, because LS-SAE02 found that 52/64 test goals failed that gate. LS-SAE02's separate scientific failure remains intact.

Run from the repo root on CPU:

    python -m unittest discover -s mechinterp/sae_causal_transfer/native_jvp_v1 -p 'test_*.py' -v
    python -m mechinterp.sae_causal_transfer.native_jvp_v1.run_native --pilot --out runs/ls_sae03/pilot.json
    python -m mechinterp.sae_causal_transfer.native_jvp_v1.audit --input runs/ls_sae03/pilot.json --out runs/ls_sae03/pilot-audit.json

The dedicated CI workflow performs the same pinned-checkpoint pilot, then the complete 24-puzzle heldout study with independently replayed numerical receipts. Its scientific result is reported only when those jobs complete.


## LS-SAE04: sealed-source full 512D Jacobian result

[LS-SAE04 PR #121](https://github.com/MoralityLabAI/LoopedTransformers/pull/121) separated 32 fit-only source calibration Sudoku puzzles from both pilot and full native JVP cohorts, archiving a byte-sealed 512→48 PCA / 48→96 top-four SAE NPZ **before any new test hidden states**. Three distinct GitHub Actions jobs loaded and independently verified the identical source artifact, fixing LS-SAE03's cross-host calibration fingerprint reproduction blocker.

The hosted [full native run #43818817318](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43818817318) passed all four CI stages and its standard-library numerical replay. The **preregistered scientific result was negative on both independent 12-puzzle Sudoku families**: 24/24 SAE feature directions were admitted, and 24/24 showed >90% task-margin slope attenuation after two native ACT steps, while 0/24 retained the required full 97×512 tangent norm >=0.25. Median full-state Jacobian gain was 0.062 (30 holes) and 0.047 (34 holes), with only about 0.3–0.4% of the *remaining* tangent energy at the edited token. Similar contraction affected random and clean task-head controls.

This distinguishes **directional generic recurrent contraction** from the original proposed preservation of a task-silent but strongly surviving SAE state perturbation. These measurements concern selected directions and do not certify the full recurrent Jacobian operator norm.

See [complete results](results/LS_SAE04_RESULTS.md), [machine-readable summary](results/LS_SAE04_SUMMARY.json) and the [preheldout protocol](native_jvp_v2/PROTOCOL.md). The original full raw, calibration-source NPZ, manifest and independent audit remain in source-locked Actions artifact #14204410412.


## LS-SAE05: prospective singular-direction escape from contraction

[LS-SAE05 registered protocol](native_singular_v1/PROTOCOL.md) tests whether the full two-ACT-step 512→(97×512) Jacobian has a **narrow amplifying singular direction** that the source SAE, native task-head and isotropic physical probes in LS-SAE04 missed. Its matrix-free runner uses four **native forward JVP plus reverse VJP power updates** at the unchanged pretrained checkpoint, without using Sudoku target labels or task gradients for singular-direction selection. It then scores full-state gain against eight source-seeded random JVP controls, with separate SAE and clean task-head comparisons, clean digit-margin readout and ±0.05 finite secants.

The experiment uses 16 new exact-unique Sudoku solutions in two independent 8-puzzle families, all disjoint from LS-SAE02/03/04. It downloads the **original LS-SAE04 preheldout calibration-only artifact** from [successful CI #43818817318](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43818817318) and byte-verifies that same original NPZ and manifest. A source-independent standard-library auditor reconstructs the four JtJ update transcripts, 97-token tangent energy, task-head readout, eight-control gains and the prospectively registered puzzle-cluster bootstrap. A positive gate would identify a narrow local state-amplifying channel; a negative gate constrains this exact four-step search, not the full Jacobian operator norm.


### LS-SAE05 source-locked result (2026-11-16)

The native [Actions #43823264749](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43823264749) completed the CPU synthetic JtJ regressions, original 7M pretrained checkpoint engineering pilot, full **8 fresh 30-hole + 8 fresh 34-hole** Sudoku test, and independent stdlib replay. The **preregistered scientific singular-gain gate passed in both fresh families (8/8 positive events each)**. Per-puzzle four-update matrix-free JVP/VJP power directions have median full-state gains **0.7115/0.6925** after two complete ACT steps, versus median LS-SAE04-style frozen source SAE gains **0.0606/0.0492**, and exceed the corresponding best of eight random fixed directions by median **10.89×/11.15×**. Paired puzzle-bootstrap lower 95% bounds for the continuous power-minus-best-random gain advantage are **+0.559/+0.506**, with original registered thresholds unchanged.

These high-gain source directions are selected **per clean heldout input using the Jacobian norm objective**, without task labels. By median they retain only **0.63%/0.35%** of their output tangent energy on the original edited goal token, and their correct-digit task-readout slopes remain small. The result identifies narrow **finite-horizon structural amplification**, independent of any claim that a sparse feature uses these directions to solve Sudoku. It reconciles LS-SAE04's observed contraction along the source SAE, clean task-head and random directions with the existence of locally optimized amplifying directions.

[Original research report](results/LS_SAE05_RESULTS.md), [machine-readable source-replayed summary](results/LS_SAE05_SUMMARY.json), [prior spectral mathematics](results/LS_SAE04_SINGULAR_BOUNDARY.md), and [complete original numerical and audit archive #14206645427](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43823264749/artifacts/14206645427).


## LS-SAE06: complete 81-cell native Sudoku causal task transfer

The [prospective LS-SAE06 protocol](native_task_transfer_v1/PROTOCOL.md) tests whether LS-SAE05's **label-blind high-gain full-state Jacobian directions** causally alter digit predictions across the entire Sudoku board. It uses 8+8 new noncanonical independent 30/34-hole Sudoku solutions, source-frozen pretrained 7M attention TRM checkpoint SHA and the byte-verified original LS-SAE04 calibration-only 48→96 sparse autoencoder NPZ. The source intervention is one normalized 512-dimensional z_H board token immediately after the first ACT step.

A computation-matched **centered 81×9 logit-sensitive power search** runs four actual JVP/VJP updates, versus LS-SAE05's four-update full-state power search. An original SAE decoder and a seeded unit isotropic direction are independent physical controls. All arms use identical ±0.05/±0.20 input-state edits, with actual native forward recurrence to ACT3 and ACT6. The numerical receipts preserve every one of the 81×9 logits before scoring missing-cell digit accuracy, prediction flips, unit conflicts, exact solved grids, original givens and signed correctness transitions. A dedicated standard-library replay independently recomputes full-board numerical effects, the exact Sudoku validity certificate and puzzle-ID bootstrap.

The preregistered primary outcome is whether the structural singular direction changes at least one more *missing-cell predicted digit* than the equal-dose random direction on at least six of eight puzzles, with a positive paired-cluster bootstrap lower bound on both fresh families. It is explicitly distinct from a claim of improved correct Sudoku solutions. The whole81 task-logit power direction is an equal-compute comparator and all original label-blind source directions are selected before reading the correct solution digits. The source-locked GitHub Actions workflow runs CPU unit tests, separate 2+2 engineering-only pilot, full fresh 8+8 native evaluation and independent numerical receipt audit.

Run from repo root:

    python -m unittest discover -s mechinterp/sae_causal_transfer/native_task_transfer_v1 -p 'test_*.py' -v
    python -m mechinterp.sae_causal_transfer.native_task_transfer_v1.run_native --pilot --source-dir runs/ls_sae06/source --out runs/ls_sae06/pilot.json
    python -m mechinterp.sae_causal_transfer.native_task_transfer_v1.audit --input runs/ls_sae06/pilot.json --out runs/ls_sae06/pilot-audit.json

The source directory must contain the original SHA-sealed LS-SAE04 calibration-only NPZ and canonical manifest. Native scientific evidence is published only after the full run and independent replay complete.


### Completed LS-SAE06 source-locked whole-board outcome

The registered [LS-SAE06 full native Actions #43827999742](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43827999742) completed all three CI stages, including a source-original pretrained checkpoint pilot, **8+8 fully fresh unique noncanonical 30/34-hole Sudoku solutions**, and independent stdlib replay of all 81×9 digit logits with signed edit arms, exact clue/unit Sudoku checks and original 2,000 puzzle-cluster bootstrap. **Both original scientific primary gates failed: 0/8 positive missing-cell prediction-flip events per family**, with paired 95% intervals [0,0]. Across every source arm, sign, radius and ACT3/ACT6 horizon, all 81 discrete digit predictions remained unchanged; ACT6 clean Sudoku exact board solve was 8/8 in each fresh family.

The continuous centered digit-logit effects were nevertheless substantial. At ACT6 under ±0.20 source input dose, mean full-board centered-logit Frobenius change per dose was **0.399/2.128** for four-update full-state STATE_POWER (30/34 holes), versus **0.025/0.037** for random; the equally budgeted centered-output LOGIT_POWER yielded **0.398/2.132**. The original source directions from these independent JtJ searches have median absolute cosine **0.9948/0.9975**, directly showing that the structural high-gain channel couples to all-board logit sensitivity even when the original edited goal's task derivative is small. Strong clean winner margins protected the output classifications: the largest finite per-cell perturbation-to-winner-gap upper-bound ratio was **0.00074/0.00811** at ACT6, satisfying the exact finite argument in [LS-SAE06 mathematical companion](results/LS_SAE06_TASK_GAP_BOUNDS.md).

See the [audited complete results](results/LS_SAE06_RESULTS.md), [machine-readable summary](results/LS_SAE06_SUMMARY.json), and original raw full-81-logit [CI artifact #14209000375](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43827999742/artifacts/14209000375). These 30/34-hole puzzles were all solved by ACT6, creating a task-accuracy ceiling. Correctness-transfer experiments now need more discriminating source-locked hard Sudoku families and fresh prospectively registered doses.


## Harder unique Sudoku: LS-SAE07 and LS-SAE08

The [LS-SAE07 original prospective protocol](native_hard_task_v1/PROTOCOL.md) froze 52-hole and 54-hole source grids and stronger physical edit doses before generating any new model states. Its 54-hole full family had a source-only admission obstruction: original full-grid seed 202627040006 can be dug to at most 53 holes while preserving the exact unique-solution requirement. LS-SAE07 is kept as a source-admission failure, with a separate [source-only admission runner](native_hard_task_v1/admission.py) and unmodified original protocol. It is **not** a negative neural scientific experiment.

The new separately registered [LS-SAE08 prospective protocol](native_hard_task_v2/PROTOCOL.md) fixes a deterministic **first model-blind admissible candidate** among 32 consecutive full-grid seeds per puzzle slot. It builds 8+8 fresh unique-solution 52/54-hole hard Sudoku inputs from noncanonical solution families, with independent 2+2 engineering pilot. All candidate decisions depend only on Sudoku uniqueness and the visible puzzle, and the source roster is frozen before the original 7M pretrained checkpoint is evaluated. The source-only audit preserves each rejected candidate and original full-grid seed.

Native physical directions and inference reuse the exact preregistered LS-SAE07 approach: four-update full-state STATE_POWER, four-update centered all-81-logit LOGIT_POWER, frozen original SAE decoder and SHA-seeded unit random control. Every direction gets both signed physical edit radii 0.20, 2.00 and 8.00, evaluated at **ACT3, ACT6 and ACT12** on complete original 81×9 digit logits, with exact missing-cell correctness, original clues and 27-unit Sudoku validation. The unchanged primary test is the ACT6 ±8.00 paired blank-prediction-flip advantage over random, ≥6/8 cases plus positive 2000 puzzle-ID bootstrap lower bound in BOTH new 52/54-hole families. Separate accuracy gains and delayed recovery are reported without changing the scientific gate.

The dedicated LS-SAE08 workflow includes source-only uniqueness and previous failure checks, pinned original calibration NPZ, the engineering-only pilot, full fresh 16-puzzle original model evaluation, and independent stdlib numerical audit.
