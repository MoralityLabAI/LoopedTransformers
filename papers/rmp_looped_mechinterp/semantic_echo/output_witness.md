# Exact-current-output edge interventions

Status: implemented assay; trained-checkpoint experiment pending. Engineering tests use random initialization and constructed causal controls. This is an extension of the existing Semantic Echo native-state adapter for Q4/Q5 LoopedChase transformers. Existing trained weights, frozen native recurrence and the checkpoint's own head are the experimental target.

## Question and exact observation contract

Can hidden edge-token information change the model's later output while its current output distribution is held exactly fixed?

For one fixed input, let h_u=(E_u,C_u) be the receiver's complete state and h_t=(E_t,C_t) an earlier native state. E comprises all edge tokens; C includes the cursor. Select t,u only when the exact permutation oracle has the same current cursor, both model predictions are correct, and the total-variation distance between the two current output distributions is at most 0.01.

Define h'_u(alpha)=(E_u+alpha(E_t-E_u), C_u), with alpha in {0.25,0.5,1}. The checkpoint reads only C_u at the current loop. Therefore its complete current logit vector is identical before and after the intervention. The implementation requires bitwise equality, including for every random and gauge control. The two original natural endpoints are only approximately distribution-matched; the receiver/hybrid equality is exact.

Continue both states with the same native transition F_e and fixed input embedding e. For future offset k=1,...,4, record

    D_k = TV(softmax R(F_e^k(h'_u)), softmax R(F_e^k(h_u))).

A positive D_k witnesses failure of the current head output to determine the future on this explicitly intervened state domain. Hybrid states may be off-manifold. This assay does not establish that two natural states have identical full output distributions, nor that arbitrary latent directions encode a useful mechanism. The finite edit can destroy computation. Future correctness and control comparisons are essential.

## Selection and analysis frozen before trained-checkpoint evaluation

Default seed 940217; 64 independent random graph/start/edge-order items; 32 loops; four-step continuation; at most two seeded pairs per item in each depth stratum. The complete selected pair list is hashed before interventions. The endpoint gate uses no intervention outcome. Empty coverage is reported as NO_ELIGIBLE_PAIRS with undefined effect estimates; thresholds are never relaxed automatically.

The primary stratum requires the entire receiver continuation to remain within cfg.loops. Cases extending beyond that training budget are depth-extrapolation diagnostics. Existing Q4/Q5 models are a development panel with previously inspected extrapolation outcomes; fresh task draws do not make the model selection prospective. Confirmation needs new checkpoints/seeds.

For each checkpoint, average within independent input item before averaging across items. The descriptive donor-minus-random future-TV statistic gets a 1,000-resample item bootstrap. No interval is emitted with fewer than two eligible items. Pair counts are not independent sample counts. Do not pool all checkpoints as independent loop-pairs, select the best seed or infer a universal trained-model mechanism from one run.

A future TV of 0.01 is a fixed descriptive effect threshold, not a floating-point certificate. Record argmax changes alongside baseline and intervened oracle correctness. A distribution change can preserve, improve or damage task accuracy.

## Controls and measured resources

Native forward and full donor continuation must replay before any effect is considered. A sham edit must leave future states and decisions unchanged. A feature-constant shift on each edge token is removed by the checkpoint's per-token input LayerNorm; any failure beyond the declared tolerance aborts the run.

Eight mean-centered random edge directions per pair use the same raw L2 norm as the donor difference separately on every edge token, at each dose. Raw-norm matching does not guarantee identical effective perturbation after LayerNorm. The receipt reports that post-LayerNorm norm separately; donor-versus-random differences alone are insufficient to attribute a particular feature.

The same random directions are scaled across doses. Changing the cursor is forbidden. Full-state donor replay is a plumbing control and is excluded from substantive intervention effects.

Receipts retain checkpoint/source SHA-256, cfg, seed, raw task inputs, complete selection and coverage counts, per-loop accuracy, future logits/labels, tokenwise edit norms and control errors. All token states are retained during inference, and their storage plus repeated continuation work are part of the assay cost. A single final output token would omit those resources.

## Run on the existing saved Q4/Q5 checkpoints

From the repository root after checking out feat/exact-output-edge-witness:

```powershell
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
python -m research_gym.scripts.loopscope_output_witness --checkpoint-dir experiments/recurrence_monitorability_v0/loopscope_toy/ckpt experiments/recurrence_monitorability_v0/loopscope_toy_q5/ckpt --device cuda --dtype float32 --items 64 --max-loops 32 --horizon 4 --pairs-per-item 2 --random-controls 8 --out experiments/recurrence_monitorability_v0/output_witness_q4_q5
```

Directories must contain *.final.pt files with model/cfg dictionaries and fixed_depth=True. Missing paths, incompatible checkpoints and failed controls are hard errors. Use --checkpoint followed by specific paths to select an explicitly declared panel. Use a fresh output directory per run. An incomplete manifest remains incomplete after failure.

CPU validation:

```sh
python -m unittest discover -s tests -p 'test_loopscope_output_witness.py' -v
python -O -m unittest discover -s tests -p 'test_loopscope_output_witness.py' -v
python -m research_gym.scripts.loopscope_output_witness --smoke --device cpu --items 16 --max-loops 16 --horizon 4 --random-controls 4 --out /tmp/output_witness_smoke
```

The explicit --smoke mode labels every run UNTRAINED_SMOKE, sets the scientific summary to null, and isolates one forced pair as an ungated engineering control. It never substitutes for a missing checkpoint.

## Verification status at implementation

Nineteen tests passed in ordinary and optimized Python using local excerpts of the fetched native model/data and recurrence functions. The source files fetched were model blob 8ef8dbb12726fad18a23573bf6228e65a151023d and adapter blob ca2869dc1bf7a70870c44ba3392d5eeaaf6e529e. Private-repository network downloads were unavailable in the local runtime, so these local test excerpts are not bundled as replacement repository files. The dedicated CI job exercises the complete repository modules.

The local float32 smoke matched native cursor outputs exactly and passed the sham, full-replay and LayerNorm controls. All these tests use random initialization or constructed controls. No trained Q4/Q5 weights were available in the current runtime; no trained intervention result is claimed here. The GPU path remains a checkpoint-run validation target.

## Methodological context

Activation patching depends on the chosen corruption and metric. See Zhang and Nanda, *Towards Best Practices of Activation Patching in Language Models: Metrics and Methods* (ICLR 2024), arXiv:2309.16042, and Heimersheim and Nanda, *How to use and interpret activation patching* (2024), arXiv:2404.15255. The exact output-preserving construction narrows one ambiguity in the current readout while retaining the usual off-manifold and causal-interpretation limitations.
