# LS-NEURON-01: recurrent MLP pruning (development pilot)

Status: **REDESIGN_OR_PILOT_ONLY**. Source: [hypothesis post](https://x.com/bunniemoder/status/2125426825509953674). Consulted [EvalMath preflight](https://github.com/MoralityLabAI/EvalMath/tree/main/skills/evalmath-mechinterp-power) and [FSI-01](https://github.com/MoralityLabAI/EvalMath/blob/main/paper/finite_sample_inference.md). No sealed readout opened; no paid compute.

## Methods and provenance

Run `python pilot.py --out results.json` and `python zero_control.py` (PyTorch 2.10.0+cpu, one CPU thread, deterministic algorithms). Nine checkpoints, 240 AdamW updates, seeds 11/23/37. Each seed uses 4096 training graph draws, 256 calibration draws, and 512 development draws with one reachable and one unreachable target per graph. The oracle is four-step Boolean reachability on a nine-node directed predecessor graph; the source itself is excluded as positive target. Draws are with replacement. Targets and interventions within a graph are correlated, and the three seed-matched checkpoint groups are the independent architecture-replication units. Attention and normalization weights are tied in ALL arms. Only MLP tying and width vary. This is a graph-constrained toy transformer, not a language model or the existing TRM checkpoint. Source graph and train/test seed offsets are in code. No confirmation data or trained checkpoint archive is claimed.

Same-width tied/untied models have widths 64 and 64; approximately parameter-matched untied has width 16. Parameter counts 3329/9713/3377. All use the same four steps and per-seed train batches. Structured activation masks zero whole GELU channels, with output bias retained. Shared-bank salience is the sum of absolute activation times loss gradient across all calibration roots, node positions and uses of that bank. Local masks select per-step salience instead. Four random masks per bank serve as exploratory controls; masks are reused across depth and dose where appropriate. Rounding and a minimum one kept neuron mean dose labels are NOMINAL. At 95%, 64-wide banks keep 3 and 16-wide banks keep 1. At 99%, both keep 1. No wall-time speedup or actual sparse-kernel dispatch measured.

## Results (three-seed mean accuracy, %)

| Nominal pruned | Tied salience | Untied same-width salience | Untied approx-budget salience | Tied random | Untied same-width random | Untied approx-budget random |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 100 | 100 | 100 | 100 | 100 | 100 |
| 50 | 100 | 100 | 100 | 99.46 | 82.10 | 99.19 |
| 75 | 100 | 98.93 | 100 | 82.53 | 80.10 | 98.66 |
| 90 | 98.93 | 81.64 | 98.93 | 81.99 | 79.59 | 98.12 |
| 95 | 98.93 | 79.59 | 98.93 | 86.43 | 79.09 | 97.85 |
| 99 | 82.26 | 77.57 | 98.93 | 82.26 | 77.57 | 97.85 |

All local salience mask accuracies equal the shared-bank salience accuracies at reported doses. This does not establish mask equivalence or identical selected neurons. Seed 11/23/37 tied versus same-width untied accuracy differences at 95%: +1.86/+9.38/+46.78 percentage points (mean +19.34); strong between-checkpoint variation. Approx-budget arm matches tied mean at nominal 90 and 95%. Compare equal per-use counts AND equal unique retained neuron banks before interpreting width scaling.

### Complete MLP removal control (retrained deterministic replay)

| Condition | Tied | Same-width untied | Approx-budget untied |
|---|---:|---:|---:|
| Dense | 100 | 100 | 100 |
| Zero neurons, bias retained | 82.26 | 77.57 | 97.85 |
| Zero entire MLP, including bias | 81.18 | 77.57 | 97.85 |
| 1 neuron per use | 82.26 | 77.57 | 98.93 |
| 2 per use | 82.26 | 79.59 | 98.93 |
| 3 per use | 98.93 | 79.59 | 100 |
| 4 per use | 98.93 | 79.59 | 100 |
| 6 per use | 98.93 | 81.64 | 100 |
| 8 per use | 98.93 | 98.31 | 100 |
| 16 per use | 100 | 98.93 | 100 |

The matched-budget MLP is largely dispensable on this easy graph family. The tied model retains a meaningful descriptive recovery over complete MLP removal, but near-perfect attention-only performance in the budget arm blocks a mechanistic claim of superior neuron sparsity due to tying. This intervention does not remove attention. It is not a full-network pruning experiment.

### Depth and single-neuron diagnostics

| Inference steps | Tied dense | Tied 90% | Tied 95% |
|---|---:|---:|---:|
| 1 | 69.30 | 78.81 | 78.81 |
| 2 | 91.24 | 91.24 | 91.24 |
| 4 | 100 | 98.93 | 98.93 |
| 8 | 94.95 | 96.45 | 81.97 |

The target remains four-hop reachability even when inference depth changes, so eight steps can propagate beyond the target horizon. No depth compensation claim. For each tied checkpoint, rank eight neurons using calibration salience and zero one at a time at each step. Calculate adjacent-step cosine of the final positive-minus-negative logit-margin effects over calibration graphs. Mean cosine by seed 0.6810, 0.6682, 0.6357; 63/72 positive, minimum about -0.8981. This is selected-neuron, within-checkpoint, calibration-only diagnostic with no matched-direction transport sham and no semantic labels. The shared output direction is fixed by construction.

## EvalMath gate and follow-up (proposed, not registered confirmation)

Unit: paired training seed/checkpoint for architecture generalization; graph root for fixed-checkpoint paired task effects. n_units=3 checkpoint pairs per contrast, 512 development graph draws per seed, two targets per draw. n_cases=3072 target decisions per architecture per intervention across seeds. Primary descriptive contrast: tied minus same-width untied dense-relative accuracy at nominal 95% MLP pruning (3/64 retained), +19.34 points. Other doses, masks, steps, absolute counts and calibration cosine are exploratory. 351 original sweep rows (mask/depth/checkpoint combinations); 9 additional checkpoint replays with complete MLP removal and seven absolute-count conditions. Multiple comparisons not prospectively frozen. Scientific floor for proposed follow-up: 5 percentage points beyond the best capacity-matched baseline, conditional on at least 10 points of task headroom above no-MLP control. No reliable matched-sham floor or power estimate yet; four random masks per bank are insufficient for tail calibration. Use independent development roots and >=59 sham worlds for a 95th-percentile 95%-confidence maximum-bound check. Estimate seed-level variance, run EvalMath preflight, plan independent checkpoint count (at least 20 for the standard FSI gate and potentially more), freeze comparisons and seal fresh task families before confirmatory runs. Do not use repeated token, dose or neuron counts as independent replicates.

1. Move to MLP-necessary Boolean/compositional tasks from the existing NEC04 series, plus trained TRM Sudoku; verify attention-only control fails enough to give headroom. Include full-block untied baselines and matched-parameter and matched-compute arms separately.
2. Compare equal unique retained neurons, equal active neurons per application, and equal total multiply-adds; use output-norm/dose-matched random masks, activation and magnitude ranks. Keep whole MLP, bias-only and residual/attention ablations.
3. Test whether the same surviving channels mediate correct reasoning through independent, held-out counterfactual task pairs. Compare loop-specific mask swaps and norm-matched random directions; distinguish local output-vector invariance from downstream causal transfer.
4. Only estimate a depth-width-precision frontier after robustness passes both matched baselines and the mechanistic floor. Separate post-training pruning from retrained sparse model performance and measure actual compact-kernel latency.

Evidence class: development-only trained-model observation, power-limited; no support yet for general loop-specific sparsity beyond the matched-budget alternative or semantic stability.

## Local execution receipt SHA-256

- pilot.py: b40745da4a22042a793c566a68e685b048f8d01b7f6891fbbdb960cc96a50ad7
- zero_control.py: 555a36d4b937e2697b439cc1ca603adc322baa02948849a13e3e490f3fcb6170
- results.json: 5003db38825fa09eb4a72c4c395b7c96f65acbfcc09334be9221469598e7eed2
- zero_control_results.json: 4877225635e7fcd90992e166d193ee1f94b62d24af5577428159d33a816b9181

Raw root outcome receipts are in the local downloadable experiment archive; GitHub contains source and this report for reproduction. Source Git blob IDs independently matched local files: e8151022a8a9bef8c04b7b9d106ef1790c0f628a and 835943ee46f4210d9bb87068e073793a67eb6104.
