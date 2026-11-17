# LS-SAE05: source-locked leading singular directions reveal hidden transient amplification

**Executed November 16, 2026.** Applied mechanistic-interpretability result in [LoopedTransformers PR #125](https://github.com/MoralityLabAI/LoopedTransformers/pull/125). The [prospective registration](../native_singular_v1/PROTOCOL.md) was committed at **36009cb071ab9d283cfae86d40f7e46cfbc608a6** before any LS-SAE05 test hidden state. Hosted [Actions #43823264749](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43823264749) passed synthetic CPU unit tests, the pretrained 2+2-puzzle engineering pilot, the complete fresh 8+8-puzzle native experiment, and its independent standard-library numerical audit.

**Preregistered scientific result: PASS on both fresh Sudoku families.** The result is structural, measuring high-gain directions in a real trained recurrent hidden-state Jacobian; a per-puzzle power search is optimized for the JVP norm itself, with the fixed controls and computation budget disclosed.

## Original registered primary result

On each unique new 30/34-hole Sudoku puzzle, choose the median blank by visible input mask, inject a unit 512D source z_H perturbation after ACT step one, and evaluate the full 97×512 z_H tangent after two additional complete ACT steps. Four source-label-blind matrix-free JᵀJ power updates select a candidate high-gain input direction. The primary gate requires its gain ≥0.25 and ≥1.20× the best of eight independently seeded isotropic random JVP directions, on at least six of eight puzzles per family, with a positive 95% puzzle-bootstrap lower bound for the paired continuous gain advantage.

| Frozen outcome | New 30-hole | New 34-hole |
|---|---:|---:|
| Fresh independent puzzles / selected goals | 8 / 8 | 8 / 8 |
| Eligible power directions | 8 / 8 | 8 / 8 |
| **Registered high-gain events** | **8 / 8** | **8 / 8** |
| Mean power gain minus best-of-eight random | **+1.180249** | **+1.267200** |
| Paired 2,000-draw puzzle bootstrap 95% | **[+0.559056, +2.087861]** | **[+0.505860, +2.447974]** |
| Registered family gate | **PASS** | **PASS** |

All eight paired power-minus-random differences are positive in each independent family. The empirical intervals are descriptive over the fixed seeded source puzzle benchmark. The source-fit weights, high-gain threshold, four power updates, eight random JVP controls and full test puzzle IDs were frozen before seeing the test results.

## Strong local amplification, concentrated outside the target readout

| Secondary geometry, medians per 8-puzzle family | New 30-hole | New 34-hole |
|---|---:|---:|
| **Power-selected full JVP gain** | **0.7115** | **0.6925** |
| Source SAE physical decoder JVP gain | 0.0606 | 0.0492 |
| Clean predicted task-head JVP gain | 0.0225 | 0.0140 |
| First seeded random JVP gain | 0.0472 | 0.0279 |
| Power gain / best-of-eight random gain | **10.89×** | **11.15×** |
| Power output energy fraction on selected goal token | **0.00628** | **0.00350** |
| Absolute power-direction true-digit margin slope | 0.02518 | 0.00914 |
| Power-direction absolute task/readout cosine | 0.0904 | 0.0503 |

The largest single power-selected gains are **4.4561** on the 30-hole family and **5.5915** on the 34-hole family. Across the 16 puzzles the minimum power gain is **0.4031**; the largest best-of-eight random gain is only **0.5078** (on a different 34-hole puzzle), and the paired power direction still exceeds that strongest corresponding random by over 11×.

The selected high-gain directions are task-label-blind and optimized for full-state transmission. Only roughly **0.35–0.63% of the surviving full JVP energy** remains on the original edited goal token at depth two, by median. Their task-margin derivatives remain small relative to their full-state gains, with weak direct task-readout alignment. This identifies a **transient high-gain hidden-state channel** whose downstream influence is carried predominantly by other tokens, while providing limited direct evidence for correct-digit task reasoning.

## Four-step matrix-free power convergence

The exact source code implements a JVP of the true frozen F₂:R^512→R^(97×512) recurrent continuation, followed by a VJP with its detached full 97×512 tangent as cotangent. It normalizes the resulting 512D JᵀJu exactly four times. The independent audit replays each saved 512D normalized power input and full reverse cotangent, checks u·(JᵀJu)=||Ju||² and checks successive normalized directions.

The **median** full JVP gain across the eight test puzzles at the four original power-search iterations progressed:

| Power iteration, pre-normalization state | 30-hole | 34-hole |
|---|---:|---:|
| Initial seeded isotropic direction | 0.0343 | 0.0188 |
| After one JᵀJ update | 0.6980 | 0.6629 |
| After two updates | 0.7113 | 0.6924 |
| After three updates | 0.7115 | 0.6925 |
| Scored after four updates | **0.7115** | **0.6925** |

Most of the observed gain emerges after the first matrix-free update. Larger per-puzzle amplification channels can persist despite the small response of many physically plausible fixed probes.

The maximum absolute source-model forward-JVP versus reverse-mode true-task-margin slope discrepancy for the power direction across each family is approximately **2.65×10⁻⁷** and **1.46×10⁻⁷**. Median absolute difference between an independently calculated ±0.05 finite physical margin secant and its power JVP task derivative is approximately **1.79×10⁻⁴** and **1.50×10⁻⁴**. These checks validate the internal numerical derivative arithmetic to the reported precision; a second independent neural forward execution and interval-certified real-arithmetic guarantee have not been performed.

## Provenance, original source and independent receipt replay

The unchanged original pretrained network is **gaoxin492/TinyRecursiveModels-Sudoku-Extreme-att**, file step_21700, with native source-checkpoint SHA-256 **932b5e0df4e90021c21a064d4e716154086c46fda7afa676ea294484a30e38c3**.

The frozen calibration was downloaded from the **original LS-SAE04 source-fit-only** [Actions artifact](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43818817318/artifacts/14203533086) and required its original full NPZ file SHA-256 **073b7771676e44eb934c50bdf1e7639ffeadbdff12998979cfe4c43e42b2da25**, fit-array SHA **83dacde8f13926fc41fc48cb17bcfba8c399538754efb7104e41e94bd93797ec**, and calibration manifest SHA **cb44b910f94210745b7a7df304e575d5a194ee8cd7bec51a88281fccc95bf33a** before any LS-SAE05 source evaluation.

Fresh solution/puzzle SHA-256 manifests:
- Pilot 30-hole: **1afe22a8f29bcedc1bf41b2172d26ef543df5eb5e277ddf178a5867cca289123**.
- Pilot 34-hole: **47d52976e7afb1e74260c9ef136d206f0000e7555a2a33651f574c24d09f36a0**.
- Registered full 30-hole: **d3ec025c2aefbdef08360126a95f29ecbcd67c8818a54449037c1007867b15ec**.
- Registered full 34-hole: **c9d6144699088ceb5f41c3c3fde0eb289987797319a19e92a7b9f31a8375b138**.

- Full original [Actions run #43823264749](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43823264749), archive [#14206645427](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43823264749/artifacts/14206645427).
- Original artifact ZIP SHA-256 **ba708ec4b5963f239ae75813c5c1fb05713035075c50048e415b631a1c7a4ac5**.
- Original **10,894,480-byte** full raw JSON SHA-256 **292a36d9833f422f8b88281c5cb22acf4a91424ac0d3995cca1db507**.
- Independent stdlib audit SHA-256 **e6f7ef37577038ffb554126ad789e3e2ae4e5e47dc10c1db4ac7c065d7cc7d89**.
- [Machine-readable full result summary](LS_SAE05_SUMMARY.json).

The audit validates source SHA-256s, fresh exactly solvable Sudoku manifests and median-blank selector, 512D physical input vector norms, full 97-token JVP energy, task-head readout arithmetic, every archived JᵀJ power update's VJP norm and normalized vector, eight precommitted random gains, finite margin secants and 2,000 cluster-bootstrap gate. This is an independent numerical receipt replay. It does not perform a separate independently written neural model evaluation.

## What the combined LS-SAE04 and LS-SAE05 evidence says

LS-SAE04 discovered strong contraction along learned source SAE, clean task-head and isotropic fixed directions, with the full-state gain of the source SAE direction typically around 0.05 after two ACT steps. LS-SAE05 used a pre-registered **four-update matrix-free singular search** to find a very different direction with median gain around 0.7, sometimes greater than 5, on genuinely new puzzles from the same pretrained network. Both findings hold under their separate prospectively registered source and puzzle splits.

This establishes a **highly anisotropic finite-horizon recurrence Jacobian** under source-state perturbation. Typical fixed probes contract; narrow structural directions amplify and redistribute most surviving influence outside the input goal token. Because the power search maximizes the state-output norm at each test puzzle, these directions have not been shown to be naturally used by a trained SAE or essential for Sudoku correctness. Follow-up should test whether the high-gain directions predict future error correction, represent a stable symmetry, or simply lie in nuisance state channels. Any such task-use claim needs another prospective evaluation with fresh puzzles and equal-budget task-only controls.
