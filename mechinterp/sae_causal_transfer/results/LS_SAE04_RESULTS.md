# LS-SAE04: full 512D native JVP reveals generic recurrence contraction

**Executed November 16, 2026.** [LS-SAE04 PR #121](https://github.com/MoralityLabAI/LoopedTransformers/pull/121). The prospective [source-fit-sealed registration](../native_jvp_v2/PROTOCOL.md) was committed at **9e4aa45635b3a2dc6358926aac3ae10e54aaa9d3** before any pilot/test hidden state. [Hosted CI run #43818817318](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43818817318) passed all four dependent jobs: numerical unit checks, calibration-only fitting and binary sealing, isolated pilot, and full 12+12 fresh Sudoku native forward-JVP evaluation with independent stdlib replay.

**Registered scientific result: FAIL for both fresh families.** The frozen event required simultaneous large initial task sensitivity, >90% loss of that sensitivity by depth two, and retained full 97×512 hidden tangent norm at least 0.25, outperforming a fixed equal-norm isotropic direction under a paired puzzle bootstrap. Every one of the 24 SAE directions was admitted; all 24 met the initial-sensitivity and attenuation thresholds, but **zero** retained global tangent norm >=0.25 at depth two. The random-direction arm likewise had zero qualifying joint events.

## Registered primary outcomes

| Frozen primary measure | Fresh 30-hole | Fresh 34-hole |
|---|---:|---:|
| Independent new puzzles / selected target cells | 12 / 12 | 12 / 12 |
| Source-frozen SAE direction admitted | 12 / 12 | 12 / 12 |
| SAE events: sensitivity lost AND signal preserved | **0 / 12** | **0 / 12** |
| Equal-L2 isotropic random events | **0 / 12** | **0 / 12** |
| Puzzle-paired event fraction difference | 0.0 | 0.0 |
| Fixed 2,000-draw puzzle bootstrap interval | [0.0, 0.0] | [0.0, 0.0] |
| Registered both-family criterion | **FAIL** | **FAIL** |

The registered gate required at least ten SAE-admitted puzzles, eight positive SAE events, and a strictly positive paired bootstrap lower bound in each 12-puzzle family. Neither family passes. No threshold, puzzle, radius, source vector or analysis denominator was retuned after testing.

## Which registered condition failed?

For each source-fitted 48→96 top-four SAE, the independently trained input-state normalizer/PCA is frozen before observing new test states. Its positive top-one physical decoder column is mapped through the frozen 512×48 PCA basis and normalized to unit L2 in the original native state coordinates. This is one of four pre-outcome direction classes: SAE, clean-model predicted head winner-minus-runner, head-null SAE, and a seeded isotropic random direction.

At depth zero (immediately after injection), **24/24** admitted SAE directions had absolute true-digit-minus-clean-rival slope >=0.05. After two further complete ACT steps, **24/24** had relative absolute slope <=0.10 of their immediate slope. The third condition requiring ||J₂u|| >=0.25 failed for **all 24** SAE directions. It also failed for all 24 random and all 24 clean task-head directions. Head-null SAE directions, which have zero initial head response by construction, survived the global-norm threshold in one puzzle per family, but they are outside the registered SAE primary event.

| Quantity, median across 12 puzzles | 30-hole | 34-hole |
|---|---:|---:|
| SAE |initial margin slope| at k=0 | 3.2940 | 3.3001 |
| SAE |margin slope| at k=2 | 0.002916 | 0.002275 |
| SAE ratio |S₂|/(|S₀|+1e−6) | 0.000902 | 0.000702 |
| SAE **full 97×512 tangent gain** after k=2 | **0.06223** | **0.04733** |
| Random full tangent gain after k=2 | 0.02866 | 0.02317 |
| Clean task-head full tangent gain after k=2 | 0.02300 | 0.02187 |
| Head-null SAE full tangent gain after k=2 | 0.09474 | 0.07911 |
| SAE target-token 512D tangent gain after k=2 | 0.003676 | 0.003008 |
| SAE fraction of retained tangent energy on target token | **0.00375** | **0.00338** |

Those last fractions mean the *remaining* JVP energy is concentrated on other tokens by two further ACT steps. Both full-state gain and goal-token gain are small. The mechanism supported by these chosen finite directions is **strong local contraction with off-token redistribution of the already attenuated tangent**, alongside loss of task-margin sensitivity. These measurements do not bound the full recurrent operator's maximum singular value; a yet-unobserved direction could amplify.

The immediate task-head direction shows a similarly strong full-state collapse, and an isotropic random direction loses most of its original norm. The result therefore favors the predeclared **generic contraction** alternative over a special surviving but task-silent SAE mechanism, on this fixed source checkpoint and selected intervention site.

## Independent mathematical consistency checks

The raw receipt retains each 97-token tangent norm, the 512-dimensional selected goal-token tangent vector, full-state gain, off-token gain, exact readout-head row and JVP task-margin inner product, plus ±0.2 physical edit secants and a full-BPTT reverse-mode goal gradient at k=2.

- The maximum absolute forward-JVP versus reverse-VJP depth-two slope difference for the SAE direction is about **9.10×10⁻⁹** across both families. This is a numerical consistency check on the trained checkpoint's actual 42 additional L_level calls, not a theorem about exact real arithmetic.
- Median absolute finite symmetric-secant minus JVP slope at depth two for SAE is about **2.72×10⁻⁵** (30-hole) and **2.26×10⁻⁵** (34-hole). The ±0.2 intervention is locally consistent with the linearized response at the precision of this measurement.
- Median empirical rank-based p against **128 separately seeded random rank-one observer directions** is 0.1202 and 0.0930. The SAE selected direction's task-gradient alignment is not uniformly exceptional; two and five individual puzzles respectively have rank p <=0.05. These secondary rank observations are neither independent replications nor the registered primary gate.
- The head-null SAE direction has numerical initial head slope approximately zero, while some post-recurrence target-margin slopes appear. This agrees with head-null directions gaining output relevance through recurrent mixing, in a regime where their full-state tangent response usually also contracts.

## Source-locked data and calibration integrity

The original pretrained model is [gaoxin492/TinyRecursiveModels-Sudoku-Extreme-att](https://huggingface.co/gaoxin492/TinyRecursiveModels-Sudoku-Extreme-att) file **step_21700**, SHA-256 **932b5e0df4e90021c21a064d4e716154086c46fda7afa676ea294484a30e38c3**, executed at FP32 CPU. Its network weights are unchanged.

The prospective 32-puzzle calibration (seeds 202624000000..202624000031) fit the 512→48 PCA and 48→96 top-four SAE with 200 frozen AdamW updates, before any pilot/test state access. Its final reconstruction-plus-activation objective is **0.3748009503**. Three separate GitHub Actions jobs consumed the *same byte-sealed calibration artifact*, with the pilot and full jobs independently verifying original NPZ SHA, canonical manifest SHA and model/source-hash provenance.

- Original calibration roster SHA-256 **a49925eb3ec76b4cdfd1a412720695b5295047ac7134ae64c4794ef30b77220a**.
- Calibration array fingerprint SHA-256 **83dacde8f13926fc41fc48cb17bcfba8c399538754efb7104e41e94bd93797ec**; physical SAE dictionary SHA-256 **14dc1a8b4130a1cbbf215e97a4a0d1c4004d23ded319546b6bf35ddc3238f5eb**.
- Original calibration NPZ file SHA-256 **073b7771676e44eb934c50bdf1e7639ffeadbdff12998979cfe4c43e42b2da25**; source manifest SHA-256 **cb44b910f94210745b7a7df304e575d5a194ee8cd7bec51a88281fccc95bf33a**.
- Engineering pilot30 and pilot34 manifest SHA-256: **e5b3adb603ade102a771f8e2495efa5e0c97d2037eecf196f6511428eb1cadb2**, **18f320c28d289fa7e093384ea12428943146d1dd7ee73078aec49259f1df4f40**.
- Full heldout test30 and test34 roster SHA-256: **766a2810a0ba4cec56a4b52039285f2b7c55ff6ec7f50febdd3e510713**, **9a23a41af705754615b30d3981c3df622ab6f4c5ba964f3a3a94a0ebe1f71f18**.

The original full raw **16,914,750-byte** JSON has SHA-256 **f1ca7ff94ce4152ca9f870c576a5bbc8aa699ac6a58d3496c8a2cddf9964f76a**, and independent standard-library audit JSON has SHA-256 **b147003447e77e12d03d50ebdbd50a57a18b2c707c8bb3ac88e630748c14aa2d**. The full archive is [Actions artifact #14204410412](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43818817318/artifacts/14204410412), ZIP SHA-256 **a7a55bb3bfa9f1f2e80287e4dd3db83b5534e0631a42b5658ce7e7b85c47b5bb**.

The separate stdlib replay verified all original roster/puzzle and source-calibration hashes, the clean rival choice, 97 tokenwise tangent energy decomposition, scalar 512D goal/readout inner products, 128 observer-null scores, finite symmetric secants and original 2000-cluster bootstrap gates. It replays recorded numerical results and **does not independently rerun the pretrained network** or certify real-number guarantees.

## Research implications and next experiment

LS-SAE02 found a near-complete task-margin effect loss after recurrence on only eight eligible selective-edit goals, with a 52/64 full-SAE reencoding abstention problem. LS-SAE04 reproduced the **task-sensitivity attenuation** on all 24 newly selected goals without requiring active-feature kill reencoding. Its full 512D JVPs then establish that **the chosen SAE perturbations themselves contract sharply as global hidden-state differences** by two ACT steps, with SAE median gains 0.047–0.062 and even smaller target-token gains. The original hypothesis of retaining strong hidden signal while losing task readout failed.

Next, test an independent frozen subspace of finite-horizon recurrent Jacobian top singular directions, not another refitted SAE or recycled heldout puzzle. A 512D *single-site input to 97×512 output* Jacobian's leading singular value may be much larger than the measured gains along decoder, head and isotropic directions; a source-locked matrix-free JVP/VJP power iteration can discriminate genuine contractive mixing from narrow transient amplification. This would require a new preregistration with separate calibration and unseen test puzzle IDs, equal-budget random singular-direction controls, and the original native checkpoint SHA.
