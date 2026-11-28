# LCD campaign: scalar, spectral, Anderson, adaptive depth, selective precision and verifier transfer

Status: all source runners implemented. Native model execution and empirical gates remain pending. The original LoopCD paper is the inspiration; LCD-02 through LCD-07 are extensions, not reported results of that paper.

## Run order and artifacts

| ID | Module | Calibration or fit | Held-out |
|---|---|---|---|
| LCD-01 | lcd01_huginn | 48 roots | 144 roots |
| LCD-02 | lcd02_spectral | 48 fit + 48 selection | 192 roots |
| LCD-03 | lcd03_anderson | 48 roots | 192 roots |
| LCD-04 | lcd04_adaptive | 48 roots | 192 roots |
| LCD-05 | lcd05_precision | 48 fit + 48 selection | 192 roots |
| LCD-06A | lcd06_sudoku | 36 roots | 96 roots |
| LCD-06B | lcd06_lean | 24 roots | 96 roots |
| LCD-07 | lcd07_compose | earlier independently screened components | 192 fresh roots |

See [LCD-01](../lcd01/README.md), [LCD-02](../lcd02/README.md), [LCD-03](../lcd03/README.md), and [LCD-04/05](../lcd04_05/README.md) for the registered model-specific protocols.

From the repository root, in the existing environment:

    python -m research_gym.scripts.lcd_campaign

Print the full native job plan without executing:

    python -m research_gym.scripts.lcd_campaign --plan --native-huginn --sudoku --lean-ckpt /path/to/verified/checkpoint

Run the cached Huginn stages on the local CUDA machine:

    python -m research_gym.scripts.lcd_campaign --native-huginn --out results/lcd-campaign --max-hours 4

Add the independent verifier transfer stages:

    python -m research_gym.scripts.lcd_campaign --native-huginn --sudoku --lean-ckpt /path/to/verified/checkpoint --lean-data lcgrt-v1 --out results/lcd-campaign --max-hours 4

The four-hour default is a hard execution cap, not a runtime estimate. Increase deliberately if needed. Driver stops at a failed stage, missing receipt, or timeout; it preserves partial outputs and never overwrites an existing receipt. --resume preserves completed stages, and successor stages validate their own freezes. Preserve incomplete/failed artifacts and use a new output root for a retry. Individual independent modules can be invoked directly if an earlier Huginn-specific gate blocks the campaign. No model downloads by default. Native prerequisites are the cached pinned Huginn revision, existing CUDA environment, cached hash-verified TRM checkpoint for Sudoku, and the completed Lean HRM checkpoint, hashed local dataset and Lean 4.19 lgym tools for Lean. Verify GPU power, VRAM and free disk. Use local absolute paths only when invoking, and scrub private paths from receipts before publishing.

## LCD-06A: Sudoku transfer

Native PureSudokuTRM, 8 ACT steps, reference 4; compare 16-step unguided. 44/48/52-hole fresh diverse synthetic roots, exact solution count 1, cross-split canonical digit/automorphism screening using the existing roster. No external training-corpus exclusion has been established. Frozen scalar and bounded Anderson terminal extrapolation chosen on calibration whole-grid accuracy then missing-cell NLL. Fit Anderson coefficients to joint H/L residuals, apply only H correction at the terminal readout. Controls match H norm and include orthogonal random direction. No global stationarity claim for this two-clock architecture. Verify every output against givens, row, column, box constraints and unique solution; no postprocessing of predictions. Report full-grid closure and missing-cell accuracy separately. Do not infer generalization to unseen benchmark distributions from synthetic roots.

## LCD-06B: Lean transfer

Native ProofNet HRM with readouts at 4/8 fast steps, compare ordinary 8/16. Model and dataset identity must match. Check every declared dataset shard SHA, regenerate statements and canonical hashes, and exclude overlap with fresh calibration/test roots. The generated fragment uses isolated Lean core REPL, only local hypothesis application and constructor, avoiding target theorem self-citation. Fixed best-first policy, 32 expansions per root, zero global features, same seed; guide only legal tactic scores, keep terminal atom/halt heads unchanged. Guided hidden readout adds a paid decoder call; logit contrast uses already-produced tick readouts. Certify regenerated target and exact tactic script through standalone Lean compile and no-axiom audit; require exit zero, replay individually if batch fails. Report certified closure, raw closure, expansions, calls, actual fast steps and readout count, search and certification times. This is a generated propositional fragment, not arbitrary Mathlib proof search. Positive transfer pilot screen is improved certified closure over same-depth base and matched random where applicable, with a cost audit; retain negative results.

## LCD-07: evidence-gated composition

Validate complete frozen/calibration and held-out receipts and their environment/source identity. Component eligibility:
- spectral: positive delta over tuned scalar, norm-matched scalar and norm-matched random; latency <=110% of fixed LoopCD.
- Anderson: AA12 positive delta over R16, tuned scalar and both norm controls, latency <=110% R16.
- precision: within 2pp of BF16, better than uniform Q4 and matched random basis, actual carried payload <=40% of BF16.
- adaptive: positive delta over fixed16 and depth-matched random, both realized budget caps, latency <= fixed16.

Require at least two eligible fixed-depth components. Otherwise write an explicit skip receipt. Evaluate every subset of eligible spectral, Anderson and precision components on a new 192-root split, against R16, R32 and fixed LoopCD. Eligible adaptive policy is a separate comparator at its frozen budget; its features were fitted on clean native states, so it is not silently applied to modified trajectories. Composition order is codec after each native transition; AA at 12 and recode after correction if enabled; spectral or scalar terminal readout at 16. Anderson on modified trajectories is extrapolation, not a proof of fixed-point acceleration. Basis costs and codec roundtrips accounted, paired closure and actual forward latency/peak memory recorded. Primary composition is frozen full eligible subset; test all subsets without choosing the best after viewing test. These are descriptive pilot gates, not powered statistical confirmations. A further fresh confirmation and native generation benchmark are required for deployment claims.

## Scope and limitations

CPU CI tests numerical mechanics, legal masks, verifier logic, policy allocation, packing, and factorial enumeration; it does not execute cached models or Lean. Native parity and calibration gates remain essential. Quantized-state payload is not peak VRAM, and BF16 recurrent matmuls are unchanged. LCD-06 transfers scalar/Anderson or scalar logit/hidden guidance; cross-architecture spectral bases, selective precision and adaptive policy transfer require new architecture-specific fits. No results, speedup or accuracy improvements should be claimed from source-only tests.
