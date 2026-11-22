# NEC-04B: frozen six-root native Looper headroom pilot

**Prospective calibration-only native training protocol, 22 November 2026.** The [NEC04A symbolic Boolean world generator](../../reports/NEC04A_BOOLEAN_WORLD_GENERATOR_RESULTS.md) produced 24 calibration root programs and 61,440 source-training labels with exact independent scalar truth-table audits. The observed 11/15/19-gate program tiers and Boolean algebraic degrees are *structural task proxies*, and **no native model headroom has yet been measured**. NEC04B is the first controlled native-model training pilot to establish empirical clean validation accuracy before deciding whether source SAE fitting or a physical sham gym is warranted.

## Frozen source and world-root allocation

The authoritative input is the original immutable GitHub Actions NEC04A calibration-source corpus artifact from run **44983509977**, artifact ID **14738712603**, with full source-case JSONL SHA256

**82c80643d3c99871913f5228f1a1cde1eec4484787c1784872b508349793e9e0**.

The original 64-new-world seed commitment registration SHA256 remains **88bf58d32fdf15a1270986c238afb3def1a7e6371d3eee64c4edd264c9cf041e**.

Register **only new NEC04A calibration world roots indices 0,1,2,3,4,5**, with exactly two roots per easy/medium/hard symbolic tier. Each root retains the immutable original composed Boolean AST, full native 12-bit histories, five focal positions and seven context positions. Each **source training** panel contains the original first **80 complete seven-bit context groups ×32 focal masks = 2560 input histories**. For a frozen within-calibration-world model selection readout, open exactly its **24 owned developmental-context groups ×32 focal masks =768 input histories**, whose ownership and SHA256 assignment were already committed in NEC04A. These *developmental-context groups of the six calibration-world roots* are distinct from the **24 new NEC04A developmental WORLD roots**, which remain uninstantiated and sealed. Each calibration world also has **24 reserved-context groups**, whose model predictions and intervention outcomes remain unopened. All complete focal blocks stay owned by one split, and source and validation history sets must be disjoint.

The original NEC03Z source groups56–63, old NEC03P recruits, original 32 five-input embargoed world roots and old 64 four-input confirmation worlds are likewise protected.

## Independent target oracles before model training

For the **composed teacher**, use the exact NEC04A frozen root-specific Boolean program's canonical input label. Verify all original 2560 source labels against the NEC04A corpus and the recursively evaluated scalar circuit oracle; generate the 768 owned-validation labels independently from the already frozen root program and context ownership.

For a **new additive/linear teacher**, do not transplant the historically saturated NEC03 additive teacher. Generate one deterministic root-specific signed linear-threshold Boolean program:

- For each 12-bit position, choose an integer nonzero signed weight by SHA256 from the old calibration-root seed and a fixed domain string. Focal absolute weights belong to {1,2,3}; context absolute weights belong to {1,2}, with deterministic SHA256-selected sign.
- For a Boolean history bit x_i, compute the signed score s(x) as sum of weight_i*(2*x_i−1).
- Evaluate possible integer score cutoffs k across the entire 4096-input symbolic teacher domain; choose the cutoff minimizing absolute positive-class count minus 2048, breaking ties by smallest absolute cutoff and then its signed value. Class1 iff s(x)>k, equivalent to 2s(x)>2k+1. Require 35–65% global class balance, all five focal bits essential and at least three context bits globally essential, without truth-label feedback from a neural model.

The teacher generator and **separate independently implemented scalar teacher audit** freeze the original 12 weights, twice cutoff, 512-byte exact 4096-input truth-table SHA256, source and owned validation labels and root identity in one pretraining manifest. The independent source/validation auditor checks **6 × (2560+768) ×2 =39,936 teacher-labelled case-arm values** and checks that no original reserved-context *model evaluation*, new development/confirmation world program, or old NEC03P model is touched. All teacher selection and original case histories are fixed before any Looper weight update.

## Native model architecture and exact initial budgets

The original native recurrent network is the repository's **PEF07 Looper**, with 32-dimensional hidden state, binary token embeddings, 12 chronological recurrent updates, shared 4-head residual attention/MLP block and two-logit classification head. The source code for this class and its underlying MiniAttentionMlpResidualBlock are pinned by Git blob SHA in the run lock, alongside the NEC04A generator/oracle and every NEC04B source file.

For every one of **six original world roots × both composed and additive teachers**, use deterministic CPU Torch 2.10.0 with single compute thread and deterministic algorithms enabled. Seed model initialization by SHA256 of the original world seed and model-arm identity; use a separate SHA-derived/stable source minibatch RNG. Train one continuous **AdamW** optimizer per teacher with learning rate **0.003**, zero weight decay, batch **64** sampled from its original 2560 source rows with replacement, gradient clipping L2=1, and **900 total optimizer updates**. Save both native model and optimizer state with checkpoint SHA256 and weight SHA256 after exactly **120, 240, 480 and 900** continuous updates. The optimizer state and sample RNG remain uninterrupted between saved budgets. Repeated 32-focal-mask examples and world roots are not independent training observations.

Seal all four checkpoint files and their parent root/arm manifest SHA256 **before looking up owned-validation teacher labels or reporting validation accuracy for that arm**. Checkpoints are a sequential training trajectory, with post-hoc validation scoring of the already registered budgets. Root identity, native arm, source input sha, seed, loss receipt, optimizer step, exact batch and weights are carried in each checkpoint.

## Independently replayed headroom and stop policy

For each checkpoint, score full **2560-case source training** and complete **768-case owned calibration-world validation** from a frozen, eval-mode native model with the same original two-logit head. Preserve complete raw 2-class logits and class predictions in original case order, class distribution, ordinary and balanced accuracy, source-vs-validation gap and checksum of model checkpoint. A separate checkpoint auditor reloads each exact SHA256 checkpoint, verifies the native model state hash and optimizer update counters, *independently regenerates both teacher label functions*, and recomputes **all checkpoint source/validation full recurrent logits** and original root accuracy. This is independent *checkpoint and output replay*, not a second from-scratch optimization trajectory. Any differing model logits, source identities, optimizer steps, case labels or raw receipts fail the original root.

After **all six world roots, both model arms and all 48 independent checkpoint audits** pass, a pure-stdlib root-level aggregator bins each original world/teacher's measured validation accuracy into the fixed target ranges:

- **Moderate:** [60%,75%)
- **Intermediate:** [75%,90%)
- **High:** [90%,98%] (inclusive at 98%)
- Other performance: out-of-range

A native budget may be considered ready for subsequent full 24-calibration-root expansion only when **at least two of six independent world roots per teacher arm appear in EACH of the three target accuracy bands**, under the *same single checkpoint budget* for that teacher family. Select the **lowest training step budget** satisfying this requirement simultaneously for composed and additive teacher arms. If no budget qualifies, report the complete six-root accuracy/headroom matrix and **STOP_NATIVE_HEADROOM_SELECTION_CALIBRATION_ONLY**. The six selected original roots, all failed checkpoint budgets and both teacher arms remain in the denominator, with no root reseeding or budget tuning after observing these validation outcomes. The eventual 24-root expansion requires a separate prospectively frozen training policy.

**No source SAE is fitted or physical sham evaluated in this stage.** The distinct follow-up source SAE fit/rank (first 40/next 40 source context groups), full-affine reconstruction test and physically matched plus functionally inert negative control belong to a separately registered experiment after headroom succeeds. No new development-world, confirmation-world, NEC03P recruited-world or previous embargoed-world model evaluation is authorized.

The future 95/95 upper-tail causal-sham tolerance still requires ≥59 independently eligible causally measured world roots under a valid physical and functional null; the original NEC03P recruited 69/88 source-eligible worlds have **zero measured causal-sham outcomes**. NEC04B model headroom is a calibration-root diagnostic only, never population-level causal mechanism confirmation.

**FSI-01: REDESIGN_OR_PILOT_ONLY.**
