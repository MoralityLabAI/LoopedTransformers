# LS-NEURON-03: causal deletion and four-channel recovery

**Development only; FSI-01 REDESIGN_OR_PILOT_ONLY.** Stacked on [LS-NEURON-02 #359](https://github.com/MoralityLabAI/LoopedTransformers/pull/359) at 080bbf163367b2d6fd5dd8cd387b5fa06018c55c. The [development lock](protocol.md) was hashed locally before execution. No sealed NEC04/Sudoku/native TRM data accessed, no paid compute. Consult the [EvalMath mechanistic preflight](https://github.com/MoralityLabAI/EvalMath/tree/main/skills/evalmath-mechinterp-power) and [FSI-01](https://github.com/MoralityLabAI/EvalMath/blob/main/paper/finite_sample_inference.md).

## Design and audit

Three original 2400-update tied affine parents (seeds 11,23,37) from the LS-NEURON-02 archive SHA256 23f76dad425acc9ab76c258d4c6835b4bf9d8c6d04a22a66784b22141c3bcea8; all three checkpoint hashes match the parent manifest. T4 data: 8 train contexts (2048 rows), 4 calibration contexts, 4 development contexts (1024 rows). Select masks on the first two calibration contexts (512 rows), remaining two diagnostic. Each context contains all 256 focal-bit masks, predicting chronological XOR prefixes. T6/T8 development readouts each contain 2048 rows; T2 contains 16. Original readouts reused, not fresh confirmation.

For the active set S, remove the channel j minimizing actual mean BCE over calibration roots and all four prefixes after the intervention S minus j. Evaluate every candidate, tie by ascending index, repeat until one survives. This is greedy search, not exhaustive subset optimization. 2079 candidates/seed, 6237 total. Comparison: dense activation-times-loss-gradient salience and two fixed random permutations. Same physical masks applied across visits. Exact compact weight extraction preserves the masked computation within 3.8147e-5 maximum absolute logit error on selection rows. All 96 mask-width combinations checked. Greedy width-four sets: [25,48,55,60], [1,31,42,51], [0,12,26,28].

## Zero-update results

Equal-seed mean accuracy, percent, at T4.

| Retained width | Greedy | Salience | Random0 | Random1 |
|---:|---:|---:|---:|---:|
|64|100|100|100|100|
|48|100|99.80|88.44|68.20|
|32|100|88.05|64.06|52.34|
|16|89.00|54.13|54.33|49.97|
|8|57.06|50.88|50.65|50.00|
|4|51.60|49.48|50.00|50.00|
|2|49.51|50.33|50.00|50.00|
|1|49.77|50.16|50.00|50.00|

Greedy 32 retains 100% in each of three parents. Width16 seed accuracies 81.84/85.16/100%. Shadow-calibration results at 16: 82.23/84.18/100%. At width4, exact deletion is near chance. Dense T4 accuracy 100%; complete MLP removal 50%.

## Sparse recovery

24 width-four runs, 513 parameters each, 96 checkpoints (0/120/600/1200 updates). AdamW .003, batch128, clip1, zero weight decay, same source batches across all arms within seed. Transplant selected weights plus parent init/bypass/head; reset-MLP arm replaces both MLP layers including bias with random initialization, retaining other parent parameters. Scratch uses same fresh width-four initialization and training schedule. Frozen MLP trains 353 parameters; frozen bypass/head/init trains 160. All other arms train all 513. This matches additional updates, not total training compute including the parent pretraining and greedy search. Greedy and salience selections require additional calibration computation, random and scratch do not.

| Arm | T4 step0 | T4 step120 | T4 step600 | T4 step1200 | T6 step1200 | T8 step1200 |
|---|---:|---:|---:|---:|---:|---:|
|greedy|51.60|55.99|87.83|95.67|81.48|68.28|
|salience|49.48|50.49|84.99|96.68|85.17|55.01|
|random0|50.00|50.62|88.90|95.12|89.68|54.12|
|random1|50.00|54.88|81.64|95.80|87.61|59.57|
|reset_mlp|50.00|49.77|62.99|95.57|90.25|72.49|
|scratch|50.85|50.00|61.07|97.85|86.47|63.75|
|freeze_mlp|51.60|52.31|52.12|53.61|51.73|52.10|
|freeze_bypass|51.60|53.94|56.51|62.70|49.80|47.92|

Parent T6/T8 99.93/99.19. Primary locked developmental contrast greedy minus mean of random0 and random1 at T4/1200: +0.21pp; seed-level differences -12.9883/+12.4512/+1.1719pp. Fails proposed +5pp scientific floor. Greedy T4 seed accuracies 87.01/100/100. Scratch exceeds greedy by 2.18pp at endpoint; the inherited arms learn faster at 600 updates, random0 also exceeds greedy. This does not identify an advantage attributable specifically to greedy selection. Freeze results suggest recovery depends on adapting both parameter groups under this optimizer/budget; no impossibility theorem for other optimizers.

Greedy at endpoint T4/T8 mean prefix-logit correlation with parent .9102/.5424; relative hidden-state RMS displacement .8545/.9657 (denominator parent RMS). T8 final class agreement with parent .6753. Reset MLP has T8 logit correlation .7631 and accuracy 72.49%. Hidden coordinate differences are not a basis-invariant mechanistic test, and output agreement is not proof of mechanism preservation. Four-channel recovery is consistent with substantial relearning.

## Evidence gate and next work

Unit: training checkpoint seed, n=3, shared task contexts and multiple masks/horizons/steps correlated. T4 endpoint 1024 rows per checkpoint, 3072 across seeds per arm, 4 development nuisance contexts per seed. Actual comparison family includes all 96 zero-update mask-width settings, 24 recovery runs, 96 checkpoints, 288 horizon-specific checkpoint evaluations, plus shadow/parent diagnostics. No confirmatory inference or p-value, no reliable sham floor, no independent seed-level power estimate. Random masks are channel-count-matched, not matched physical output-dose shams; freezing controls alter trainable parameter budget. Calibration and development data are reused; no new untouched population. Evidence class: exploratory trained-model observation, power-limited. Primary effect floor not met. For confirmation, freeze candidate families, establish matched shams and independent checkpoint/world counts using EvalMath preflight, and lock new task families. No hardware latency or quantization result. Synthetic affine recurrence is not a native transformer.

Next prioritize width32 compact extraction with long-horizon controls, then iterative pruning **during training** (or dense-teacher distillation with random-mask and scratch controls) at fixed additional and total compute. Evaluate native MLP-dependent checkpoints only with headroom qualification and separate protected-cohort protocol. For a depth-width frontier, report recoverable performance and original-function retention as separate quantities. Test compact models using length curricula and precision controls on untouched horizons.

## Reproduction

The companion archive includes this exact parent pilot source, three parent weights/manifest, all 96 new checkpoint weights, raw candidate loss ledgers, correctness vectors and audit. From archive's ls_neuron03 directory:
```sh
sha256sum -c lock.sha256
for seed in 11 23 37; do
  python recover.py --seed "$seed" --base parents --out replay
done
python audit.py --base parents --out replay
```
Use PyTorch 2.10.0+cpu; deterministic single-thread workers. Archive contains original parent manifest with entries for other files not included; only the three used parent checkpoints are required. Audit directly replays selected removal losses and verifies ledger minima (not independent recalculation of all unchosen candidate losses), all 96 checkpoint hashes, 288 correctness/correlation replays, 189 chosen deletions, max loss error 1.1920929e-07, frozen tensors and matched sampler identities. It does not independently retrain. Source Git blob IDs match local files: parent_pilot.py 19bd4184dd36fcd92c676565b910e2aa3c8e7652, recover.py f4e2ad182e599b8d58da3a05fa58d526b1fc9889, audit.py 6f1c383125ede75d7cce2edd58ea9ea59e365b9a, protocol.md fd4971ada30dd9c7a2076daedb9fab90bc44a438.

Receipt SHA256:
- results-11.json: 1d129af88327b24fd8dad7b4a0245ad4449efc571c1080ff96122cb10ff5fe19
- results-23.json: a03b2ceddb1860a920af58f5ed46c7d3d3cc8ccc5d322eabd206359739cd484f
- results-37.json: b8c87605165afbb88402761efa739dc3acd5b0c10cbcdb43eea1f00fcd43b462
- selection-11.json: 2a8bf7841d9ea2971825c027f81f79b1e557314a857b9b8dec82b4c0b56a1b1c
- selection-23.json: fe5314cd7183519e92b29511cc4b8400ec9de7393aaee3de3ac22f9f50c65b44
- selection-37.json: c152b90d5660792a39cbc993824e065b400c198b1df76466f737f77340258ac5

Archive SHA256: 3afa0cf6c8566ae1afd497dbf3cd10a771d16688e952fa1962faea109a02f5bd. Raw local binary artifacts supplied separately; source and report on GitHub.
