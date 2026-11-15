# RAM-02 frozen local protocol

Frozen before RAM-02 fitting and before its FC-13 regression. The configuration
file is JSON-compatible YAML; its byte SHA256 is embedded in every checkpoint.

The only supervised corpus is original artifact 14095156819 from run 43547762837,
ZIP SHA256 d91a12d725ca0a611b4de308c80365bfd3888d88449b03b2c5f07137a1c2025a.
Verify RAM-01's state and semantic-effect hashes and all 51 states / 961 probes.
Do not regenerate or relabel any training probe. The model/data code is derived
from RAM-01 ca1982833af7f1adeb55029ef886f56ebf82b5d4, with portable monitoring
and device-aware scoring. Keep the existing 39-state / 766-row training split
and 12-state / 195-row development split. Only printed state text, candidate
script and tactic family enter features. Original reference commands and
future actions are excluded. Whole modules and theorem prefixes stay together.

Required fits: loop32 (82,692 parameters), then loop64 (198,148), each with
seeds 17, 29, 43 and four shared passes. Only after those six fits, run optional
flat64 (173,252) for the same seeds. No loop128 is planned. No all-51-state
refits: early stopping and best-checkpoint selection use development CE only.
Maximum 2,000 updates; evaluate every 100; stop after four validations without
an improvement of at least 0.0001. AdamW .002/.01, gradient cap 1.0, effective
batch64 from microbatch16 and four accumulation steps. Class weights are
inverse-square-root training counts, normalized to mean one. Normalize the
accumulated loss by the sum of class weights of the complete effective batch.

Compare identical FP32 loop64 updates on CPU and CUDA after ten warmup updates,
using 40 measured updates. Choose CUDA only at >=1.15x CPU throughput. Existing
PyTorch 2.6.0+cu124 is reused instead of upgrading to RAM-01's 2.10.0. No compile,
mixed precision or GPU compilation. Use deterministic algorithms and two CPU
threads. Monitor the entire training/Lean process tree every 0.1 seconds,
available physical and commit memory, CUDA allocated/reserved memory, and
actual nvidia-smi dedicated memory (two-second polls). Stop at 6 GiB RSS or
2.5 GiB allocated VRAM, or below 2 GiB available RAM/commit or 512 MiB dedicated
GPU headroom. Preserve checkpoint and failure receipts. Training tensors are
CPU resident and under 2 MiB, transferred only per microbatch.

Search is a bounded depth-first prototype over complete ordered goal lists,
retaining sibling goals after induction/rcases/constructor. Deduplicate
syntactically canonical goal-list signatures, preserving goal order and
multiplicity. Generate candidates solely for the first active goal using the
unchanged FC-12 grammar. Every controller receives the same 20-second deadline,
12 actual Lean calls including final certification, and three successful
transitions per path, including the closing transition. Candidate utility is
2*P(scratch closure)+P(single-goal progress)+P(multi-goal progress). Compare
native ordering, training-only family frequency and every trained model.
Do not expand budgets after seeing scores; no secondary budget is planned.

Evaluation is exclusively a clearly labeled post-hoc regression of the published
FC-13 sixteen-theorem suite, using MathCan c84e2025334e9a967c1297425a490f98d5b3d189,
the selector in Compactification-Theory e8a780254b5cf41d479095f4f22bcfc4aa76ed72,
and Lean 4.35.0. New source families stay unevaluated. Separate source-oracle
compiles never count as policy proofs. Scratch code may contain sorry for
state observation; only separate complete-theorem compiles without sorry,
printed dependency audits allowing propext/Classical.choice/Quot.sound, and
independent exact-source replay release a proof. Record original source hashes,
selected scripts, candidate hashes, logs, call/time budgets and replay hashes.

Inference-only pass ablations use 1/2/4/8 shared passes on each selected recurrent
checkpoint, reporting cached development metrics and CPU scoring time. These
are not separately trained networks and cannot establish proof gains.
Keep negative outcomes and class/family metrics. Report certificate cost and
unique solved sets separately from cached action prediction.
