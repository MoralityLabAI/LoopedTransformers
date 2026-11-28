# TFL-02 / TFL-06: frozen decision repair and causal component pilot

Status: **development harness; REDESIGN_OR_PILOT_ONLY**. This is a new implementation on main. The previously reported TFL-01 local commit and receipts were unavailable and are not included or represented as reproduced.

## Run

From this directory, use Python 3.11 and PyTorch 2.5.1. The structural smoke test uses only PyTorch:

```sh
python gym.py --self-test
```

For a local GPU pilot, install Transformers 4.57.1 and compatible PyTorch/CUDA. NF4 additionally needs compatible bitsandbytes and accelerate. Start with a tiny smoke run, then the full development set if native/repeat equality and label-token guards pass:

```sh
python gym.py --model Qwen/Qwen3-0.6B --device cuda --nf4 --cal 4 --dev 8 --out smoke.json
python gym.py --model Qwen/Qwen3-0.6B --device cuda --nf4 --cal 64 --dev 256 --out dev_nf4.json
python gym.py --model Qwen/Qwen3-0.6B --device cuda --cal 64 --dev 256 --out dev_fp16.json
```

A 4 GB RTX 3050 is the initial NF4 target; memory and throughput have not been measured. Use an immutable checkpoint revision via --revision for recorded experiments. The default model revision is main and the receipt records the resolved model config commit if available. This adapter expects a tensor-returning Qwen-style decoder layer. It disables KV cache, freezes parameters and uses full-prompt forced-choice logits; it has not yet been exercised against a real HF checkpoint in this environment. No SAE or native BPD implementation is imported.

## Design

Each independently seeded root generates a four-branch directed acyclic graph. A traversal oracle checks that exactly one first move reaches the goal. Root IDs 0 through 63 calibrate the uncentered leading singular direction of the last-token anchored-minus-native residual (without using labels); roots 100000 through 100255 are development-only. One generator grammar, not a cross-family reasoning benchmark. Graph structure permits specialized reachability solvers and potentially shortcut heuristics; report a heuristic/graph oracle separately before interpreting pretrained results.

Let y=g(x), m=x+(y-x)/2, e=m+(g(m)-m)/2, and a=(y+e)/2. A selected decoder layer is called once for native and twice for every other arm. For the final position let d=a-y, v be the frozen unit calibration direction, p=(v dot d)v, and q a root-seeded random vector orthogonal to v rescaled to ||p||.

| Arm | Output at final position |
|---|---|
| native | y |
| repeat | y (second g(x) must agree bitwise) |
| anchor | a |
| ablate | a-p |
| sham | a-q |
| component | y+p |
| sham_component | y+q |

Other positions are unchanged relative to the arm's anchor or native reference. The latter two arms isolate sufficiency of the last-position component. Sham comparisons are rank-one, equal pre-rounding edit energy; actual post-cast edit norms are retained because FP16/NF4 can break dose matching. All arms record paired correctness, raw label logits and selected block calls. Native/repeat discrepancies fail closed; warm-up and adapter debugging may be necessary on the real checkpoint. No label-conditioned intervention is used.

## Mechanistic-power registration (development, not sealed)

* Unit: one independently seeded graph root within this generator; 64 calibration roots, 256 planned development roots per checkpoint/precision; zero pretrained roots measured here. Within-root arms are paired. Precision or layer comparisons require additional multiplicity planning.
* Primary contrasts (three): anchor versus repeat (net decision repair); sham versus ablate (necessity relative to matched removal); component versus sham_component (sufficiency relative to matched addition). Positive direction in each case. Report all arms.
* Scientific minimum: +0.02 absolute accuracy; dose: one rank-one projection with its per-root norm-matched orthogonal sham, at a fixed layer (13 default). Realized dose mismatch requires redesign or an explicitly registered tolerance before confirmation.
* Resolution at n=256 is 1/256 = 0.00390625. Observed sham floor and power are unavailable until pretrained development data exist. The conservative FSI-01 positive-effect lower bound is observed paired mean minus sqrt(2 log(m/alpha)/n); at n=256 and m=3 it subtracts about 0.179. A development win at 2 pp alone does not authorize confirmation.
* Readout exposure: calibration and development are open. No confirmation split allocated or opened. Freeze model revision, layer, precision, doses, comparison family and root sampling before a fresh confirmation set. Calibrate matched sham variability from independent development roots, then use EvalMath's mechanistic-power preflight and FSI-01 bridge. Do not enlarge an all-zero pilot or claim equivalence from a null.
* Evidence class: exact structural smoke checks on sampled generated roots and one synthetic block; no pretrained causal or generalization claim.

The `rows` array contains root_id, prompt, gold, scores, correct, dose and calls. For each contrast, export case_id=root_id, root_id, candidate_correct and control_correct to the repo's research_gym/scripts/evalmath_fsi_gate.py JSONL bridge with a separate registration. Keep sealed_readout=false for this pilot; no confirmation verdict is available.

## Measured local smoke test

The source blob was tested in a CPU sandbox with PyTorch. All 256 generated roots had a unique verifier-approved first move and reproduced exactly. The toy block passed native and repeat equality, anchored recurrence, equal-dose edits (within 1e-6 in float32), isolation of the final position, call counts and hook removal.

```json
{"scope":"structural CPU checks only","roots":256,"matched_doses":{"ablate":0.24896566569805145,"sham":0.24896562099456787,"component":0.24896566569805145,"sham_component":0.24896568059921265}}
```

The tested git blob is b38c1792b1b03094db52821caab400447f8fd44d. Transformers and CUDA were unavailable. No real pretrained decisions, accuracy improvements, quantized dose comparisons, GPU runs or model parameter hashes have been measured. The smoke test is a wiring check, not mechanistic evidence.

## Next applications

Replace the rank-one SVD direction with frozen BPD or SAE component bases, with an independent calibration and equal-rank, equal-energy shams. Extend the same verifier protocol to Lean tactic repair and consequential storyworld actions. Once real repairs are identified, use DataGoblin to collect verified trajectories and train a tiny student on them. A training-time steering A/B should compare an unsteered student, a same-energy sham-steered student and a causally selected component-steered student with matched data and optimization, measuring verifier success with steering removed at test time as well as with steering on.
