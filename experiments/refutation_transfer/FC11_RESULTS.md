# FC-11: first imported MathCan multi-step tactic experiment

The first valid held-out comparison completed under [Lean 4.35 CI
#43543404745](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43543404745).
It reused the **original frozen pretest model weights** from
[training job #43541360928](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43541360928)
and repaired only the malformed Lean-native search baseline.

## Real original research theorem sources

All model state inputs were compiler-emitted `trace_state` results from
original MathCan research proofs at source commit
`c84e2025334e9a967c1297425a490f98d5b3d189`
under `leanprover/lean4:v4.35.0`.

- Preregistration source commit:
  `b57e2500e829b12fdfc257077b8e410171c93b98`
- TRAIN-ONLY source revision:
  `2e7e749d4c7f8610a4083f4f8551d084102ea55a`;
  16 original MathCan research theorems, 31 intermediate Lean states;
  JSONL SHA `47fae96eac1f9b176519faaa0a64cbf016203b6f23ba56e40a2295e9fedaa97d`.
- Later HELD-OUT selector:
  `b821e0d85423606cf971b1f80db8092ede693d81`;
  14 original research theorems in three disjoint MathCan modules,
  29 real tactic states; JSONL SHA
  `a8e7bee63f80c03c640ccdbb9bada1f8d943e33fb403e37ad1fc7dbf536e11ad`.
- [Original train source CI #43540113159](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43540113159)
  and [original held-out source CI #43540299693](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43540299693)
  compiled the original proofs and saved actual compiler states.

The training matrix tried every registered tactic
`omega, decide, rfl, simp_all, aesop, assumption, intro, constructor`
on all 31 real training states, producing **248 real Lean compiler
scratch probes**, 33 accepted/progressive and 18 that closed their
scratch goal. The source-only audit [CI #43541101044](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43541101044)
preserves all trials. Scratch runs may append `sorry` only to obtain
state feedback. They are explicitly **untrusted**, never accepted as
finished proof certificates. In MathCan's original import environment,
`aesop` is unavailable and contributes only failed action examples.

The original LR-01 `TypedLoop`, **62,401 parameters**,
uses four tied recurrent passes. Seeds **17** and **29** trained
for 300 fixed AdamW steps, batch16 on actual training Lean
states and observed candidate effects. Original checkpoints:

- Seed17:
  `f79fcbbdd4d86edb77d7b43ade42198bdde3ac7b0f527d181ee781df848af365`
- Seed29:
  `f9c052406e08d5bbe24712f649ce564c772a07e4b7e9f448e21d39eae848b182`

The train-only model source did not contain the held-out theorem selector.
The evaluation reloads exact SHA-identified checkpoints and independently
validates the held-out actual goal-state SHA before scoring.

## Registered result, including corrected baseline

A complete proof is accepted **only** after recompiling the original
MathCan theorem with the policy's actual selected tactic sequence and
**no sorry**, under the pinned real Lean4.35 compiler.
At most three selected tactics were allowed per theorem.

| Policy | Intervention /6 | HookState /4 | GraphAuthorization /4 | Total /14 |
| --- | ---: | ---: | ---: | ---: |
| LR-01 seed17 | 0 | 0 | **4** | **4** |
| LR-01 seed29 | 0 | 0 | 1 | 1 |
| CPU-budgeted native `first`, seed17 allowance | 1 | 0 | 2 | 3 |
| CPU-budgeted native `first`, seed29 allowance | 1 | 0 | 3 | **4** |
| Original complete source proof oracle | 6 | 4 | 4 | 14 |

**Registered proof-count predicate:** technically passed on seed17
(4 > 3), but due a **startup-sensitive one-proof advantage**.
The native baseline experienced **two timeouts** under seed17's
per-case time budget and **one** under seed29's. A separate
nonbinding native search check independently compiled all four
GraphAuthorization arithmetic theorems.

Both models failed **all ten genuinely multi-step theorem
proofs** in Intervention and HookState. Each model-generated
accepted theorem used a single arithmetic tactic. No evidence
of successful learned multistep proof synthesis emerged.

### Real measured wall-time

| Seed | LR-01 total CPU-runner wall ms /14 | Native total wall ms /14 | LR-01 ms/certified proof | Native ms/certified proof |
| --- | ---: | ---: | ---: | ---: |
| 17 | 22,920.84 | **14,475.85** | 5,730.21 | **4,825.28** |
| 29 | 32,006.77 | **14,662.53** | 32,006.77 | **3,665.63** |

The registered **compute-efficiency criterion failed** on both seeds.
The tiny source corpus and repeated Lean subprocess launches make
the matched deadline sensitive to CI runner load and startup costs.
The source benchmark is public, and its original reference proofs
remain available as a 14/14 oracle.

## Independent replay

[Permanent research ledger](https://github.com/MoralityLabAI/Compactification-Theory/blob/feat/refutationbench-fc11-imported-lean-search/research/refutation_bench/results/fc11_imported_mathcan_first_run_20261114.md)
preserves all source hashes, original compiler
proof status, failed preliminary native controls and
paired results. The [independent archive audit](https://github.com/MoralityLabAI/Compactification-Theory/blob/feat/refutationbench-fc11-imported-lean-search/research/refutation_bench/results/fc11_independent_audit_43543404745.json)
verified the two frozen checkpoint hashes, the actual 29 held-out
goal-state SHA, all **28 model response rows** and **28 CPU-matched
native response rows**, exact selected action vocabulary, original
case identities, all positive certificates marked as final sorry-free
Lean compilations, and the equality of every per-case native time
allowance to the original frozen actor elapsed time.

The full original test artifact is
`fc11-verified-frozen-replay-43543404745`.
It includes compiler logs and the SHA-frozen original theorem
state sources. The original train artifact is
`fc11-frozen-imported-lean-models` from run #43541360928.

## Failure mode and next stage

The registered candidate bank lacks parameterized `exact`,
`rw`, `simp only`, `have`, `rcases`,
`calc`, and `induction`, which are present in the
original research proofs. Additionally, `aesop`
is not available under MathCan's import-only setup.
Most genuinely multistep original proofs cannot be
expressed as a maximum of three bare, unparameterized
registered tactics.

The learned policy had only 31 real training states,
20 with any candidate progress, from four modules.
The source success and kernel verification infrastructure
are sound within these constraints. Neural advantage
over correctly functioning Lean-native search is
not established.

A more informative next corpus should include
typed parameterized tactic actions from original
training proof steps, a larger training source,
and a real Lean tactic search baseline within
a calibrated, warmed-up shared compiler process.
The original theorem corpus remains independently
kernel checked, not proof of any unformalized mathematical claim.
