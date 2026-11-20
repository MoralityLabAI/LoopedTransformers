# Native BF16 learned rank/site results

5/6 conditions are fully completed and audited in this committed snapshot. All six remain listed. Blank metrics mean no corresponding closed audit, not a measured zero.

The fits reuse eight exposed CAL and eight exposed DEV roots from FIT07 on one checkpoint. Training uses each CAL main root once; DEV/copy examples are not training inputs. NF4 remains excluded by its failed fixed fitting screen. Original algebraic gates remain failed.

| Condition | Closed status | Audited steps | Factor update L2 | Controller update L2 | Added saved-tensor CPU peak GiB |
|---|---|---:|---:|---:|---:|
| down_proj r16 T1 | completed_and_audited | 8 | 0.201667 | 0.172222 | 0.210012 |
| down_proj r16 T2 | completed_and_audited | 8 | 0.168751 | 0.188456 | 0.525277 |
| o_proj r8 T1 | completed_and_audited | 8 | 0.111095 | 0.115184 | 0.193461 |
| o_proj r8 T2 | completed_and_audited | 8 | 0.10633 | 0.127596 | 0.487765 |
| o_proj r16 T1 | completed_and_audited | 8 | 0.158125 | 0.115801 | 0.19407 |
| o_proj r16 T2 | fit_audited_physical_probe_in_progress | 8 | 0.154516 | 0.128328 | 0.489165 |

Update magnitudes depend on factor parameterization. The CPU column measures added saved tensors under the unchanged 2-GiB cap; it is not whole-process RAM.


## All DEV routing controls

| Condition | Loop | All-one | Component only | Static CAL mean | Clean-informed | Actual reached-state |
|---|---:|---:|---:|---:|---:|---:|
| down_proj r16 T1 | 1 | 2/8 | 2/8 | 2/8 | 2/8 | 2/8 |
| down_proj r16 T2 | 1 | 2/8 | 2/8 | 2/8 | 2/8 | 2/8 |
| down_proj r16 T2 | 2 | 7/8 | 7/8 | 7/8 | 7/8 | 7/8 |
| o_proj r8 T1 | 1 | 2/8 | 2/8 | 2/8 | 2/8 | 2/8 |
| o_proj r8 T2 | 1 | 2/8 | 2/8 | 2/8 | 2/8 | 2/8 |
| o_proj r8 T2 | 2 | 7/8 | 7/8 | 7/8 | 7/8 | 7/8 |
| o_proj r16 T1 | 1 | 2/8 | 2/8 | 2/8 | 2/8 | 2/8 |
| o_proj r16 T2 | 1 | 2/8 | 2/8 | 2/8 | 2/8 | 2/8 |
| o_proj r16 T2 | 2 | 7/8 | 7/8 | 7/8 | 7/8 | 7/8 |

## All CAL routing controls

| Condition | Loop | All-one | Component only | Static CAL mean | Clean-informed | Actual reached-state |
|---|---:|---:|---:|---:|---:|---:|
| down_proj r16 T1 | 1 | 2/8 | 2/8 | 2/8 | 2/8 | 2/8 |
| down_proj r16 T2 | 1 | 2/8 | 2/8 | 2/8 | 2/8 | 2/8 |
| down_proj r16 T2 | 2 | 7/8 | 7/8 | 7/8 | 7/8 | 7/8 |
| o_proj r8 T1 | 1 | 2/8 | 1/8 | 2/8 | 2/8 | 2/8 |
| o_proj r8 T2 | 1 | 2/8 | 1/8 | 1/8 | 1/8 | 1/8 |
| o_proj r8 T2 | 2 | 7/8 | 6/8 | 7/8 | 7/8 | 7/8 |
| o_proj r16 T1 | 1 | 2/8 | 1/8 | 2/8 | 2/8 | 2/8 |
| o_proj r16 T2 | 1 | 2/8 | 1/8 | 1/8 | 1/8 | 1/8 |
| o_proj r16 T2 | 2 | 7/8 | 7/8 | 7/8 | 7/8 | 7/8 |

Each fraction uses eight main contexts from eight roots at that split. Invalid outputs stay in the denominator. The same roots recur across routes, loops, ranks and sites. Static controls retaining correctness limit behavioral specificity. T1/T2 differ in fit depth, ablation and adversary rules; differences do not isolate a BPD advantage.

## Audited SAE cross-reference coverage

| Condition | Signed finite writes | Finite feature groups | Derivative groups | Cached vectors | Cached groups |
|---|---:|---:|---:|---:|---:|
| down_proj r16 T1 | 9216 | 72 | 36 | 644 | 28 |
| down_proj r16 T2 | 9216 | 72 | 36 | 644 | 28 |
| o_proj r8 T1 | 4608 | 72 | 36 | 644 | 28 |
| o_proj r8 T2 | 4608 | 72 | 36 | 644 | 28 |
| o_proj r16 T1 | 9216 | 72 | 36 | 644 | 28 |
| o_proj r16 T2 | — | — | — | — | — |

Finite probes cover all 16 main fitting prefixes, both native loops, three bases, every component, requested relative doses .001/.01/.1 and both signs. Groups retain all 128 features, including rounded-away writes. o_proj responses use the actual residual/norm/MLP path; they have no identity-coordinate geometry cells. JVPs remain derivative surrogates, with all finite-write discrepancies preserved. The CPU audit checks archive/dose/coverage/native additions; native MLP/JVP outputs are source-bound worker measurements.

The cached suite retains 644 vectors/89 older source groups, all 28 suite/split/copy/loop strata and all 128-by-rank correlations. These clean-input probes are observational. T1 application at loop2 is outside fit depth. Full feature arrays, source groups and derivative diagnostics remain in the linked aggregate; none are selected by outcome.

## Registered physical operator comparisons

| Module/rank | Relative V@U difference | V@U cosine | Output-subspace principal cosine range |
|---|---:|---:|---|
| down_proj r16 | 0.0164246 | 0.999865 | 0.999306–1 |
| o_proj r8 | 0.00553064 | 0.999985 | 0.999943–0.999998 |
| o_proj r16 | 0.0040618 | 0.999992 | 0.999951–0.999999 |

These compare physical module-output operators/subspaces, not individual factor columns or an unbridged o_proj-to-SAE identity. Shared SVD initialization and small updates can produce high alignment. No pooled accuracy, inferential comparison, equivalence, semantic mediation or donor transfer is claimed. Causal effect/sham floors and power remain uncalibrated. The original 168-case test stays sealed.

Source commit: `7e7b5fc77aec228f13dab72b07f9e105758c8842`. Aggregate SHA-256: `ce5e0c96d2b61694c739f9106a8548f1a35ac9f7d062d680a95f8d80172f7b16`. Producer SHA-256: `b7f46d98e1732cd48b58e5706c373243e812e6bb2126945f9dc1bb25fdcad1a5`. Experiment seal: `27b3e5e636abcdf9217f88a5f19999c46b56d9723c76e6f5263c781e93239ddc`. HOST01 registration: `e93a1df940cc8c263caff6a5c7dfa3acf7395d9a20e01cdcb07017753a129a7a`.

[Exact source aggregate](https://github.com/MoralityLabAI/LoopedTransformers/blob/7e7b5fc77aec228f13dab72b07f9e105758c8842/reports/nanbeige_bpd_gym/nan_ranksite13_progress.json) · [Latest worker/audit status](NAN_RANKSITE13_PROGRESS.md) · [HOST01 amendment](../../experiments/nan_ranksite13_host01/PROTOCOL.md) · [Compact machine-readable summary](nan_ranksite13_results.json)

This is a post hoc presentation of committed audited endpoints, prepared after the first three completed cells were exposed. It reads no raw runs or live progress files and performs zero model calls or new grading. Statuses describe closed artifacts, not current process liveness.
