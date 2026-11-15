# RAM-05 verified bounded-tactic CPU training | 2026-11-15

The original [RAM-05 training-only CI #43675801084](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43675801084)
completed its source-only audit and all nine model fits before any RAM-05
held-out theorem script was submitted to the scorer. Its training
runner source commit was `033f554c761f50c4c6e4fa00e049a0481144830c`.
The original frozen train-only artifact
`ram05-frozen-audited-train-43675801084` has 18,901,757 compressed
bytes and GitHub artifact SHA256
`56a8a7544e43e1d4a2192ec707b6a7cb64710685e75db906dce12a54e9d4468b`.

## Fresh true compiler labels

Original MathCan train-root source
`c84e2025334e9a967c1297425a490f98d5b3d189`, Lean4.35.
The original FC-12 51 source states and 961 compiler outcomes were
required as immutable inputs. The new RAM-05 collector used a distinct
17-family bounded-only script vocabulary and audited every new
successful or unsuccessful observed transition in 27 fresh Lean
sessions before fitting models.

| Training portion | Roots | Real Lean goal states | Multiple-goal states | Fully replayed candidate outcomes |
| --- | ---: | ---: | ---: | ---: |
| Model optimization | 22 | 171 | 53 | 4,083 |
| Development-only selector | 5 | 42 | 14 | 1,040 |
| Total | 27 | 213 | 67 | 5,123 |

The *distinct* goal-state total is **213**, compared with the
209-state prior RAM-03/04 training corpus. Different state
exploration follows from the new candidate ordering and label
observation; the original previous dataset is preserved rather than
silently overwritten.

Compiler-probed and independently replayed SHA256:
`45d2c7d7d29bf17c48a09c431a3cbe295f78690c10baf6c01edef69221ce7b62`.
Exact RAM-05 config SHA256:
`39dc726206af0b6b9078d879561082585089fd24e39e5e47c732be6213b221a8`.
The independent audit checked `5,123` rows and found **zero**
compiler outcome disagreements.

## Parameterized training supports

These counts are actual compiler-observed training candidate progress
(outcomes >0), not certified complete theorems.

| Family | Positive train | Positive development |
| --- | ---: | ---: |
| `exact_local` | 0 | 0 |
| `rw_local` | 49 | 0 |
| `rcases_local` | 2 | 0 |
| `induction_list` | 53 | 27 |
| `simp_only_visible` | 93 | 32 |
| `simp_all_only_visible` | 197 | 41 |
| `dsimp_only_visible` | 161 | 41 |

The regenerated RAM-05 dataset **does not contain any positive
`exact_local` training example**. The original FC-12 source corpus
independently has a positive exact-application example, but RAM-05
did not inject that historical label into its newly collected training
effects. The previous training-coverage achievement remains valid for
FC-12; RAM-05's newly trained exact capability is unsupported.
Any later supplemental exact-specific data and model refit require
a new named experiment or clearly labeled pretest amendment with
all original attempts retained.

## Original nine CPU fits and model selection

The existing four-pass tied recurrent architecture was used with
the new 17-family action IDs. The same RAM-03 loss and original
five-theorem development selector chose one seed per architecture.
All three chosen fits used seed29.

| Trained model | Parameters | Selected seed | Best step | Balanced dev loss | Exact best checkpoint SHA256 |
| --- | ---: | ---: | ---: | ---: | --- |
| Looped32 | 85,061 | 29 | 200 | 0.806983 | `31f5e33952b737ba4ac52b6c62682dadbd5cf4d5fabcf0aff10f7404273c5b77` |
| Looped64 | 206,981 | 29 | 200 | 0.905749 | `1bbdd51f1025cf6d4b1335f79e97fd2124f9d68ee4b0510259135bd89b18cf3b` |
| Flat64 | 182,085 | 29 | 100 | 0.783007 | `ff8830e9ce3bc3befcca7cfc282fbf140e08cb606a7bd2ce08c37e7c98bfacaf` |

The original output contains all nine best/last model
weights, optimizers, RNG states, validation receipts and exact
SHA256 checks. Model selection was completed before the
first prospective theorem state was evaluated.

## Memory and throughput

CI used Python3.11, CPU PyTorch2.6.0, original Lean4.35,
two CPU training threads. No CUDA device was present.

- 30-step calibration: **54.736978** optimizer updates/sec.
- Peak sampled process-tree RAM: **417,738,752 bytes (398.39 MiB)**.
- Lowest sampled available system RAM: **7,038,394,368 bytes**.
- No resource-limit abort or nonfinite loss.
- Nine original fits used effective batch64 and predeclared early stop.

The low-memory safety target is met on this GitHub CPU runner.
These measurements do not constitute new RTX3050 GPU
benchmarks; earlier Windows RAM-02/03 runs found CPU faster
for similarly small networks.
