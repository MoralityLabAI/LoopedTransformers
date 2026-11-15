# RAM-01 CPU training results | 2026-11-14

All four fixed model configurations were trained with seeds 17 and 29 in the interactive CPU sandbox and again in [GitHub Actions run 43557076648](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43557076648). Each environment completed eight development fits and eight final fits. The eight final CI checkpoints are preserved in artifact `ram01-cpu-models-43557076648`.

## Measured training cost

| Environment | Runtime | CPU threads | Completed fits | AdamW updates | Training campaign wall time | Peak process RSS |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Interactive cloud sandbox | Python 3.13.5, PyTorch 2.10.0+cpu | 2 | 16 | 6,400 | 27.877 s | 447.66 MiB |
| GitHub Actions | Python 3.11.18, PyTorch 2.10.0+cpu | 2 | 16 | 6,400 | 33.852 s | 519.54 MiB |

The interactive sandbox's cgroup limit was 4 GiB with four CPU cores of quota. No CUDA device was used. Reported time covers the training driver, including both fit phases and cached outcome scoring; package installation and live Lean proof searches are separate. Peak RSS is the high-water mark of the whole training process, not isolated per-model memory.

Each fit used 400 AdamW updates with batch 64, learning rate 0.002, weight decay 0.01, and gradient clipping at norm 1.0. All saved checkpoints passed exact output comparison after CPU reload in their training environment.

## Data and supervision

The verified FC-12 training artifact contains 51 authentic Lean goal states from 27 original MathCan theorems and 961 concrete candidate probes. The outcome head preserves four classes: 844 failed or unchanged candidates, 38 single-goal advances, 32 multi-goal advances, and 47 scratch closures.

A model receives the printed context and conclusion, plus the concrete candidate script and its tactic-family ID. It predicts an outcome distribution. Candidate ordering uses `2*P(scratch closure) + P(single-goal progress)`. The score accounts for the current controller's inability to continue multi-goal branches. Multi-goal actions retain their successful transition label during training.

The development fit uses five complete source modules: 39 states and 766 candidate rows. DeletionMemory and TOECosmology supply the 12-state, 195-candidate development split. Every prefix of a theorem remains within one split. Exact-state hash overlap is rejected. Training features exclude filenames, theorem identifiers and original proof commands.

The development training subset contains positive examples for every mandatory action family: exact 1, rewrite 9, rcases 1, list induction 30. Full final training uses all 51 states, including 13 successful rewrite probes. Exact and rcases support remains extremely sparse.

## Cached action-choice results

A first choice counts as executable when its recorded compiler probe closes the scratch goal or advances to a single remaining goal. These counts do not certify complete theorem proofs. Exactly eight of the twelve development states have any executable candidate in the fixed grammar.

| Policy | Parameters | Sandbox seed 17 | Sandbox seed 29 | CI seed 17 | CI seed 29 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed native order | 0 | 3/12 | 3/12 | 3/12 | 3/12 |
| Training-only family frequency | 0 | 6/12 | 6/12 | 6/12 | 6/12 |
| flat64 | 173,252 | 7/12 | 7/12 | 7/12 | 7/12 |
| loop32, four shared passes | 82,692 | 6/12 | 5/12 | 6/12 | 5/12 |
| loop64, four shared passes | 198,148 | 7/12 | 7/12 | 7/12 | 7/12 |
| loop128, four shared passes | 527,364 | 8/12 | 5/12 | 7/12 | 5/12 |

The largest model's seed-17 fit differed across environments. Tensor comparison confirmed different trained weights, so this is not merely a checkpoint archive-format difference. The exact numerical cause was not isolated. Loading the CI development checkpoints into the cloud sandbox reproduced all eight CI top-1 score counts.

Every configuration was subsequently fitted on all 51 training states and retained. Development results did not select, tune or stop the final fits. Final-fit cached scores are resubstitution diagnostics only.

## Live proof regression

The separate [Lean regression run 43557331256](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43557331256) evaluates all eight final CI checkpoints and a fixed native comparator on the previously published FC-13 sixteen-theorem suite. It uses the same twenty-second deadline, twelve actual Lean subprocess calls, and three-transition ceiling for each controller. It archives source scripts and compiler logs, and freshly recompiles each accepted proof.

Proof release also requires `#print axioms` to report only `propext`, `Classical.choice`, or `Quot.sound`, or no axioms. Its real Lean preflight includes a custom-axiom proof that compiles successfully and must still be rejected. These post-hoc regression results belong to RAM-01; the original FC-13 scores remain unchanged.
