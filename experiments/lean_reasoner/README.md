# LR-01: Proof-Compiled Looped Reasoner

Date: 2026-11-05 (Asia/Manila)

A CPU experiment in learning proof dependencies from a generated Lean corpus. The policy has **20,161 trainable parameters**. Its four-pass version completed **100% of standard held-out proofs** and **92.01% of deeper/larger proofs**, averaged over three seeds. Running that same four-pass-trained network for eight inference passes raised deeper/larger completion to **97.57%**.

## What was built

Each problem becomes a typed bipartite graph whose nodes represent propositions and implication rules. Hypotheses mark the propositions already available, and a separate feature marks the goal. There are no learned symbol-name or position embeddings.

A shared graph-attention cell updates rule embeddings from their premises and conclusion. It then updates proposition embeddings through separate incoming-rule and outgoing-rule channels. A scorer reads both kinds of final state and chooses an applicable rule. The executor adds the resulting proposition and constructs its proof term. Every successful rollout is replayed by a separate recursive type checker for this finite fragment.

The GCPT connection is explicit ownership: graph incidence and proof execution are exact; the neural component learns where to spend a proof step. Language generation is outside LR-01. This is an experimental composition of graph attention and certified execution; novelty and transfer to dialogue require separate evaluation.

Training targets come from the generated proof ASTs. Postorder traversal produces proof steps, with sharing for repeated subproofs. At each training state, the remaining proof dependencies supervise rule relevance. A second loss assigns probability mass to any currently applicable useful rule. Gold dependencies, depth labels and minimum proof lengths never enter the neural features.

## Corpus and splits

The generated fragment consists of acyclic propositional Horn implications, with one producing rule per noninitial proposition. All statements make their hypotheses explicit as Lean theorem parameters.

| Split | Theorems | Proof dependency depth | Propositions |
|---|---:|---:|---:|
| Train | 768 | 2–4 | 11–17 |
| Validation | 128 | 2–4 | 11–17 |
| Standard test | 192 | 2–4 | 11–17 |
| Depth/size OOD | 192 | 5–7 | 18–23 |

Training uses **2,333 proof states**. Atom identifiers and rule order are randomized. Whole-graph isomorphism-invariant WL fingerprints are disjoint across splits. Goal-subproof motifs can recur across splits, so these results concern held-out whole graphs rather than a proof-family-held-out benchmark. OOD changes both depth and graph size.

The 1,280-theorem `Corpus.lean` passed **Lean 4.19.0** in GitHub Actions. The source hash matched the locally generated file. `results/kernel_receipt.json` records that initial audit. The workflow also reconstructs and checks the complete 384-task rollout receipt for the prespecified four-pass seed 17. That receipt contains 373 successful proof certificates and 11 budget failures. Failed attempts are retained.

## Results

The metric requires reaching the goal within the exact minimum number of rule applications. The evaluator supplies the same oracle-derived budget to all policies. Exact applicability masking is common to the learned and random policies.

Each percentage averages three seeds, 17, 29 and 43, on the same 192 theorems per split. Consequently each column contains 576 runs over 192 distinct theorem graphs.

| Policy | Standard test | Depth/size OOD |
|---|---:|---:|
| Random applicable rule | 2.78% | 0.17% |
| Immediate goal, otherwise random | 12.15% | 0.35% |
| 1 pass, 300 updates | 77.08% | 26.39% |
| 1 pass, 1,200 updates | 77.60% | 27.78% |
| 2 passes, 300 updates | 99.13% | 68.40% |
| 4 passes, 300 updates | 100.00% | 92.01% |
| Train 4 passes, infer 8 | 100.00% | 97.57% |
| 4 passes, goal features removed at evaluation | 15.10% | 6.08% |
| Exact dependency oracle | 100.00% | 100.00% |

All neural arms allocate the same 20,161 parameters. The long one-pass control matches the four-pass arm's total shared-cell visits across training minibatches. FLOPs and wall-clock time are not exactly matched; the long one-pass control sees four times as many minibatches.

Four passes exceed the long one-pass control by 22.40 percentage points on standard tasks and 64.24 points on depth/size OOD. Paired theorem-bootstrap 95% intervals are [16.84, 28.12] and [57.81, 70.66] points, conditional on these three seeds. This bootstrap does not estimate uncertainty over a population of training seeds.

The four-pass OOD counts are 181, 169 and 180 out of 192. Eight inference passes give 181, 189 and 192 using the same weights. The seed-17 total stays unchanged, while the other two seeds improve.

## Interpretation

This pilot supports learning goal-conditioned dependency selection in a compact, shared operator. Additional passes let information travel farther through the typed graph. Increasing optimization updates at a fixed communication depth produced little improvement in this experiment.

The exact dependency oracle solved every task. LR-01 measures whether a small neural module can acquire this operation from proof examples. Large-model interrogation and natural-language commitment tracking remain untested. Quantifiers, alternative derivations and inconsistent premise sets require a broader successor corpus.

For the original dialogue application, a useful next task is to remove one premise from a verified proof and train the module to identify the missing obligation. A separate language interface can turn that obligation into a question. The proof module would preserve the assumptions needed for each conclusion.

## Resources and validation

Training ran on sandbox CPU with a 4 GiB cgroup memory limit, Python 3.13.5 and PyTorch 2.10.0+cpu, using two threads. Training-process peak RSS ranged from 350.81 to 377.05 MiB. The sum of the 12 recorded training durations is 217.57 seconds, excluding orchestration, cold imports and evaluation. The language runtime is included in RSS.

All nine local unit tests passed, including malformed-proof rejection, incomplete-proof rejection and neural equivariance under atom renaming plus rule permutation. The CI audit needs only Python and the pinned Lean toolchain; the torch-specific unit test is skipped when torch is absent.

Before sealed test evaluation, a validation-only engineering smoke found that the original scorer failed to use the final proposition states. The scorer was changed so every parameter participates even with one pass. Its excluded smoke receipt is retained in `development/`. Final checkpoints were used throughout, without test-driven checkpoint selection. Two shell calls hit tool time limits during orchestration; completed deterministic reruns supplied the reported evaluation.

## Reproduce on Linux or WSL

```bash
cd experiments/lean_reasoner
python -m pip install -r requirements.txt
python proof_data.py generate
python -m unittest -v test_proof_reasoner
for seed in 17 29 43; do
  for loops in 1 2 4; do
    python train.py train --loops "$loops" --seed "$seed" --steps 300
  done
  python train.py train --loops 1 --seed "$seed" --steps 1200
done
python train.py evaluate
python report.py
python audit.py
python proof_data.py verify
```

The final command requires Lean 4.19.0 on PATH. It fails explicitly when Lean is missing. `report.py` produces full summaries and the compressed seed-17 proof receipts. The committed `results/metrics.json` gives compact counts and resource records. `results/outcomes.tsv` stores one 192-bit, zero-padded hex outcome vector per split and seed; the leftmost bit corresponds to theorem index 0000.

Generated corpus files, full evaluation traces and model checkpoints are reproducible outputs. The accompanying run bundle retains them. Checkpoint SHA-256 values appear in `results/metrics.json`.

## Prior work and semantic basis

Lean's propositions-as-types representation supplies the proof-term interpretation: https://lean-lang.org/theorem_proving_in_lean4/Propositions-and-Proofs/

LeanReasoner (Jiang, Fonseca and Cohen, NAACL 2024) provides prior work on using Lean for logical reasoning: https://aclanthology.org/2024.naacl-long.416/

The LR-01 corpus is generated for this experiment. It contains no imported Mathlib training collection.
