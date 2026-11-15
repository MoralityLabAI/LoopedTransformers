# PEF-03 | Frozen tied-attention history-retention results

## Provenance and experimental status

- This campaign's [preregistration](PRE_REGISTRATION.md) was committed as
  \`0394417c0aecbfb0d7d7c74c1db82ff59f3aec44\` **before PEF-03 training
  or evaluation**.
- Original successful [two-seed registered hosted run 43680103769](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43680103769).
- Repeated hosted [run 43680455308](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43680455308)
  additionally includes the explicitly **posthoc** paired-probe replay;
  complete artifact \`pef03-frozen-weights-and-causal-receipts\`,
  artifact ID **14148351667**.
- The head task used exactly 250 CPU optimizer steps on two fixed seeds
  (17 and 29). The 256 unique eight-bit histories were SHA256 split into
  **148 parameter-training, 56 probe-training, and 52 heldout** histories.
  Target labels encode only the two most recent bits.
- Trained model: tied 16-dimensional causal attention/MLP with one
  transient input token per visit and one persistent memory token;
  eight loop visits. Model state and checkpoints were SHA256-sealed
  and unchanged through evaluation.

## Registered measurements from run 43680455308

| Metric | Seed 17 | Seed 29 |
| --- | ---: | ---: |
| Two-bit heldout accuracy | 51/52 | 52/52 |
| Oldest-bit uniform heldout linear probe | 40/52 | 52/52 |
| Oldest-bit discrete flip, mean final hidden L2 | 1.7395 | 2.8761 |
| Newest-bit discrete flip, mean final hidden L2 | 19.4818 | 17.5652 |
| Oldest-bit flip changes predicted task class | 2/52 | 0/52 |
| Largest ordered Jacobian/full-recurrence JVP relative error | 9.94e-7 | 2.32e-6 |
| Random-initialized head accuracy | 8/52 | 13/52 |

The seed-29 model retains an old-bit signal recoverable by a linear
probe despite the supervised task head ignoring the oldest bit at the
level of its predicted class. Hidden effects and raw task logits
still change, so zero *class changes* must not be interpreted as
zero output-level causal influence.

## Clearly posthoc paired causal-decoder replay

The paired control was selected **after** examining the registered
run. It fits the oldest-bit probe on the original 56 probe-only
histories and evaluates each of the same 52 heldout histories together
with its counterfactual created by toggling only the oldest bit.
All seven newer bits and the supervised task label are fixed.

| Paired result | Seed 17 | Seed 29 |
| --- | ---: | ---: |
| Old-bit probe simultaneously correct on original and flip | 29/52 | 52/52 |
| Old-bit probe predicted label flips | 29/52 | 52/52 |
| Task-head predicted class unchanged | 50/52 | 52/52 |
| Label-only control probe predicted label flips | 0/52 | 0/52 |

The oldest-bit probe in seed 29 is correct on both members of all
52 controlled pairs, including across the four supervised target
classes. This goes beyond a source-correlation or task-label-only
explanation of the *probe*, while remaining limited to this
synthetic architecture, training regime, and finite data set.

Both saved checkpoint bytes were independently compared with their
recorded SHA256, and a separate local checkpoint replay reproduced
the target task accuracy and old-bit causal hidden-state differences.

## Source-distribution and precision boundaries

A separate Markov source with persistence 19/20 was used for
observational probes. The source-probe and source-evaluation samples
contain repeated eight-bit histories, so its very high scores do not
supply independent history-disjoint evidence. Recent-bit-only
analytical and empirical priors are retained as controls.

Readout experiments measure **FP16/BF16 recurrent storage casts** with
network computations restored to FP32. They do not simulate
full low-precision arithmetic or establish a universal bit-depth
capacity law for learned networks. The affine PEF-01/02 formulas
remain mathematical null models.

## Cross-run reproducibility

The original successful hosted run 43680103769 showed the same
qualitative pattern, with 51/52 and 52/52 task accuracy. It recorded
oldest-bit linear-probe accuracy **37/52 and 51/52** across seeds,
versus 40/52 and 52/52 in the later hosted run. The corresponding
oldest-bit hidden L2 means also differ slightly. The CPU workers
and numerical execution can produce different final training weights
even under identical seed labels. For reproducible scoring, freeze
and identify the **exact checkpoint SHA256** and rerun its evaluation,
rather than treating a training seed alone as a complete artifact ID.

Latest hosted checkpoint hashes:
- Seed 17: \`c174fbfd8bef4fa026806e62e66ad48a38827935c1121a487514de9948f61432\`
- Seed 29: \`76dfcf7c401a8d60ae4570cf281b0aa788362bb76bde7f0c040438333a5a6e01\`

## Interpretation and next experiment

The trained models mastered the two-bit task, while older input
information survived in hidden-state effects. At least one seed
yielded a causally responsive, nearly perfectly linear-decodable
older-bit channel with near-total task-class invariance. This is an
initial **task-irrelevant latent-memory mode**, in a tiny trained
tied-attention model.

A fresh preregistered campaign should use several independently
trained models, stronger history-disjoint probes, conditional
within-target-class controls, and natural language or formal-math
tasks. The next dependent measurement is to identify the memory
directions via sparse features, patch them into target tasks, and
evaluate whether those directions support capability or safety
signals. This PEF-03 result makes no claim of improving heldout
Lean proof performance or predicting pretrained LLM behavior.
