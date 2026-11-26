# MOLT-06 measured development result

[Corrected fixed CPU run](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45850366924)
passed at source head fc403ea05b88bab5d2fe70d820e83bd81662a5f1. Four model
initializations, 64 shared development worlds, 3,072 scored configurations.
The numerical ranks (terminal, full stack) were (2,5) at every corrupted cut.
All 512 stacked-observer matches passed; no zero dose or unmatched cases.
All four trained models solved all affected bits on every intact world.
Corruption destroyed cone correctness on every world; full restoration
reproduced the intact logits. The analytic Jacobian and causal-cone checks passed.

| Intervention | L8 corrupt /128 | L8 clean /128 | L12 corrupt /128 | L12 clean /128 |
| --- | ---: | ---: | ---: | ---: |
| Target rank four | 94 | 91 | 89 | 97 |
| Terminal sham | 2 | 93 | 3 | 95 |
| Full-stack sham | 95 | 92 | 89 | 98 |

Balanced target minus terminal sham: +90/256 (L8), +88/256 (L12).
Balanced target minus full-stack sham: -2/256 (L8), -1/256 (L12).
This is a striking demonstration of the inadequacy of a terminal-only control
for a multi-readout task. The full-stack comparison supplies no donor-specific
superiority. The models share evaluation worlds; these counts are descriptive
development evidence, not independent Bernoulli samples or confirmation.
The linear observer rank is measured in a fixed chart and threshold; no exact
nonlinear quotient or native BPD mechanism is established.

Next: cache and replay local observer bases under explicit serialized-byte
budgets, while measuring the actual nonlinear loss of discarded directions.
