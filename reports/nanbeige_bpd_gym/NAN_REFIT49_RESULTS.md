# REFIT49 initialization and native-input audit

On a common pass-1 input difference, the finished T2 NF4 and BF16
BPD banks produce donor write differences with median cosine
**0.9998**. On their own native inputs, the same banks give a median
cosine of **−0.490**. The symmetric decomposition assigns nearly all
of the pass-1 disagreement to the reached input, with the median
operator-change norm below **1%** of the total disagreement.

## What agreement the fitted banks inherited

The four IO18 runs each contain sixteen audited optimizer steps.
T1 and T2 within a backend start from the same SVD and use different
recipes. The fitted banks reproduced the initializer's optimal rank-one
component assignment in all four comparisons. These assignments allow
reciprocal sign and scale changes within each rank-one atom.

| Pair | Initial full-operator cosine | Fitted full-operator cosine | Learned-update-only operator cosine |
|---|---:|---:|---:|
| NF4 T1 vs T2 | 1.000000 | 0.999704 | 0.628 |
| BF16 T1 vs T2 | 1.000000 | 0.999715 | 0.683 |
| T1 NF4 vs BF16 | 0.989963 | 0.989910 | 0.904 |
| T2 NF4 vs BF16 | 0.989963 | 0.989938 | 0.837 |

The update norms range from 1.99% to 3.27% of the initializer’s
Frobenius norm. They place only 1.96% to 5.64% of their squared energy
inside the initial joint read/write core, and 0.41% to 2.05% on
the eight original paired SVD atoms. The fitted atom correspondence
can survive a small update that points mostly outside the initial
core. Initializer agreement is an essential control for interpreting
the fitted atom match.

## Reached states explain the precision difference in this panel

These are medians over 16 already-exposed COMP14 focused DEV roots,
with the A/B down-projection input differences observed on two
physical passes. The symmetric additive decomposition averages the
input effect over both fitted operators and the operator effect over
both native inputs.

| T2 measurement | Pass 1 | Pass 2 |
|---|---:|---:|
| Cosine of native A/B input differences, NF4 vs BF16 | 0.021 | 0.865 |
| Fitted donor write cosine with one fixed NF4 input carrier | 0.9998 | 0.9988 |
| Fitted donor write cosine with each precision’s native inputs | −0.490 | 0.888 |
| Operator-only difference norm / total native disagreement | 0.010 | 0.078 |
| Signed input contribution projected onto total native disagreement | 1.001 | 1.000 |

The projection shares sum to one and can be negative or exceed one
because the input and operator vectors can oppose one another.
The initializers themselves gave a median of −0.490 on the native
cross-precision pass-1 comparison. Restricting the final banks to
the seven aligned DELAY39-style writes raised that median to 0.066,
with the common-input comparison still near 0.996.

The T1-versus-T2 learned-only donor updates on the same native
within-precision inputs had median pass-1 cosines of −0.045 NF4
and 0.036 BF16. These small updates are distinct from the large
SVD-initialized common direction. Qualification of a delayed donor
edit should bind both the factorization and the state that reaches
it. A cross-precision transfer would need to remeasure donor geometry
on the recipient’s native state and preserve the same physical dose.

The audit checked four fit receipts and 32 archived native worlds
per precision. Dense references checked the compact update
projections and the exact rank-one assignment. A second calculation
reconciled 116 reported transport cells and checked the symmetric
projection sums. The initial reader rehearsal was corrected to
match the IO18 audit’s final-checkpoint hash field before scoring.
Peak sampled private commitment was 72.9 MiB. There were zero
new decoder forwards, zero formal tests, and no access to the
sealed ARG-03 test.

[Protocol](../../experiments/nan_refit49/PROTOCOL.md) ·
[Aggregate results](nan_refit49_results.json)
