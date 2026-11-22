# REFIT49 initializer-controlled BPD stability

Audit all four completed IO18 rank-eight factorizations. Keep the
backend-specific SVD initializer and the final checkpoint for both
the T1 and T2 recipes. Require the existing independent fit audit,
target digest, checkpoint hashes, exact SVD initialization and
16-step optimizer receipts. T1 and T2 share an initializer within
precision and change both depth and training recipe.

Compare paired rank-one operators by the product of their normalized
read and write inner products. Choose a deterministic exact one-to-one
assignment that maximizes the sum of absolute atom cosines. Compare
assignments chosen from the two initializers and from the fitted
banks. Report signed scores under that alignment, principal angles of
the full read/write spans, and physical operator differences. Read
and write sign and scale factors can exchange inside each atom.

Separate the learned update D = V_final U_final − V_init U_init
from the full operator. Report its Frobenius norm relative to the
initializer, the cosine between paired updates, and how much squared
update energy projects into the whole initial read/write core and its
eight paired SVD rank-one atoms. The initializer's directions are the
common reference for that fit. Compute these small projected quantities
without materializing the full dense decoder matrix.

On the 16 already-exposed focused COMP14 DEV roots, use the difference
between the A and B last-token layer-21 down-projection inputs in each
physical pass. Compare the two induced write differences under each
final bank and initializer, and compare the learned-only differences.
Within precision the banks receive the same captured inputs. Across
precisions report each common input carrier and the native-carrier
pair, so apparent disagreement has a separate input-state comparison.
Retain near-zero denominator failures explicitly, and report one root
mean per phase and comparison. No labels, outcome-guided fitting or
post-selection enter the geometry calculation.

The first replay motivated a separate descriptive decomposition.
For the native cross-precision carriers, average the carrier change
over both fitted operators and the operator change over both captured
carriers. The two physical vectors sum to the observed difference.
Report their norms relative to the total difference and their signed
projections onto that difference.

This is an initializer-controlled observational comparison over one
checkpoint at two precisions and four recipe runs. The results can
separate shared physical geometry from update-specific agreement.
Mechanism invariance would require independent initializations,
matched policies and causal transfer. The original ARG-03 test is
outside the audit. Use the lightweight CPU lease with zero decoder
forwards and zero formal comparisons.
