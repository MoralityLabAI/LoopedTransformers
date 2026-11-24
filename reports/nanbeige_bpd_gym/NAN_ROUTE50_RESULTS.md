# ROUTE50 input-dependent routing at equal soft budget

A calibration-selected static component preference produced lower
local write error than the learned IO18 T2 controller on **all 16
focused DEV roots** in both precisions and both physical passes.
Each alternative used the learned gate's exact total mass on that
same captured input. The constant component preferences were chosen
from 258 candidates on the eight focused CAL roots and held fixed
during DEV evaluation.

## Direct physical write comparison

| Precision and pass | Learned error | CAL-selected static error | Learned twin-swap error | Initial controller error |
|---|---:|---:|---:|---:|
| NF4 pass 1 | 0.053829 | 0.043409 | 0.053829 | 0.053697 |
| NF4 pass 2 | 0.289371 | 0.283079 | 0.289445 | 0.289328 |
| BF16 pass 1 | 0.042346 | 0.039475 | 0.042434 | 0.041769 |
| BF16 pass 2 | 0.306610 | 0.291442 | 0.306529 | 0.305233 |

Errors are relative L2 against the native last-token down-projection
write on the same archived clean input. The optimized static arm
used a phase-specific preference and access to the learned controller's
scalar mass, so the comparison isolates component allocation given that
mass. It also won on all 16 roots of every copy and erased DEV cell.
The strongest measured gains are local write improvements. Native
execution cost and downstream success require separately matched tests.

The learned masks changed across focused twins. The mean largest
slot difference increased from about 0.001 at pass 1 to 0.012 in
NF4 and 0.024 in BF16 at pass 2. Swapping the learned twin preferences
had little effect on the relative write error in the table, and all
erased twins had exactly identical mask vectors. The data show
input-dependent gates with little observed twin-specific value for
the local-write metric under the registered soft budget.

## Original native outcome check

A separate pairwise recount of the original IO18 diagnostics, under
their original unmatched-dose routing policies, compared both
adaptive arms with the original all-token static mask at pass 2.
For either adaptive arm, NF4 had 25 jointly correct worlds, six jointly
wrong worlds, and one static-only correct world. BF16 had 32 jointly
correct worlds. The adaptive arms gained zero native exact answers over
that original static control. This check is separate from the new
mass-matched local comparison.

## Verification and next measured contrast

The native CI fixture closed within 9.3e-8 in absolute mask value.
The pass-1 archived clean-informed and static local write errors
replayed within 1.7e-7 absolute. Capped-simplex mass closure before
the float32 write calculation was within 7.2e-15. An independent
recount verified all cell means against the private row receipts.

The result uses 16 exposed DEV roots per precision. Worlds, passes,
arms and precisions repeat those roots. The CPU replay peaked at
125.2 MiB private commitment, with zero new decoder forwards and
zero inferential tests. The sealed ARG-03 test was untouched.

For a native test of adaptive routing, carry forward the optimized
per-phase static component preferences as a control and match the
physical dose at the reached state. Use separate fresh source roots
to test whether the learned allocation changes the later margin or
a verified exact outcome.

[Protocol](../../experiments/nan_route50/PROTOCOL.md) ·
[Aggregate results](nan_route50_results.json)
