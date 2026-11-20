# Six missing native rank/site cells: random pipeline preflight

Before any pretrained rank/site extension, run every missing cell on a random
native 22-layer model: down_proj rank16 T1/T2, o_proj rank8 T1/T2 and o_proj
rank16 T1/T2. Existing down_proj rank8 fits are historical controls and are not
silently rerun. No donor transfer, state patching or new competence claim.

Use the actual pinned BPDTrainer and RECOMP09 memory implementation, preserving
all eight optimizer steps and loss coefficients, two PGD inner steps, fixed
CAL order, before/after CAL means, all-one state/logit closure, nonzero gradient
gate, initial/final diagnostics and static/clean/reached-state masks. The new
rank dimensions affect schedules, initialization, controller outputs and finite
positive diagonal/permutation gauge checks. Original helper functions and frozen
files remain unchanged. Generated source is inspectable; no global patching.

The random model uses seed 2026112013, width64, MLP128, eight query heads,
two KV heads, head_dim16 and full 166144-token head. Thus o_proj has a distinct
128-wide input and 64-wide output. CPU embeddings, four resident BF16 layers
plus reusable stage, unchanged BF16 GPU head. Use all 32 public FIT07 IDs/splits
only as engineering metadata, with freshly generated fixed random 19-token
histories (integers 0..99). No pretrained tensors or host labels are opened.
Initial factors are the exact small random module's rank8/rank16 SVD. Every
cell starts from the same frozen target, with no factor/controller carryover.

Require completion of each full eight-step recipe, exact all-one vocabulary
logits and normalized states, factor/residual closure <=1e-5, finite nonzero
pre-optimizer factor gradients, positive factor/controller updates, unchanged
target digest and finite gauge checks stable at 1e-5. Preserve every cell and
all failed/partial artifacts. Continue the remaining cells if one fails; do not
change a bound, drop a cell or rerun a result because it is unfavorable.

Resource policy: one numerical CPU thread, Below Normal, CUDA allocator cap .05
and added saved-tensor CPU cap64 MiB, 1200 seconds per cell checked between
stages. Confirm >=512 MiB device headroom before starting. Only this tiny random
fixture may overlap the registered large model; no second pretrained model.

This tests actual rank/site framework integration, not learned component
quality or mechanistic specificity. No o_proj-to-layer-residual identity bridge
is valid: the following MLP intervenes. A future real o_proj/SAE comparison must
use a registered, validated source-dependent Jacobian/finite-write bridge or
remain visibly missing. Do not substitute matching output widths for that bridge.

Unit: one random target/seed, six correlated cells, eight synthetic CAL and
eight DEV metadata roots with copy controls. Physical causal dose zero; no
scientific/sham floor or causal power calibrated; formal comparisons zero.
The original 168-case test remains sealed. Only after all engineering checks
pass may a complete separate real-checkpoint protocol/source seal be frozen
and pushed, retaining the original BF16 eligibility and failed NF4 gate.
