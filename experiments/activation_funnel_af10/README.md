# AF-10: exact causal-cut probe for looped Horn proof state

The source-hash-pinned original `TypedLoop` can propagate a downstream rule token back into both logically independent subgoal atom tokens, and its neural Q-to-Q Jacobian can be nonzero. AF-10 introduces an **optional role-selected causal cut** at the carried-code interface during downstream recurrence. It restores both Q codes to their source (recipient/donor patched) boundary values after each native cell call and code encoding, before the following decode. The original model, weights and unlatched control remain unchanged.

The [causal-cut theorem](../../docs/research/AF10_CAUSAL_CUT_THEOREM.md) proves exact noninterference and prefix-state factorization. The new code retains all hidden and public graph semantics; synthetic Horn oracle truths continue to come from AF-05's least fixed point. The model receives no derived label as an input.

## Run

Use Python 3.12, Torch 2.5.1+cpu and NumPy/SciPy as pinned by the parent. From the repository root:

```bash
python -m unittest discover -s experiments/activation_funnel_af10/tests -v
python -O -m unittest discover -s experiments/activation_funnel_af10/tests -v
python -m experiments.activation_funnel_af10.diagnostic --source-zip /path/to/original_af09_seed17_training.zip --seed 17 --out artifacts/af10/s17
python -m experiments.activation_funnel_af10.verify --source-zip /path/to/original_af09_seed17_training.zip --seed 17 --evaluation artifacts/af10/s17 --out artifacts/af10/s17/independent_audit.json
```

The pinned original AF-09 GitHub Actions run is `43682916363`. The diagnostic verifies the exact training ZIP SHA, original model/checkpoint exports, source ledger, and development qualification before reading any held-out input. Two frozen seeds and three original arms are retained. There is no optimization or new model superiority hypothesis in AF-10.

The original AF-05 depth-4/depth-5 graphs were examined when choosing this architecture, so reported changes are retrospective diagnostics. New efficacy claims will require a prospective AF-11 corpus, compute-matched training with the cut active, and uninspected graph families.
