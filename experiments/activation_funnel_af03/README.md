# AF-03: causally supervised compact recurrence

AF-03 fine-tunes the exact native `TypedLoop` cell from AF-02 with training gradients that pass through actual carried-code interventions. The first experiment uses five or ten float32 code coordinates (one or two per token). The decoded state still contains 80 working coordinates. The source has 5,590 trainable recurrent/adapter/head parameters; the codec adds a small separately counted capacity.

The [preregistered protocol](PROTOCOL.json) was frozen before running the primary three-seed, two-width matrix. AF-02 source checkpoints are verified against its frozen CI training manifest, and AF-02's source classes and gym stay unchanged. All arms begin from identical source weights and identical train-only PCA codec initialization.

**Training arms:** `task` receives only original host action supervision, `status` adds branch-status supervision, and `causal` additionally receives changed-prefix code-patch cross-entropy against AF-02's exact composed-history oracle, untouched-branch preservation, and an auxiliary output-commutation loss. All three optimize the original TypedLoop cell and codec end-to-end. The comparison matches seeds, minibatches, initial states, learning rate, steps, and code width, although the causal objective performs additional computations.

The causal patch replaces only the addressed atom and rule code after the second visit of the first preview. The donor matches the recipient's first branch, public depth, and schedule, and always has a different first-branch status. The recipient suffix remains unchanged. Whole-code patches must reproduce a direct donor-prefix/recipient-suffix run. An other-branch-pair patch is the negative control.

AF-03 uses AF-02's registered held-out `depth_context`, `delay2`, and `delay4` families. Tensor-only extensions add 1 or 4 waits to delay4 to test 16 and 22 recurrent visits; those extensions receive **no new original-host proof credit**. All holdouts are read strictly after the final train checkpoint is frozen.

The primary preregistered gate compares eligible `status` versus `causal` models at ten carried coordinates on depth_context and delay4. Each seed must gain >=0.05 addressed-patch first-branch correctness while losing <=0.02 clean task accuracy, retaining >=0.95 of the untouched branch, reaching >=0.20 addressed-versus-other selectivity, and adding zero tainted attempts. Eligibility requires train and dev task and both audit slots >=0.95 for both receipt orders. All three eligible seeds must pass for an aggregate positive claim; negative and ineligible seeds are kept.

## Reproduction

Use Python 3.12, PyTorch `2.5.1+cpu`, NumPy, SciPy, and two CPU threads. Download the exact AF-02 training artifact from [run 43556164814](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43556164814). Verify its SHA-256 against the protocol, extract it into `parent`, then run:

```bash
python -m unittest discover -s experiments/activation_funnel_af03/tests -p 'test_*.py' -v
python -m experiments.activation_funnel_af03.experiment train \
  --af02-checkpoints parent/checkpoints --out output/train --seed 17 --width 2
python -m experiments.activation_funnel_af03.experiment evaluate \
  --train output/train --out output/eval --seed 17 --width 2
```

Run every seed `17,29,43` and width `1,2`; the GitHub workflow executes the matrix with isolated output locations, then checks complete receipt/hash coverage in `aggregate.py`. Source, weights, predictions, changed-prefix interventions, scientific gate outcomes, and the original AF-02 provenance are retained. This is an addressed receipt-memory mechanism experiment; latent Lean proof generation requires a separate gym and registration.
