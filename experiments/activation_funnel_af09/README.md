# AF-09: true-class causal margin in compact recurrent Horn proof state

AF-09 is a new, preregistered experiment stacked on the failed AF-08 study.
It reuses both **independently qualified** AF-06 seed-17/29 width-2 warm-up
checkpoints without changing their tensors or training them again. Its purpose
is to test whether the exact carried-code margin certificate in
`docs/research/AF08_Q_MARGIN_THEOREM.md` can serve as a stable, interpretable
training signal for a small native TypedLoop.

## Research question

AF-07 showed near-zero protected-code drift despite chance-level subgoal truth.
AF-08's class-prototype and donor-distance margins were already inactive on the
qualified source states and did not improve deep causal locality. AF-09 uses the
**true Horn class**, the actual Q code *after every recurrent call*, and the
*current trained* decoder/readout sensitivity. The penalty is active for
insufficient semantic margins even if the code is stable.

Let z be the original protected Q code, z' its code after the selected other Q
was patched, and y its exact recipient Horn truth. The original decoder and
subgoal classifier are affine:

    m_y(z) = (W_R[y] - W_R[1-y]) · (W_D z + b_D) + b_R[y]-b_R[1-y]
    K_y    = ||(W_R[y]-W_R[1-y]) W_D||_2
    L_y    = m_y(z) - K_y ||z'-z||_2

By Cauchy-Schwarz, m_y(z') >= L_y for every actual intervened recurrent state.
A strictly positive L_y certifies protected-subgoal classification **for that
known correct class**, even when the recurrent TypedLoop is nonlinear.

The `semantic_margin` arm adds, on addressed rows and all three downstream
training steps:

    + 0.30 * mean softplus(0.5 - L_y)
    + 0.15 * mean softplus(0.5 - m_y(z'))

The softplus supplies a gradient beyond AF-08's already-satisfied hinge margins.
Both calculations differentiate into the original native TypedLoop, encoder,
decoder and status readout. An exact Horn counterfactual supplies the labels.

## Registered controls and data

See [PROTOCOL.json](PROTOCOL.json). Seeds: 17 and 29. Each arm receives 160
additional AdamW updates after the exact AF-06 480-step shared warm-up. The
three arms are `status`, `locality`, and `semantic_margin`. The first two use
AF-08's originally registered training losses; the third adds only the above
signed margin objective to the **same** addressed patch exposure and locality
terms. All three arms make the same source-cell forward calls, although the
margin arm performs additional arithmetic and backward FLOPs.

The original finite-Horn oracle and corpus have 672 training, 224 development,
and 496 heldout cases (16 novel compositions, 224 layouts, 128 depth-4 and
128 depth-5). Only train is used to fit. Dev qualifies. All heldout scores are
computed separately after fixed-step export and retain every negative result.

The primary gate requires both source and compared arms to meet 90% train/dev
goal and exact joint-Q accuracy. At **each** heldout depth 4 and 5, the margin
arm must improve protected-Q accuracy and positive-certificate coverage by
at least 0.05 over locality, reach 0.90 absolute protected-Q accuracy and 0.90
irrelevant-patch goal retention, preserve clean-goal performance within 0.03,
and add no false-positive goals. Any eligible seed failure makes the aggregate
fail. Unqualified seeds are inconclusive.

## Independent evidence and limits

`evaluate` saves 2,976 clean predictions, 5,952 counterfactuals, 9,024
stagewise records and 9,024 signed margin/certificate records across both
seeds. Every certificate retains the clean and intervened **two-float** actual
code, true-label signed logit margins, exact trained readout sensitivity,
physical code distance and computed lower bound. The independent NumPy verifier
recalculates the bound, checks each stored predicate and logits against the
original model and recomputes each Horn label and aggregate. A false positive
certificate is a fatal engineering error, irrespective of scientific results.

Lean 4.19.0 recompiles the 4,176 *closed synthetic Boolean-Horn oracle*
certificates. These prove facts about the synthetic source truth, not the
neural model's proof discovery or correctness on unseen Lean mathematics.

## Reproduce

Python 3.12, CPU Torch 2.5.1, NumPy and SciPy as in parent CI. From repo root:

```bash
AF09_TEST_SOURCE_ZIP=<exact-AF06-seed17-artifact.zip> python -m unittest discover -s experiments/activation_funnel_af09/tests -p 'test_*.py' -v
python -m experiments.activation_funnel_af09.experiment source --seed 17 --source-zip <seed17.zip> --out /tmp/af09-source.json
python -m experiments.activation_funnel_af09.experiment train --seed 17 --source-zip <seed17.zip> --out /tmp/af09-train-s17
python -m experiments.activation_funnel_af09.experiment evaluate --seed 17 --training /tmp/af09-train-s17 --out /tmp/af09-eval-s17
python -m experiments.activation_funnel_af09.verify --seed 17 --training /tmp/af09-train-s17 --evaluation /tmp/af09-eval-s17 --out /tmp/af09-eval-s17/independent_audit.json
```

Full seed/width/checkpoint roster and source hashes are frozen before scoring.
AF-01 through AF-08 remain unchanged. Keep AF-09 in its own draft PR.
