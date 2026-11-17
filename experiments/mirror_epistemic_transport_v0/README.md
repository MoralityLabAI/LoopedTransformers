# MIRROR-01: Counterfactual Belief Transport for Looped-Transformer Dialogue

**Status:** runnable synthetic research gym with optional PyTorch. It does not interrogate a live LLM and makes no claim of superiority over a frontier model.

MIRROR learns a latent model of another system's **expressed commitments**, transports that model across equivalent question contexts, and picks follow-up questions that can produce an auditable contradiction witness. The controller is a shared-weight looped Transformer rather than a TRM debate generator.

## Model

- **Belief field:** one latent token for each proposition-family/context pair, including observed answer and missingness.
- **Low-rank counterfactual transport:** T_c(h) = V((Uh) * g_c), with a learned gate for each context.
- **Shared-weight recurrence:** a single Transformer encoder block applied K times; K=2 in the CPU-size reference.
- **Predictive heads:** probabilities for as-yet-unasked answers and estimated family inconsistency.
- **Epistemic action selection:** candidates are ranked by expected disagreement with observed commitments, uncertainty, and estimated risk.
- **Independent witness checker:** only two active, explicit conflicting answers under a *declared* equivalence relation count. Revisions replace earlier commitments.

The reference neural core has **36,866 parameters** (64-wide shared block and low-rank transport); 31M parameters is a proposed later scale, not a claim about this experiment.

For family f and context c:

    h[f,c]^0 = E(features[f]) + E(f) + E(c) + T_c(E(features[f])) + E(observation[f,c])
    h^(k+1) = SharedTransformer(h^k)
    p[f,c] = sigmoid(W h[f,c]^K)

The loss is masked binary cross-entropy for unasked answers plus a family-risk auxiliary BCE term weighted 0.25. Coefficients are starting hyperparameters.

## Reproduce

From repository root (Python 3.10+, `pip install torch pytest`):

```bash
python -m pytest experiments/mirror_epistemic_transport_v0/test_mirror.py -q
python -m experiments.mirror_epistemic_transport_v0.run --mode smoke
python -m experiments.mirror_epistemic_transport_v0.run --mode train --steps 300 --episodes 200 --seed 7 --checkpoint output/mirror01.pt --out output/mirror01.json
python -m experiments.mirror_epistemic_transport_v0.run --mode eval --checkpoint output/mirror01.pt --episodes 200 --seed 7
```

Smoke trains eight small batches on CPU; full training selects CUDA when available. Evaluation uses disjoint deterministic RNG seeds: training `seed`, evaluation `seed+100000`.

## Target model and evidence contract

Synthetic worlds contain six proposition families with public binary features and a hidden persona. Personas are stable, role-asymmetric, consequence-asymmetric, scope-only, or mixed. Question contexts are neutral, role swap, consequences, and explicitly qualified scope. The first three are **defined by the gym** to preserve the proposition's meaning; the fourth is a permitted scope exception.

Hidden personas and base commitments are used to generate target responses and gold labels. The policy's neural inputs consist only of public family features and replies already observed, although the synthetic dialogue object itself still contains the target oracle; a future live-adapter interface must isolate this boundary.

The verifier refuses to issue a contradiction witness based on probability estimates alone. Explicit corrections replace earlier active answers.

## Baselines and caveats

The comparator policies are random, deterministic paired-question search, the learned MIRROR controller, and a privileged omniscient oracle that reads the hidden target. The oracle is a diagnostic upper bound.

Report discovery recall on inconsistent targets, false positives, witness precision, and mean target queries, at equal query budgets. The synthetic world shares its generative grammar between training and evaluation. A real-model transfer result would require new subject matter, paraphrase and scope annotation, human auditing, and held-out target architectures.

The interview transcript motivating this experiment provides qualitative design inspiration; the toy data and numerical results are synthetic.

## Acceptance gates

1. All unit tests pass, including evidence correction and scope exceptions.
2. The trained and paired baselines use identical held-out worlds and query budgets.
3. Every witness can be reconstructed from recorded questions and binary answers.
4. Architecture benefits are established with repeated seeds and matched parameter counts.
5. Test on unseen transforms and target architectures before making generalization claims.

See [NEXT_EXPERIMENT.md](NEXT_EXPERIMENT.md) for the MIRROR-02 registered-protocol proposal.
