# MIRROR-02: Compositional Epistemic Transport

Status: **working synthetic experiment, local-only measurements** (2026-11-05). This extends MIRROR-01 in this directory. No live language-model target was interrogated.

## What was built

The target answers propositions in families f under transformations t ∈ (Z₂)³. Each hidden target has a persona mask v ∈ (Z₂)³ and a family-specific base bit b_f. The three public feature bits x_f determine where each transformation interacts with v:

    answer(f, t) = b_f XOR parity(v AND x_f AND t).

Questions with a scope-change flag have a separate hidden answer modifier and cannot produce a contradiction witness with the unscoped propositions. A witness requires two recorded answers, equal declared semantic scope, and opposite commitments. Explicit corrections replace the active earlier answer.

Training uses six families, the identity and three singleton transformations, one scope variant, and personas {0, 1, 2, 4}. Held-out evaluation uses nine families, composite transformations {110, 101, 011, 111}, a scope variant, and personas {0, 3, 5, 6, 7}. Thus neither the composite transforms nor the nonzero multi-bit personas are sampled during training. It is still a *synthetic generative-family* split.

The policy receives a frozen View of public features, transforms and previously observed answers. The hidden persona and base bits remain inside the evaluator.

## Architecture

A 48-wide looped Transformer updates proposition/context tokens through two shared-weight passes. There is local attention across contexts in each family, plus pooled evidence across families to maintain a latent target-persona estimate. The decoder predicts unseen answers, family risk and next-question action logits.

The primary transport operator uses learned orthonormal axes u_i and commuting Householder reflections:

    T_t(h) = h - 2 SUM_i t_i <h, u_i> u_i

so that for binary transformation vectors a and b:

    T_a(T_b(h)) = T_(a XOR b)(h).

This is an exact algebraic constraint on *latent transport*, not a statement that the decoded replies must be consistent.

A "free" comparator uses an unconstrained low-rank gated transform. A "none" comparator leaves the base embedding unchanged. All share a learned context input, the same recurrent block and action heads. Their parameter counts differ; this experiment is an initial ablation, not a matched-parameter architecture study.

## Why the actor matters

Prediction-only training gave poor short-horizon question selection despite exact operator composition. The successor adds supervised next-action logits distilled from an analytic Bayesian question selector. That teacher enumerates the eight possible persona masks and conditions on acquired replies. The training objective is masked answer BCE + 0.2 family-risk BCE + 0.8 action cross-entropy in the reported runs. The last term is necessary for meaningful policy performance in this toy setting.

## Reproduce

From the repository root, with Python 3.10+, torch and pytest:

    OMP_NUM_THREADS=2 python -m pytest experiments/mirror_epistemic_transport_v0/test_v2_compositional.py -q
    python -m experiments.mirror_epistemic_transport_v0.v2_compositional --transport reflection --steps 120 --imitation-weight .8 --episodes 100 --seed 7 --budgets 4 --output output/mirror02_reflect_s7.json
    python -m experiments.mirror_epistemic_transport_v0.v2_compositional --transport free --steps 120 --imitation-weight .8 --episodes 100 --seed 11 --budgets 4 --output output/mirror02_free_s11.json
    python -m experiments.mirror_epistemic_transport_v0.v2_compositional --transport none --steps 120 --imitation-weight .8 --episodes 100 --seed 19 --budgets 4 --output output/mirror02_none_s19.json

Local verification: seven tests pass. Exact reflection composition MSE across test cases is below 1e-10; sample measurement is around 1e-15.

## Initial held-out results

Each seed contains 100 synthetic target worlds, and each policy receives four questions per world. The same worlds are used for all variants within a seed. A result is counted when the verifier has two active, incompatible commitments within the *declared* scope. Zero false positives were observed under that synthetic verifier.

| Variant | Seed 7 | Seed 11 | Seed 19 | Found / available |
|---|---:|---:|---:|---:|
| Two-pass, reflection transport | 77/77 | 74/74 | 78/78 | **229/229** |
| Two-pass, free transport | 77/77 | 70/74 | 78/78 | 225/229 |
| Two-pass, no transport | 77/77 | 70/74 | 78/78 | 225/229 |
| Scripted paired questions | 77/77 | 74/74 | 78/78 | **229/229** |
| Bayesian teacher | 77/77 | 74/74 | 78/78 | **229/229** |
| One-pass reflection (seeds 7, 11 only) | 77/77 | 63/74 | n/a | 140/151 |

The two-pass reflection policy approaches the analytic teacher's query efficiency. On seeds 7, 11, 19 its mean queries across *all* worlds were 2.72, 2.75 and 2.64; the teacher's were 2.69, 2.76 and 2.61.

The out-of-distribution Brier scores on held-out composite answers were:

| Variant | Seed 7 | Seed 11 | Seed 19 |
|---|---:|---:|---:|
| Reflection | 0.2372 | 0.2371 | 0.2426 |
| Free | 0.2391 | 0.2373 | 0.2442 |
| None | 0.2317 | 0.2363 | 0.2465 |

The reflection structure achieves exact composition by construction. The differences in question success across these seeds are small. The Brier comparison gives no consistent advantage to the reflection operator. An identity transport also has zero algebraic composition error, illustrating why the algebraic certificate alone cannot establish utility.

## Limits and next research gate

The teacher knows the synthetic response grammar, giving the student a highly informative training signal. The paired-question baseline already succeeds on all positives at a budget of four. The model's gain here comes predominantly from explicit action supervision; the distinctive transport geometry awaits validation.

MIRROR-03 should incorporate uncertainty over whether transformations preserve meaning, independently reviewed semantic-equivalence labels, ambiguous and self-revising LLM replies, and a language-decoding interface. Challenge tasks must make paired questioning meaningfully less effective. Compare parameter-matched tied, untied, and GRU controllers across enough held-out targets to estimate uncertainty. Train with a question-value objective rather than only imitating the analytic teacher; use locked target-model prompts and raw response logs for any live trials. Keep the pragmatic frame/irony mechanism as an explicit separate experiment.

No model has learned Bassem Youssef's full conversational skill at this milestone. The code establishes an inspectable route toward that experiment.
