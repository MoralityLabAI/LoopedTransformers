# SEM-ECHO-01 v0.1: development protocol

**Date:** 2026-11-11  
**Theory:** [semantic recovery and recurrence](theory.md)  
**Model family:** fixed-depth LoopedChase checkpoints on branch rmp-v0-phase2  
**Initial observer:** the checkpoint's own output head  
**Primary horizon:** four recurrent transitions  
**Evidence stage:** retrospective development, followed by checks on fresh graphs

This protocol asks whether two complete recurrent states representing the same current pointer predict the same continuation. It begins with saved trajectories and the model's own head. A full-state replay checks that the implementation resumes those trajectories correctly.

The known Q4 and Q5 checkpoint outcomes have already shaped this design. Results from those 24 models carry the label retrospective development. Fresh graph instances support conditional claims about a fixed checkpoint under the declared graph generator. A future prospective test requires fresh training seeds and a decision rule frozen before their results are read.

SEM-ECHO-01 is an independent development identifier. The existing [LOOPSCOPE-01 registration](../../../docs/loopscope01_registration.md) and its Q1 through Q9 numbering remain the record of those experiments.

## 1. Scope of v0

The first runner uses the existing model's output head and fits no probe. It traces the full state at every position after each loop, while the head reads the cursor at the final position.

The compatible task has fixed-depth training, eight loops for the original training objective, and no depth token. During evaluation, the oracle after loop $t$ is $\pi^t(s_0)$. The runner rejects a checkpoint configured with variable-depth tokens: that training setup supervises holding the requested answer after its requested depth, which gives a different oracle.

The default trace has 32 loops. All eligible endpoint pairs contribute to the trajectory metric. Actual patched reruns have a separate global cap of 16 pairs per checkpoint to bound the cost of secondary interventions and replay controls. The endpoint pair and its future have the same complete prompt, including edge order and start token.

The repository supplies two executable entry points:

~~~bash
python -m research_gym.scripts.semantic_echo_controls --out OUTPUT_DIRECTORY
~~~

~~~bash
python -m research_gym.scripts.loopscope_semantic_echo \
  --checkpoint CHECKPOINT_PATH \
  --out OUTPUT_DIRECTORY \
  --items 64 \
  --max-loops 32 \
  --horizon 4 \
  --max-pairs 16 \
  --seed 940117 \
  --device cpu \
  --dtype float32
~~~

The checkpoint option is repeatable; a checkpoint-directory option supports a collection. The runner also has an exclusive smoke mode that constructs an untrained model. Synthetic analytic outputs carry CONTROL_ONLY. Untrained adapter outputs carry UNTRAINED_SMOKE. A neural result requires a real checkpoint identity and a receipt from that run.

These commands describe the intended entry points. The top-level folder README records the implementation and validation status of the current revision.

## 2. What the previous experiments establish

Q5 applied a previously chosen F1/F8 threshold to fresh training runs. Those trained models are now known. Reusing them to choose a semantic-echo rule returns the work to development.

Q7 and Q8 already examine contraction in selected subspaces and drift beyond the training depth. Q9 adds a clock probe and a causal ablation. The semantic-echo experiment asks whether equal current task states carry equal futures under the complete recurrence. A decodable clock can be useful for constructing a donor comparison; its causal role requires an intervention result.

The Q6 proof-transfer endpoint encountered a ceiling under its best-first b4 budget. Later proof-state work will expose a budget curve and exact continuation outcomes in a finite environment. The pointer pilot has its own oracle and avoids treating those proof-transfer observations as interchangeable evidence.

## 3. Trace capture and the semantic oracle

For each generated item, record its permutation $\pi$, original start $s_0$, exact tokens, and the complete post-loop state $h_t$. The input embedding $e(x)$ is constant throughout one item:

$$
h_{t+1}
=\operatorname{core}(\operatorname{inject}(h_t)+e(x)).
$$

The v0 observer is

$$
\widehat s_t
=\arg\max
\operatorname{head}(\operatorname{ln_f}(h_t^{\mathrm{cursor}})).
$$

The symbolic oracle is $s_t=\pi^t(s_0)$. Post-loop visits start at $t=1$; this matches the model's returned states.

Before interpreting a checkpoint, verify that explicit tracing reproduces its normal forward readout. The next attention block receives all recurrent positions, so trace capture includes the complete tensor.

## 4. Endpoint pairs and selection

Let $T$ be the trace length and $H=4$ the primary future horizon. Construct candidate pairs deterministically:

$$
\mathcal P_x=
\{(t,u):1\le t<u\le T-H,\ s_t=s_u\}.
$$

In a permutation graph, $u-t$ is a positive multiple of the start vertex's cycle length. Every multiple within the fixed trace is eligible.

The primary conditional set is

$$
\mathcal G_x=
\{(t,u)\in\mathcal P_x:
\widehat s_t=s_t,\ 
\widehat s_u=s_u\}.
$$

This gate uses the two current endpoint labels. Future labels, later accuracy, phase residual, and any hoped-for rescue effect have no role in selecting pairs.

Report candidate counts and gated counts for every item and checkpoint. An item is one independently sampled graph, start, and prompt draw. Also report the number of items with a nonempty gated set, with the eligible fraction as a coverage measure. An item without a gated pair has a missing conditional statistic and a visible zero count. Unique permutation counts appear separately; independent draws may repeat a permutation.

The default analysis uses every pair in $\mathcal G_x$. The global rerun cap applies only to the secondary intervention batch, including its execution control. A fixed seeded sampler selects the capped pairs using item identities, times, and endpoint gates. The selected identities and seed appear in the receipt.

Record strata for both endpoints within loops 1 through 8, one endpoint within that range and one beyond it, and both endpoints beyond 8. Strata describe where a failure occurs. The primary horizon remains four across them.

## 5. Primary statistic: conditional continuation disagreement

For a gated pair, define

$$
D_{t,u}^{(H)}
=
\frac1H\sum_{k=1}^{H}
\mathbf1[\widehat s_{t+k}\ne\widehat s_{u+k}].
$$

The two oracle futures agree because their current vertices agree under the same permutation. For each item, average $D_{t,u}^{(H)}$ over its gated pairs. The checkpoint statistic is the unweighted mean over items with a defined item statistic. Each eligible independent graph/start/prompt draw receives equal weight, including accidental repeated permutations. Report the pooled pair mean as a descriptive companion.

Also retain the disagreement rate at each future offset and the fraction of pairs with at least one future disagreement.

Under the ideal quotient equation $q_xF_x=\pi q_x$, equal semantic states have equal future semantics at every depth. A positive observed disagreement supplies a finite witness against that property for the tested own-head candidate. It identifies a limitation of this decoder and these states. A later decoder would define a separate candidate abstraction.

## 6. Companion oracle metrics

Two continuations can agree on an incorrect sequence. Report their oracle agreement:

$$
A_{t,u}^{(H)}
=
\frac{1}{2H}
\sum_{k=1}^{H}
\left(
\mathbf1[\widehat s_{t+k}=s_{t+k}]
+
\mathbf1[\widehat s_{u+k}=s_{u+k}]
\right).
$$

Also report each donor side separately, the proportion of pairs whose entire future is oracle-correct, and the checkpoint's accuracy at every trace depth.

The output should retain the four cells of the following comparison:

| Future comparison | Oracle check | Interpretation on the observed pair |
|---|---|---|
| Agrees | Both sides correct | Compatible with the proposed semantic continuation |
| Agrees | A shared error occurs | Agreement conceals an incorrect transition |
| Disagrees | One side correct | Equal current labels conceal a difference relevant to the future |
| Disagrees | Both sides make errors | The own-head continuation differs and both sides need further diagnosis |

The word semantic in this v0 analysis refers to the explicit oracle and the candidate own-head map. Its use remains tied to these measurements.

## 7. Full-state replay is an execution control

For a selected gated pair $(t,u)$, the earlier state $h_t$ is the donor and the later state $h_u$ is the recipient. Copy the complete saved donor state into the recipient slot with the same prompt and run $H$ additional steps.

The resumed outputs should reproduce the cached donor future $h_{t+1},\ldots,h_{t+H}$ within the declared numerical tolerance. The recurrence receives the same dynamic state and the same fixed input. Its source code receives no external loop index in this task.

This replay validates state capture, prompt injection, and resume behavior. Comparing its labels with the recipient's original future yields the donor comparison already present in the trajectory metric. Label a successful replay as an execution check.

Retain raw maximum state deviation and logit deviation, plus equality of discrete labels. Tolerances depend on dtype and execution backend and must appear in the receipt. A failed replay blocks interpretation of further patched runs until the state-accounting error is resolved.

## 8. Koopman phase diagnostics

For an oracle cycle of length $\ell>1$, map its vertices in successor order to roots of unity:

$$
z(v_j)=e^{2\pi i j/\ell},\qquad
\omega=e^{2\pi i/\ell}.
$$

For predictions on that cycle, calculate

$$
r_t=
\frac{|z(\widehat s_{t+1})-\omega z(\widehat s_t)|^2}
{|1-\omega|^2}.
$$

Exact advancement gives zero. Holding a vertex gives one. Report valid transition counts and all off-cycle predictions; singleton cycles have an uninformative phase field.

This residual describes transition structure. A fixed phase offset also gives zero, so the report pairs it with oracle accuracy and the initial semantic anchor.

The finite-horizon identity in the theory note gives an optional certificate on an observed complex phase trace:

$$
|z_T-\omega^Tz_0|
\le |1-\omega|T\sqrt{R_T}.
$$

With a correct initial root, $T\sqrt{R_T}<1/2$ suffices for correct nearest-root decoding at that endpoint. Report the horizon and every premise when using this bound. It concerns the checked trace.

In v0, the phase values are exact roots obtained from discrete labels. On a complete trace with every label on the chosen cycle, each wrong one-step transition contributes at least one to the normalized residual, so $R_T<1/T$ ensures every observed transition follows the cycle relative to its initial label. The RMS endpoint bound is intended for later continuous phase observers as well as algebraic control checks.

## 9. Deterministic controls before checkpoint interpretation

The analytic controls distinguish several mechanisms with exact expected outcomes.

| Control | Expected observation | Purpose |
|---|---|---|
| Exact permutation trajectory | Zero continuation disagreement and phase residual; full oracle agreement | Checks indexing and cycle arithmetic |
| A trajectory that holds one vertex on a nontrivial cycle | Phase residual one on valid transitions | Checks normalization across cycle lengths |
| Correct current endpoints followed by different schedules | Positive conditional continuation disagreement | Checks the semantic-sufficiency failure case |
| A fixed wrong phase offset | Low phase residual with oracle errors | Checks that accuracy accompanies phase consistency |
| Nuisance-only contraction with terminal Gaussian noise | Semantic inverse MSE $\sigma^2$ | Separates retained semantics from raw contraction |
| Semantic-coordinate contraction with the same noise | Semantic inverse MSE $\sigma^2\alpha^{-2k}$ | Detects sensitivity of the inverse to finite noise |

The controls use explicit source parameters and emit CONTROL_ONLY receipts. Any Monte Carlo estimate also records its sample size and generator seed, alongside the analytic target.

The finite-horizon phase bound has an exact algebraic control. Its residual sum must equal the endpoint defect before the inequality is tested.

## 10. Secondary interventions in capped reruns

The default capped batch includes the full donor replay, cursor and edge transfers, a LayerNorm shift, and a matched random perturbation for each of the cursor, edge, and full-state scopes. Partial state interventions carry secondary exploratory result labels.

**Cursor transfer.** Replace the recipient's cursor state by the donor cursor while retaining the recipient's edge positions. Compare future oracle agreement with the unpatched recipient.

**Edge transfer.** Replace the edge positions while retaining the recipient cursor. The same donor-pair rule applies. These two patches test where information beyond the current pointer is carried.

**Matched random perturbations.** Use the same affected positions and match the perturbation norm. The current runner draws one direction per scope with a recorded seed. Freeze any later increase in the number of controls before reading effects. Differences from this control help assess whether a result depends on the transferred direction.

**LayerNorm shift control.** Add a constant across all feature coordinates within each selected position before the next input LayerNorm. In exact arithmetic, LayerNorm removes this shift. Its observed numerical effect provides a control for raw activation distance that leaves the normalized input invariant.

A hybrid patch can leave the observed trajectory distribution. Full donor states establish a useful reference, and matched random changes quantify the effect of perturbation size. A later causal claim must specify the high-level intervention that the patch represents.

## 11. A proof-state continuation experiment

The next task family uses finite Horn logic. A state records available facts, implications, and a goal; one primitive action applies an implication whose antecedent is present. Successful actions carry a symbolic proof witness.

Exact shortest derivations and finite continuation signatures can be computed for small instances. Measure success at several fixed expansion budgets and report the minimum successful budget. The action grammar counts primitive inference applications so the budget has a stable meaning.

Construct paired states by removing a fact needed on every short route to the goal or a fact irrelevant to those routes. Match edit size. Compare recovered representations with the same original proof state or with the expected symbolic successor.

A Doob bridge, computed from exact finite-horizon success probabilities, can produce paired traces conditioned on a checked proof. Its conditioning information is an explicit teacher privilege. Held-out evaluation must charge the declared search budget and account for every oracle call.

The pointer quotient supplies an exact cycle model. Proof states have branching and action-dependent continuations, so their primary signature is $\chi_B$ from the theory note.

## 12. Inference and a later prospective study

Per-item aggregation handles the many overlapping endpoint pairs within a trajectory. For the independent item generator, a bootstrap for a fixed checkpoint resamples complete items and preserves all their pairs. Repeated permutations produced by independent draws remain valid items. If a later experiment deliberately derives several starts or contexts from one sampled graph, its bootstrap clusters those items by graph. Trained seeds are the units for claims about the training procedure. Multiple snapshots from one training run remain dependent observations.

The known 24 models may be used to develop candidate predictors, examine failure strata, and identify useful controls. Every such choice should be recorded. A rule chosen from those extrapolation labels receives its unseen test on a fresh set of trained models.

Before a prospective run, freeze the new training seeds, model configurations, graph generator, and checkpoint-selection rule. Specify the primary metric and its aggregation, the endpoint gate, and how a model with too few eligible items will be reported. Fix any decision threshold and planned confidence procedure before training or inspecting the new outcomes.

No universal success threshold is introduced by v0.1. Its output provides conditional disagreement rates, coverage, and oracle checks from which a later proposal can be made.

## 13. Receipt requirements

Every result must identify the code revision, model configuration, and checkpoint hash. Record the command, graph seed, item count, trace length, future horizon, dtype, backend, and numerical tolerances. Include the fixed-depth compatibility result.

The receipt should retain candidate and gated pair counts by item; item-level primary scores; pooled pair scores; endpoint accuracy; future oracle agreement; and depth strata. Report unique permutation counts separately. Phase coverage includes cycle lengths and invalid transitions.

Replay records name the selected pairs, state and logit deviations, and the global cap. Secondary patches record their masks, donor identities, random seeds, and norm matching.

Mark the evidence stage explicitly:

| Label | Meaning |
|---|---|
| CONTROL_ONLY | Analytic or synthetic control |
| UNTRAINED_SMOKE | An untrained model exercises trace and runner code |
| Retrospective development | A previously examined checkpoint contributes to method development |
| Fresh-graph conditional check | New graph draws test a fixed model and a frozen analysis |
| Prospective fresh-seed study | A frozen model-level rule is evaluated on newly trained seeds |

Saved artifacts describe what ran. The mathematical all-depth result requires an invariant domain and a verified commuting transition, or certified invariant regions with the bounds stated in the theory note. Sampled trajectories estimate behavior on their declared distribution.
