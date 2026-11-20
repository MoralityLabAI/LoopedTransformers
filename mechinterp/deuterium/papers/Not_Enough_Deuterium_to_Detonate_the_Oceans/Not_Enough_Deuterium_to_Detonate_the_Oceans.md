---
title: 'Not Enough Deuterium to Detonate the Oceans'
subtitle: 'Mechanistic Interpretability-Steered Self-Improvement in Tiny Looped Transformers'
author: 'Patrick Dugan | MoralityLabAI'
date: '20 November 2026'
lang: en-US
bibliography: references.bib
abstract: |
  Can a small recurrent model use an interpretable account of its own computation to improve its weights, and eventually become better at making further improvements? We investigate this question through parameter-component transfer, controlled weight edits, and studies of recurrent state steering. An exploratory six-seed search recorded 1,305,171 evaluation rows, including 131,376 candidate head-component evaluations, and accepted 16. Reanalysis shows that every accepted candidate tied its strongest eligible random comparator on task metrics; its advantage came from a smaller change in output logits. In controlled experiments, directions learned by adVersarial Parameter Decomposition (VPD) did not establish a reliable advantage over equally searched random directions. The mean random-minus-VPD loss difference was +0.000890 nats in a 1,763-parameter recurrent decision model and −0.000026 nats in a 44,001-parameter four-pass model, with both model-level intervals spanning zero. A diagnostic of weight sharing found that the fitted VPD implementation reused a final-pass importance mask during earlier recurrent passes. Companion Behavioral Parameter Decomposition (BPD) experiments exposed poor persistence of a fixed state-steering subspace across recurrence, while online sparse-operator routing improved task fidelity in a small engineering cohort. These observations identify concrete obstacles between interpretable components and beneficial autonomous edits. We develop a falsifiable experimental program for closing that gap and distinguish its proposed mechanisms from the results demonstrated so far.
keywords:
  - mechanistic interpretability
  - tiny recursive models
  - parameter editing
  - recurrent computation
  - self-improvement
---

# Introduction

A Tiny Recursive Model (TRM) can return to the same network repeatedly, gaining computational depth while keeping its parameter count small [@jolicoeur2025]. Hierarchical Reasoning Models (HRMs) extend the idea with recurrent computations at different timescales [@wang2025]. Work on looped transformers suggests that each return to a shared layer can perform a different part of a calculation [@saunshi2025; @kaissis2026]. Once a small network can reason through repeated passes, a tempting question follows: could it learn enough about those computations to change its own weights for the better?

A model that solves a puzzle has demonstrated useful computation. A model that improves the computation needed for its next puzzle would have crossed a far more demanding threshold. It would have to identify an internal failure, choose an edit, and test whether the improvement survives subsequent use of the same weights. Our title borrows its image from the history of nuclear weapons research. A 1946 Los Alamos calculation found that radiative losses would prevent an atomic explosion from igniting the atmosphere [@konopinski1946]. A later study examined the possibility of a self-propagating detonation in the oceans [@weaver1979]. The 1975 Livermore analysis found that seawater's actual deuterium concentration reduced an already minute simulated burn by a factor of about 20,000 [@batzel1975]. The analogy here is about reaching a threshold at which one improvement can help produce the next.

Mechanistic interpretability offers a way to search for such edits. Rather than treating a neural network as an undifferentiated collection of weights, parameter-decomposition methods try to isolate simpler contributions to its behavior. This line of work runs through Attribution-based Parameter Decomposition (APD), Stochastic Parameter Decomposition (SPD), and adVersarial Parameter Decomposition (VPD) [@braun2025; @bushnaq2025; @bushnaq2026]. VPD learns components that reconstruct a model while making their importance sparse and testing them under adversarial partial ablation. Goodfire demonstrated it on approximately 28 million non-embedding parameters of a four-layer, 67-million-parameter language model [@bushnaq2026]. Other model-editing methods have localized and altered factual associations in transformers, and low-rank adaptation gives a practical way to constrain updates [@meng2022; @meng2023; @hu2022]. Those precedents motivate the experiment here: can components that explain a network also suggest changes that improve its behavior through repeated computation?

We study that question in two controlled models, following an exploratory search among small decision networks. The parameter-decomposition code comes from Goodfire's optimizer and component model **as of upstream commit [`1f531b3`](https://github.com/goodfire-ai/param-decomp/tree/1f531b3)**, dated 6 May 2026. Both files have the same Git blob hashes in the [`MoralityLabAI/param-decomp` fork at `2df0092`](https://github.com/MoralityLabAI/param-decomp/tree/2df0092d5be4d8e4748b3c66ef8325dedef17549), whose additional code provides our experiment wrappers and toy model. Commit `2df0092` belongs to the MoralityLabAI fork's `rust-accel-scoring` branch. Our findings concern this specific fitting procedure and the interventions tested on small recurrent models. After examining those weight edits, we turn to Behavioral Parameter Decomposition (BPD), whose Sudoku experiments investigate how causal directions and sparse component choices behave over multiple passes.

The models answer different parts of the question. Our 1,763-parameter decision gate and 44,001-parameter four-pass network reuse weights but contain **no transformer attention stack**. The BPD pilots use TRM-style recurrent Sudoku networks. The looped-transformer system in the subtitle is the proposed destination for this research, rather than a model on which we claim to have demonstrated self-improvement. We call its prospective small editing controller an *organelle*: a trainable system that observes recurrent states, proposes localized changes, and learns from their consequences.

# Interpretable edits in a recurrent network

Consider a model that repeatedly updates a hidden state using a shared parameter set:

$$
 z_{t+1}=F_{\theta}(z_t,x),\qquad t=0,\ldots,T-1.
$$

Here $x$ is the input and $z_t$ is the state after the $t$-th application. A physical change to $\theta$ acts whenever the same weights are called. For example, a rank-one edit to a matrix $W$ is

$$
 W' = W + \epsilon uv^{\mathsf T}.
$$

An edit to shared weights acts every time the network returns to them. Its final effect is therefore shaped by the entire rollout, including states the edit itself has changed.

Parameter decomposition represents a weight matrix through components that can be selected or suppressed. VPD adds an importance mechanism that depends on its input [@bushnaq2026]. In a feed-forward network, a module is encountered at one point in the calculation. In a weight-tied network, the same module encounters different hidden states on successive passes. This makes the timing of a component choice part of the mechanism: should the choice remain fixed, or should the model reconsider it at every pass?

To compare edits, we define the primary effect as

$$
 \Delta_{\mathrm{VPD}} =
 L(\text{selected random edit})-L(\text{selected VPD edit}),
$$

Here $L$ is held-out negative log-likelihood. **A positive difference favors VPD.** Both methods search their candidate banks on calibration data before the selected edit reaches the held-out evaluation. This guards against an easy source of false progress: a bank of random directions can yield a promising candidate when enough possibilities are tried.

We assess uncertainty across independently trained models, which we call *roots*. Each root can be tested on many inputs, but those inputs do not constitute fresh model replications. The controlled studies therefore report Student intervals over three model-level effects (two degrees of freedom). The intervals are necessarily broad, especially when teachers were previously examined. Their positive-result criteria and seeds were frozen in a local Git commit before scoring, and the relocated runner authenticates the source and configuration [@dugan2026]. This was a local source freeze rather than an independently timestamped preregistration. The [protocol](https://github.com/MoralityLabAI/LoopedTransformers/blob/39db0ab08a2c7c69b3b0bb04e74b13c7b59bdc7a/mechinterp/deuterium/experiments/deuterium/PROTOCOL.md) and [registration record](https://github.com/MoralityLabAI/LoopedTransformers/blob/39db0ab08a2c7c69b3b0bb04e74b13c7b59bdc7a/mechinterp/deuterium/experiments/deuterium/registration.json) preserve the decision rules.

# What the first component searches found

The first experiments took place in an arena of small decision modules that learned actions such as routing, verification, and abstention. We transferred output-head directions from one module to another, varying their strength and comparing the result with random directions. These were singular-vector (`head_svd`) and head-row (`head_row`) interventions. **The exploratory search did not train VPD's adversarially learned parameter components.** It asked whether a simple direction extracted from one module could usefully influence another.

The six-seed replication recorded **1,305,171 evaluation rows**. Within that total were **131,376** candidate head-component rows marked `vdp_or_class_row`, of which **110,731** passed the stability filter. Sixteen rows satisfied the final acceptance rule: nine `head_svd` and seven `head_row`. The archived preview displayed only twelve. Because the same source-target-component combination could be tested at multiple strengths, the sixteen accepted rows represented fourteen distinct combinations [@archive2026; @dugan2026].

Yet those sixteen apparent successes had a decisive limitation. **Every accepted edit tied the best eligible random comparator on the measured task outcomes.** The acceptance rule broke those ties by preferring the smaller movement in output logits. It thus rewarded a gentler perturbation rather than a gain in task performance. The historical `vdp_or_class_row` label offers no evidence that the search contained fitted VPD components.

The archive also contains attempts to improve behavior using explicit policies over cached scores. A Wordle rule changed 24 correct decisions to 42 among 72 distinct recorded observations, although the original puzzle lineage was incomplete. For **ARC Challenge**, the natural-language multiple-choice benchmark [@clark2018], grouping repeated trajectory identifiers left only **5, 4, 2 and 6 genuinely unseen rows** across four held-out cards. The selected rule changed correctness on **none of those unseen rows**; its four net rescues among 62 cached examples came from previously represented observations. ARC Challenge here is distinct from **ARC-AGI**, the grid-reasoning benchmark often discussed alongside TRMs. These cached-policy results illuminate the problem of evaluation overlap but offer little evidence for lasting weight improvement [@dugan2026].

A multi-family search then attempted to retain successive beneficial changes. None of the recorded sequences established sustained improvement. Its outer-loop estimates sometimes substituted simulated effects for measured ones, and its eligibility rules treated some learned-component proposals differently from their controls. That experience motivated a narrower experiment in which each candidate would be a physical weight edit, the perturbation size would be fixed, and the random comparator would receive an equal search budget [@dugan2026].

# Controlled parameter steering

## A small recurrent decision model

The first controlled model was a **1,763-parameter recurrent decision gate** trained to choose among commit, veto, and repair on a synthetic distribution. Because this test edited its output head after the recurrent transition, it tested candidate selection without introducing the additional complications of changing a weight on every loop. Its VPD fit used **200 warmup and 300 main updates**, with **two projected-gradient adversarial mask steps**. Across three independently trained roots, the VPD arm searched 32 learned rank-one directions in both signs, while the random arm searched 32 Gaussian outer products in both signs. Each edit had a Frobenius norm equal to **1% of the original output-head weight norm**. A supervised task-gradient direction provided another reference, with the extra information supplied by its labels kept explicit [@dugan2026].

VPD achieved a small positive random-minus-VPD loss difference in each of the three roots. The initial local mean was **+0.00089040 nats**, and a later GitHub Actions replay of the same exposed seeds gave **+0.00088999 nats**. Across roots, the replay's 95% interval was **[−0.000160, +0.001940]**, too wide to establish a positive advantage. The source-frozen success rule required clean accuracy of at least 90%, adequate class representation, a positive difference for all three roots, a positive lower interval bound, and no more than two percentage points of collateral accuracy loss per root. The interval failed that rule. The supervised rank-one task-gradient control also achieved lower target loss than VPD in all three roots, although that baseline had access to task labels [@dugan2026].

The uncertainty comes primarily from fitting only three independent systems. Thousands of predictions measure those systems well; they say much less about the behavior of a newly fitted decomposition. What this experiment establishes is a reproducible, physically applied weight-edit comparison with a matched search. A general VPD steering advantage remains unproven.

## Editing a weight reused across four passes

We next edited a weight that the model reused at every recurrent pass. This network had **44,001 parameters** and applied the same reasoning block four times. It learned a transparent cyclic-copy task: for an 81-position digit string, the required output at each position was the digit from the preceding input position, wrapping at the boundary (`x.roll(1)`). Goal loss was measured at the eight prespecified indices **0, 10, 20, 30, 40, 50, 60 and 70**. Because a simple deterministic rule solves this task, it isolates how a shared-weight edit propagates through the network.

The target was a **48-by-128 channel-mixing matrix**. VPD fitted sixteen rank-one components over **32 faithfulness warmup and 64 main optimizer updates**, using batches of eight strings and two adversarial mask steps. The matched random bank also contained sixteen rank-one outer products. With both signs evaluated, each arm searched **32 signed candidates** on the same 256-string calibration set. Every edit changed the shared matrix at a dose of **1% of its Frobenius norm**, and evaluation used another 256 strings with the target positions fixed in advance [@dugan2026].

All three unedited teachers were already strong, scoring between **98.4% and 98.8%** digit accuracy. The average random-minus-VPD goal loss was **−0.0000261 nats**, with a descriptive across-teacher 95% interval of **[−0.0002874, +0.0002352]**. Teacher **211** favored random more clearly: its within-teacher paired-string interval was **[−0.000268, −0.000022] nats**. The source-frozen criterion required a positive effect in every teacher and an across-teacher interval entirely above zero. This test failed; the gradient control, which had access to labels, also reached lower goal loss than VPD for every teacher.

| Model | Independent model roots | Mean random-minus-VPD NLL (nats) | 95% across-root interval |
|:--|:--:|--:|:--|
| Recurrent gate (1,763 parameters) | 3 | +0.00088999 | [−0.00016005, +0.00194002] |
| Four-pass network (44,001 parameters) | 3 previously examined teachers | −0.00002608 | [−0.00028738, +0.00023521] |

: Matched searches of learned and random weight edits. Positive differences favor VPD; both source-frozen superiority tests failed. The second interval is descriptive because the teachers were previously examined.

The digit task offered little remaining accuracy to recover, which limits the study's power to detect useful interventions. Nevertheless, the comparison measured a real edit to a weight used on every pass, with an equal random search and a collateral-performance check. Its result is evidence about this particular fitting recipe and task, leaving the behavior of harder recurrent models to further experiments.

# A weight-tied implementation has a temporal meaning

The shared-weight experiment also exposed a subtle implementation issue. In Goodfire's component model **as of upstream commit [`1f531b3`](https://github.com/goodfire-ai/param-decomp/blob/1f531b3/param_decomp/models/component_model.py)**, preserved unchanged in the [`MoralityLabAI` fork at `2df0092`](https://github.com/MoralityLabAI/param-decomp/blob/2df0092d5be4d8e4748b3c66ef8325dedef17549/param_decomp/models/component_model.py), the activation cache is indexed by module path. Each call to the shared module overwrote the preceding entry. After four passes, that entry contained only the **final clean-pass input**. The importance network calculated a mask from it and reused the mask at all four applications [@dugan2026].

This observation describes what happened when that implementation encountered a weight-tied four-pass module. On **eight previously seen training strings**, separately evaluating the fitted importance network on each recorded pass produced first-to-fourth mask differences averaging **0.27 to 0.30** across the three teachers. The differences diminished toward the final pass, where the comparison is zero by construction. Goodfire's original 67-million-parameter application had a different execution setting, so this diagnosis applies specifically to the reused module studied here.

A mask learned from a state near the end of the calculation may be poorly suited to the state encountered at its beginning. That possibility offers a concrete explanation to test, although the experiment has yet to isolate its contribution to the negative weight-edit result.

We therefore built a pass-indexed adapter that records each invocation's input and computes a corresponding mask while keeping the physical weight factors shared. Pointwise identity tests passed. Over the full **32-warmup/64-main-update** fitting schedule, however, a comparison on **one excluded untrained teacher** accumulated differences even when the adapter was forced to reuse the final-pass mask. Its maximum parameter deviation reached **$3.59	imes10^{-4}$**, beyond the **$2	imes10^{-5}$** tolerance fixed for the full trajectory. The engineering parity gate remained closed, and the twelve reserved teachers were left untouched [@dugan2026].

The cache finding identifies a recurrent timing problem; the adapter test identifies an obstacle to measuring its effect cleanly. The companion BPD experiments examine the same question from the perspective of hidden states and sparse operators.

# What the BPD depth-steering experiments add

**Behavioral Parameter Decomposition (BPD)** studies how an interpretable basis should behave when the network revisits the same weights. Its depth-steering gym uses recurrent Sudoku models, testing interventions in hidden states alongside sparse decompositions of shared operators [@bpd2026]. Unlike the selected-target loss measured in the VPD experiments, these studies often ask whether a model produces an entire correct, constraint-satisfying puzzle grid.

## Static directions lose their grip on recurrence

In **DT04**, the first depth-steering study, researchers selected an eight-dimensional intervention subspace from the full hidden state while protecting the model's immediate output. A *task Gramian* estimates which perturbations should influence later task behavior. Each proposed direction was compared against *signed-Jacobian shams*, controls matched both for perturbation size and for the immediate signed derivative of the task output. This asks whether the putative mechanism predicts anything beyond a matched local effect. The study attempted **59 model roots**; **56** passed its root-record audit, and **16** met every admission gate. A retrospective analysis of all 59 gave a mean depth-eight advantage of **0.000265 exact-board fraction** (**0.027 percentage points**, across-root SD **0.0118**) over eight signed shams per root. Each root contributed sixteen **4×4 Sudoku (Sudoku4)** board cases. Averaging across shams makes this a fractional board-equivalent contrast, rather than a count of independently rescued puzzles. Among **19 retrospectively selected competent roots**, the contrast was zero. That subset included the three models flagged by the 36-pass float32 gauge audit (`2604008`, `2604026`, `2604050`), so its findings carry the corresponding eligibility limitation [@bpd2026].

The selected subspace also changed its orientation as the network ran. Its one-step closure defect was approximately **0.9904**, meaning the recurrent tangent map sent almost all of the selected basis's mapped energy outside its original subspace. By depth eight, its principal-angle drift reached about **52 degrees**. A direction that describes behavior at one pass can therefore become a poor description of what matters later. This measurement concerns the fitted basis and sampled dynamics; it leaves other ways of steering open.

**DT05's transport pilot** tried to follow a task-relevant subspace across depth using Jacobian-vector products, which describe the effect of a small perturbation on later states. Across eight exposed engineering roots, the transported basis yielded **−0.0078125 exact-board fraction** relative to the static comparator, or **one fewer board-equivalent across 128 root-by-board comparisons**. With sixteen boards per root, the paired root effects were **+1/16, −1/16, −1/16, and five zeros**. An EvalMath planning preflight marked the design `REDESIGN_OR_PILOT_ONLY`; this was an engineering pilot, not a confirmatory trial. The receipt records `git_dirty: true` at head `8ccdcb2` and preserves per-file source hashes. Accurate numerical transport produced little task benefit in a setting where many perturbations were already corrected by the model itself.

A related diagnostic measured how many independent directions influence later outputs. On three roots, the **mean future-response Gramian participation ratio** rose from **2.95 at depth one** to **4.71 at depth eight**. The change came chiefly from one root: the three trajectories were **5.81→10.57**, **1.20→1.75**, and **1.85→1.80**. By comparison, the participation ratio of the broader state remained near twenty. These are local measurements, sensitive to the probe and remaining horizon. They suggest that the dimensions important for a future task response may be far fewer than the dimensions available in the hidden state.

## Sparse components work better when they are selected online

Another BPD experiment decomposed a weight-tied operator into twenty-four shared components and enabled twelve during each rollout. With component selections supplied by later states of a **dense teacher**, a prior replay solved **94.90% of all evaluated Sudoku4 boards at depth eight** and **95.39% at depth twelve** on nineteen retrospectively selected competent roots. These percentages use all evaluated boards as their denominator. They show that sparse components can reproduce much of a teacher's behavior when given privileged information about where the teacher's computation will go [@bpd2026].

The **DT05 online-routing pilot** asked what happens when those component choices must come from the sparse model's own current state. Across eight excluded engineering roots, each scored on 64 Sudoku4 boards, it compared a fixed selection, an online selection, and a privileged teacher-informed selection using the same fitted factor bank. It also retained the unmodified dense model as a reference. Table 2 reports the depth-eight results, while Figure 1 shows the differences within individual roots. The dense scores come specifically from `dense_teacher.by_depth["8"].exact_board_accuracy` in the pinned receipt.

| Route | All 8 engineering roots, mean exact boards | 3 retrospectively competent roots |
|:--|--:|--:|
| Dense teacher | 46.68% | 89.06% |
| Fixed sparse components | 1.56% | 4.17% |
| **Online sparse components** | **27.34%** | **58.85%** |
| Privileged teacher-informed components | 38.48% | 81.77% |
| Random component route | 0.00% | 0.00% |

: Depth-eight exact-board accuracy with a common factor bank and 12 of 24 active components. Eight exposed engineering roots contribute equally; the three competent roots were selected retrospectively.

![Depth-eight exact-board accuracy for all eight excluded engineering roots. Fixed, online, teacher-informed and dense predictions are shown for each model. The teacher-informed route uses privileged states.](routing_by_root.pdf){width=92%}

Across all eight roots, online routing improved exact-board accuracy over fixed routing by **25.78 percentage points**. The gap widened to **54.69 points** among three roots selected retrospectively for teacher competence. These averages hide an uneven pattern. All eight fixed-route solves occurred in those three roots; among the other five, the dense teacher solved **68/320** boards (**21.25%**), online routing **27/320** (**8.44%**), and fixed routing **0/320**. Two roots (`2605903` and `2605905`) had no successful dense-teacher solutions at all. The privileged teacher-informed route exceeded online routing by **11.13 points** overall, although in root `2605904` online solved **8/64** boards and the teacher-informed route solved **7/64**. Teacher information thus provides a useful reference, with no guarantee that its finite-sample accuracy will bound an autonomous policy.

One ambiguity in the source limits how to interpret the competence-selected results. The receipt contains a second field, `raw_depth8_exact_board`, which disagrees with the explicit dense-teacher depth-eight accuracy in **five of eight roots**. For root `2605904`, it reports **33/64** solutions, compared with **22/64** in the dense-teacher field. Selecting competent roots from this raw field would yield four roots instead of three. Because the source does not establish how the raw field was generated, Table 2 and Figure 1 consistently use the explicit dense-teacher value. Fresh training roots and independently evaluated puzzles will be needed before the subset can support a more general claim.

The routing pilot also contains a defective diagnostic. Its archived quantity labeled *top-k set overlap* compared components only when they appeared in the same rank position, then divided the result by the selected count a second time. It therefore cannot measure unordered component-set overlap. The source has been corrected, but no corrected measurement exists for the eight-root cohort. The **accuracy differences remain recorded observations**; the proposed explanation in terms of component membership awaits an appropriate measurement [@bpd2026].

The online arm also incurred additional teacher-mask capture and diagnostic instrumentation inside its timed region. Its recorded elapsed time cannot be read as a clean estimate of deployment overhead relative to fixed routing. The next comparison will need equal instrumentation and a common inference-compute budget.

## Solving a problem can be easier than preserving its solution

The replay of two third-party Tiny Recursive Model checkpoints illustrates another risk of extended recurrence: a model may solve a puzzle and then spoil the answer by continuing to run. On **64 solver-verified 9×9 puzzles**, the locally supplied **6.83-million-parameter [`yagizdevre` checkpoint](https://huggingface.co/yagizdevre/trm-sudoku-extreme-mlp)** solved **60/64 at depth sixteen**. A rule that stopped at the first valid solution retained four earlier correct boards and achieved **64/64** on the same exposed set. The [`gaoxin492` checkpoint](https://huggingface.co/gaoxin492/TinyRecursiveModels-Sudoku-Extreme-mlp) already solved **64/64** at depth sixteen and preserved that score under early stopping [@bpd2026]. The case set (`bench_9x9_64cases.json`) has SHA-256 `68e7b403fba1aa7f43f5241a80eb63f4b44a28055d3530da5255ae951af74201`; the two checkpoint digests are `d67723394a7fe439f1fa498b58196399ee89cd82a794c7da9ce0a8cc3df7fa39` and `932b5e0df4e90021c21a064d4e716154086c46fda7afa676ea294484a30e38c3`, respectively. The study received these weights locally, so those hashes authenticate the tested files but do not identify a particular Hugging Face download revision.

The stopping rule used the puzzle's public constraints to recognize validity, requiring no hidden answer. The experiment shows how a policy can preserve solutions already found by a recurrent solver. Because compacting the active batch changed one BF16 trajectory, exact numerical equivalence between stopping implementations still needs care. These exposed checkpoint results evaluate inference-time stopping, leaving weight improvement for a separate experiment.

# Why interpretable components have not yet produced self-improvement

The experiments expose several places where an attractive intermediate result can fail to carry through to the next scientific claim.

A decomposition can faithfully reconstruct a network and still provide a poor direction for editing it. The exploratory SVD and head-row search produced sixteen low-motion edits whose task outcomes matched random controls. The controlled VPD searches yielded small, inconsistent advantages. This leaves the useful link between an interpretable component and a beneficial weight change to be established.

Recurrence adds another difficulty. The same component can affect a network differently as its hidden state changes. A final-pass mask cannot be assumed to describe earlier passes, and DT04's measured subspace drift shows why a fixed state direction may lose its relevance. Online routing recovered more teacher fidelity than fixed routing in an exposed pilot, suggesting that a controller should attend to when and where it acts.

The choice of task matters as well. The native digit model began near ceiling, and several BPD networks readily repaired the perturbations introduced in the steering tests. With little remaining error to correct, even a well-designed intervention may yield an uninformative performance comparison. Future experiments need failures that a reference intervention can genuinely rescue, leaving enough room to measure collateral damage too.

The first-valid stopping results offer a related lesson: a policy can improve task success while leaving the underlying weights untouched. Teacher-informed routing can likewise reproduce a dense model when supplied with states that the deployed sparse model will never see. To establish self-improvement, a controller must choose and retain its own parameter edits, then show their value on later tasks.

Model editing already has a strong literature on causal localization and the protection of unrelated behaviors [@geiger2021; @meng2022; @meng2023]. The remaining challenge here is unusually temporal: a tiny model uses the same weights at multiple depths, so the intervention both changes future states and changes which states subsequent interventions will encounter. A causal controller must remain useful while its target is moving.

# What would be needed to detonate the proverbial oceans?

BPD suggests the next research program: build a controller that can recognize a failure in recurrent computation, propose a localized change, and test whether it improves future behavior. Each stage below establishes a different requirement before the full self-improvement claim can be assessed.

## A faithful decomposition across recurrent passes

A candidate controller should first reconstruct the base model at each actual use of a shared weight, with component importance conditioned on the state reached at that pass. The pass-indexed VPD adapter is a starting point. It requires full-optimizer equivalence under a declared numerical tolerance and independent validation against the stock network before its importance scores can be interpreted as the effect of temporal conditioning.

Fidelity is necessary for an intervention study because a decomposition that changes the model while being fitted confounds the proposed steering mechanism with reconstruction error. The appropriate initial endpoint is equality of forward and training behavior under an identity policy, followed by measured departures when the per-pass policy is deliberately enabled. Passing this gate would validate an instrument for the causal work ahead.

## Causal directions that remain useful after another loop

The observed BPD subspace defect suggests that an edit should be evaluated by the change it produces in **future task behavior**, not merely by its local alignment with a probe. Candidate directions should be matched for physical dose and compared with signed-gradient, random-subspace, and sham interventions. Before testing improvement, the experiment should verify that the selected failure states can in fact be rescued by a reference intervention.

The relevant endpoint is the incremental effect on an unsolved problem after several recurrent steps, with previously correct solutions and unrelated competencies checked for damage. A learned direction that predicts a Jacobian-vector product but produces no additional task recovery would be a mechanistic measurement result. A direction that consistently rescues independent held-out failures would cross the next empirical threshold.

## A state-conditioned operator with its own memory

The online-routing pilot motivates a small controller that chooses low-rank components using the state it has reached. A compact persistent memory may help when the information needed to select an edit was present in an earlier loop but has become difficult to recover from the current state alone. One candidate design is

$$
 \begin{aligned}
 z_{t+1} &= F_\theta(z_t,x) + \lambda_t B_\phi(z_t,m_t)a_\phi(z_t,m_t),\\
 m_{t+1} &= G_\phi(m_t,z_t,z_{t+1}),
 \end{aligned}
$$

where $m_t$ is a small controller state, $B_\phi a_\phi$ is a low-rank intervention, and $\lambda_t$ bounds its strength. This is a **proposed architecture**, with no training or success result asserted here.

A decisive ablation would compare the persistent-memory controller with a state-conditioned memoryless router, a fixed-factor operator, and randomized routing under matched parameter and inference-compute budgets. A teacher-informed route can provide a privileged fidelity reference and must be excluded from the autonomous arm. Its performance need not bound the online route in every finite pilot. Measuring true unordered component overlap on fresh, exposed-only pilot cases would help explain the existing routing gap before treating it as a training target.

## Retained edits and a closed improvement loop

The stronger test moves beyond steering a single rollout. Let $M_k$ denote the model after $k$ accepted weight edits. The controller uses failures and permitted internal traces to propose a candidate update $\delta_k$. Acceptance is decided on development data; later outcomes are measured on independently held-out inputs:

$$
 M_{k+1}=\operatorname{Accept}(M_k,\delta_k;\mathcal D_{\mathrm{dev}}).
$$

Here *Accept* denotes a prespecified validation gate, including a limit on collateral degradation. It is not an oracle supplied with test answers. The program should track the full sequence of accepted and rejected proposals, including the compute spent searching for them. The appropriate controls include a matched random-edit search and a gradient-based editor given its clearly stated information access.

The central endpoint is the cumulative improvement of $M_k$ on newly drawn task instances under a fixed total compute budget, with repeated evaluation of earlier competencies. A more demanding test asks whether the controller becomes **better at choosing subsequent edits** after experiencing earlier successes and failures. To interpret that trend, task difficulty, training opportunities, and test exposure must be controlled across the sequence. Model improvement after one edit is a promising local outcome; an increasing and independently replicated capacity to make further improvements is the self-improvement phenomenon the title asks about.

This program has an informative failure condition. If selected components do no better than equal-dose controls on new model roots, if online routes fail once teacher information is withdrawn, or if successive edits erase earlier gains, then the hypothesized mechanism has not crossed the required threshold. These outcomes should remain publishable results because they distinguish plausible intermediate mechanisms from an autonomous learning process.

# Limitations and conclusion

The controlled VPD comparisons each involved only three model roots, and both models were much smaller than Goodfire's language-model decomposition. Angsten's independent adversarial analysis of the released larger-model VPD decomposition reported failures under extended adversarial mask searches [@angsten2026]. That study tested a different decomposition and endpoint; it points to a shared need for careful reconstruction and faithfulness checks. Our two controlled models are also substantially simpler than the published 7M TRM, and the native teachers had been examined during development. Some historical searches reuse inputs or model identities, while one earlier sweep survives only as a summary. The results can be independently recounted where the underlying records exist, with their generality limited by the models and exposures studied.

BPD's routing advantage comes from eight exposed Sudoku4 engineering roots, with the three-root competent subset chosen afterward. The DT04 analysis includes roots that failed its original audit, and its competent nineteen-root subset contains all three float32 gauge-audit outliers. In DT05, the unexplained disagreement between `raw_depth8_exact_board` and the explicit dense-teacher field affects the competence boundary. The teacher-informed policy also has privileged state access, while the original mask-overlap diagnostic was flawed. The proposed explanation of routing drift will need fresh roots, the corrected diagnostic, and matched runtime instrumentation. The persistent-memory organelle and a sequence of retained self-improving edits remain prospective.

Weight reuse allows a small recursive model to perform a long computation with relatively few parameters. Our experiments show how difficult it is to turn interpretable components into reliable weight improvements: controlled VPD edits have yet to outperform matched random searches consistently. BPD gives a direction for the next studies, particularly the need to condition intervention choices on the changing state of a recurrent network. The decisive test will be whether such a controller can retain beneficial edits and learn to choose better ones on future tasks.

# Data and reproducibility

The evidence is available in the source repositories and preserved experimental records. A companion reproduction guide, `REPRODUCIBILITY.md`, distributed with this paper, lists exact source paths, provenance boundaries, and executable audit commands. The principal study registrations are [DEUTERIUM-FOLLOWUP-01](https://github.com/MoralityLabAI/LoopedTransformers/blob/39db0ab08a2c7c69b3b0bb04e74b13c7b59bdc7a/mechinterp/deuterium/experiments/deuterium/PROTOCOL.md), [DEUTERIUM-NATIVE-01](https://github.com/MoralityLabAI/LoopedTransformers/blob/39db0ab08a2c7c69b3b0bb04e74b13c7b59bdc7a/mechinterp/deuterium/native_trm/PROTOCOL.md), and [BPD-DT04](https://github.com/MoralityLabAI/BPD/blob/8e02d09c1f67e82b6aed61199df6603a766758b9/mechinterp/bpd/experiments/depth_steering_gym/dt04/PROTOCOL_DT04.md). The DT05 routing and transport pilots are engineering studies, as their saved results explicitly state. The VPD/TRM recovery package is hosted as [*VPD-TRM Original Records*](https://huggingface.co/datasets/AlephFunk/vpd-trm-original-records-202606), with archive SHA-256 `a45db5ef5e603852efb4f247112655c5572eff5aa85f9ab94674a4ae9dc8f003`. The fully counted replication, its selection logic, the controlled steering data, and the native adapter receipts are documented in [*LoopedTransformers/mechinterp/deuterium*](https://github.com/MoralityLabAI/LoopedTransformers/tree/39db0ab08a2c7c69b3b0bb04e74b13c7b59bdc7a/mechinterp/deuterium) [@dugan2026]. The BPD DT04 and DT05 engineering receipts are pinned to [commit `c24c676`](https://github.com/MoralityLabAI/BPD/tree/c24c676bad2f577cba94960093795c7dead5d218/mechinterp/bpd/experiments/depth_steering_gym) [@bpd2026]. The diagnostic correction and retained erratum appear in [BPD PR 136](https://github.com/MoralityLabAI/BPD/pull/136).

The historical data were independently recounted where candidate-level records were available, and the controlled comparisons were replayed from recorded logits and interventions. The BPD companion aggregates were recomputed from their root-level receipts. The paper reports a synthesis of existing measured outcomes, with the experimental designs in the BPD section identified as prospective. No reserved teacher or confirmation cohort was opened to produce this synthesis.

# References

::: {#refs}
:::