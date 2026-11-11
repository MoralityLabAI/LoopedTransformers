# RMP-MG-01 development protocol

Date: 2026-11-11. Target repository: MoralityLabAI/LoopedTransformers, branch rmp-v0-phase2.

## Scope

The initial experiment measures task-state visibility and local causal sensitivity in frozen fixed-depth LoopedChase models. It uses the existing semantic-echo native adapter. The attached Systems Philosophy transcript motivates the question about readable reasoning. The toy model provides a native cursor readout; it has no natural-language chain-of-thought channel. A claim about CoT monitoring requires a later model and observation channel that produce CoT.

Every receipt must declare its observation map, checkpoint hash, visit, horizon, intervention scope, and status. Existing Q4/Q5 models are development data because their extrapolation outcomes have already been examined. Fresh training seeds are required for prospective prediction claims.

## Part A: bounded-probe progress (implemented)

Choose fixed target horizons K=8 and K=16, and analyze each horizon separately. The target within a comparison is always Z_K=P^K(s). Use visits 1,2,4,8 for K=8, and add visit 16 for K=16. No claim of monotonic progress is assumed.

Separate prompt ingestion from useful recurrent processing. Compare an input-only feature representation with [input features; state at visit t], using a fixed feature layout and padding where necessary. Report the native head independently. Probe families should include affine logistic and a bounded small MLP, with parameter counts, optimization steps, regularization, and inference costs recorded. Fit any projection or normalization using development training data only.

Split by underlying permutation graph so alternate starts, edge orders, and renamed or revisited versions of the same graph cannot cross splits. Use explicit graph hashes; generating new item seeds alone does not establish graph-disjointness. Select hyperparameters on validation graphs and seal a test set. Report a shared decoder across visits alongside separately fitted visit decoders, so representational drift remains visible.

Primary quantity: paired held-out log-loss change for a fixed target. Accuracy is a secondary measurement. Keep a loss estimate and its uncertainty separate from the population infimum defining predictive V-information. The fitting pipeline is `research_gym.scripts.loopscope_bounded_probes`. It saves every candidate validation score, the chosen decoder weights, and the shared training-only normalizer before test scoring. Raw item losses accompany graph-clustered summaries.

## Part B: causal readout sensitivity (implemented)

The adapter loads frozen checkpoints through `loopscope_semantic_echo.load_model`, reconstructs all token states, and checks parity against `LoopedChase.forward`. It preserves the native input reinjection and every core block. The observation is centered native cursor logits. A stronger observation-channel control uses centered logits read from every token.

At each sampled item and declared visit, compute Jacobians C of the observation and B of future cursor logits after a fixed continuation horizon. Extract the strongest numerical observation-null direction. Keep every sampled item in the output, including zero directions and incorrect baseline futures.

Evaluate the chosen direction at both signs and relative L2 radii 1e-4, 1e-3, and 1e-2 times the state norm. For every radius and sign, compare equal-size full-state random directions with random observation-null directions. Also run per-token constant shifts, which test the LayerNorm gauge, and zero-write controls. Each raw effect reports actual perturbation norm and both output drifts. Oracle correctness and changes in the future argmax remain separate fields.

The Jacobian-guided direction is optimized using the chosen future output map for each item. Its resulting sensitivity is an assay of possible local effects. It does not independently predict an unseen intervention's direction or demonstrate a learned monitor-evasion strategy. All optimized and control results remain in the report.

For a scientific run, repeat the SVD rank-tolerance sweep and compare float32 with float64 on a subset. Check directions at radii above numerical noise. Report state dimension and singular-value spectra. Keep baseline-correct and baseline-incorrect strata, with all-items coverage first. Include random untrained models at the same dimensions.

The code records observation drift in centered logits. It does not declare a safety monitor fooled when that drift is small. The actual monitor receives its complete allowed input, and direct monitor performance is a separate pending experiment.

## Part C: semantic and monitor validation (pending)

Connect candidate directions to a declared symbolic variable using counterfactual cases with exact oracle consequences. Use same-input semantic-echo donors to test phase consistency, and matched natural donor states to estimate how far perturbations leave the state distribution. An intervention that simply damages an answer is counted as damage until it satisfies the declared semantic criterion.

Measure the complete observation history available before the monitored action. For actual monitor evaluation, freeze the monitor on development data, calibrate a false-positive threshold on an independent calibration set, and report sensitivity plus false-positive confidence intervals on held-out data. For a binary harm or policy-violation label, use an independently checked behavioral outcome. Toy cursor errors cannot establish a harmful intention.

Use 5% false-positive rate for an initial adequately sized pilot. A 1% operating point requires enough independent negative examples to support a useful uncertainty interval. Bootstrap or model comparisons must cluster by graph and checkpoint; thousands of token/visit measurements from one model do not provide thousands of independent training-seed observations.

## Decision rules

A decrease in bounded-probe loss supports increased accessibility for that probe budget and target. Positive local sensitivity supports a causal effect outside the chosen observation Jacobian's numerical row space. Finite-radius effects determine whether the first-order construction survives at the tested scales. Evidence that a natural computation contains a semantic variable requires its own intervention and oracle validation.

A claim of reduced monitorability requires direct evidence from a declared monitor and outcome distribution. Deception additionally requires evidence about the agent's behavior and incentives. This assay supplies neither by itself.

## Compute and integration

The initial run uses frozen weights and small item counts. It schedules no actor training and no cloud jobs. Full Jacobians can be expensive at large widths; begin with four items and a small visit set. The projection implementation avoids a dense d-by-d projector. The checkpoint adapter is additive and imports the existing semantic-echo implementation rather than copying model-forward code.

Local validation now includes 81 passing tests, ten constructed controls, and native-forward smoke execution using selected definitions extracted from the pinned upstream source. The validation receipt distinguishes this extracted dependency snapshot from a complete repository checkout. A dedicated lightweight workflow runs the same interfaces against the repository modules. Trained Q4/Q5 evaluation still requires the external checkpoints.


## Implemented fitting contract, revision 2

The fixed target horizons are 8 and 16. Visit zero is included in every study. One affine softmax classifier and one ReLU MLP family are evaluated. The default MLP has 32 hidden units. Both modes use exactly 200 AdamW updates per candidate, each with 128 sampled feature rows; a shared decoder receives the same update/example budget as each individual visit decoder. Candidate learning rates are 0.003 and 0.01 with weight decay 0.001. Smoke receipts can use smaller, explicitly recorded budgets.

The feature vector concatenates the prompt embedding at every token with the complete carried state. One normalizer is fitted to the training graphs pooled across selected visits, including zero. Its mean and scale are reused for validation and test. Separate-visit decoders share the same initial weights and sampling seed for a family/configuration. The shared decoder has no explicit visit token. Selection uses mean validation loss with equal weight per structural group and per selected visit. Every candidate receives the fixed budget. Test loss cannot choose hyperparameters or a stopping point.

Node relabelings of permutation graphs are exactly characterized by their sorted cycle lengths. The runner assigns whole cycle types to one split. This enforces the protocol's restriction on renamed graphs and creates a structural holdout. Types are selected uniformly from integer partitions of n, followed by random node renaming, edge order, and start node. The estimand gives these selected types equal weight. It differs from uniform sampling over labelled permutations. The actor's original training distribution is outside this split guarantee.

Enumeration is limited to at most 32 nodes. Requested group counts must fit the available number of cycle types, or the command fails. The default counts are 12 training, four validation, and four test groups, each with 16 sampled prompts. Additional prompts within one type increase within-group precision; they leave the number of independent structural groups unchanged. Confidence intervals use paired resampling of whole groups. With one group, the interval is withheld. Intervals from a small number of held-out types remain exploratory and conditional on the frozen actor.

The optional shuffled-label control permutes training targets once and uses the same shuffled target for an item at every visit. Validation and test retain their oracle labels. Probe receipts include optimization budgets, parameter counts, approximate linear multiply-accumulate counts, and parameter hashes. Two selected families across shared/separate modes, true/shuffled training labels, and both horizons produce 16 scored studies.

Actor parameters stay frozen. A before/after hash checks this requirement. The selected decoder receipt is written before test predictions and its hash is checked afterward. Raw split manifests retain each permutation, start, and edge order. Saved weights and normalizers permit inspection and replay.
