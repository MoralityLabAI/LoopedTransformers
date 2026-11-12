# RMP-GG-09: Hidden-Markov belief transfer beyond the fixed-world Bayesian grammar

**Registered November 12, 2026, before training or viewing GG-09 outcomes.**
Base: LoopedTransformers \`main\` at \`ffee28c83ab8320aa8749b567280b23780dc2a78\`, with GG-08 merged.

## New task grammar

GG-01–08 use a binary latent world \(S\in\{E,D\}\) **held constant** while observations arrive. GG-09 defines a two-state *hidden Markov model* in which the current world can change between observations. The prior is \(P(S_1=E)=p\); between successive cues the world switches with the **fixed** probability \(q=1/4\). Conditional emissions remain \(P(1\mid E)=3/4,\ P(1\mid D)=1/4\). The reward ratio \(r\in\{1/2,1,2\}\) has the same semantics as GG-01. The network input has the same two-dimensional context \((\log(p/(1-p)),\log r)\) and the same binary tokens, so a new task does not receive a special indicator.

The exact rational unnormalized forward filter initializes \((a,b)=(p,1-p)\). For cue \(x_t\), at \(t>1\) first compute
\[
a^-=(1-q)a+qb,\quad b^-=qa+(1-q)b,
\]
then multiply \(a=a^-P(x_t\mid E)\) and \(b=b^-P(x_t\mid D)\); at \(t=1\) directly apply the emission to the prior. The exact gold choice is **protect iff \(r a>b\)**, ties favor exploit. Prefix choices use the same recursion. The observation marginal likelihood is \(a+b\) before reward weighting.

This is a genuinely different *mathematical conditional-output grammar* even though the sensory vocabulary is reused. Exhaustive exact-rational enumeration of the complete \(9\cdot2^n\) context/sequence grammar yields label disagreements between \(q=0\) and \(q=1/4\) of **35/144 at depth 4, 164/576 at depth 6, and 727/2304 at depth 8**. The CI oracle independently checks these counts and a brute-force hidden-state-path oracle on shorter sequences. No claim is made that this is a novel real-world distribution.

## Frozen input identity split and datasets

At each depth \(n\in\{4,6,8\}\), enumerate all \(9\cdot2^n\) exact \((p,r,\mathbf x)\) combinations. Sort SHA-256 identifiers over a canonical \`gg09.markov-input.v1\` JSON object containing rational prior/ratio and bitstring. First **2/3** of sorted combinations are training support and final **1/3** are heldout: counts **96/48** (depth 4), **384/192** (depth 6), **1536/768** (depth 8). Splitting is by complete input/context combination, and the two training arms are restricted to the SAME training support. The original GG-08 source has previously seen those combinations under the *constant-world* task; they are held out only from the **new GG-09 continuation**.

For model seed \(s\), depth \(n\), build pools of **8192 draws with replacement** from each task's exact training-support roster. Draw probabilities are proportional to the exact sequence likelihood \(P(\mathbf x\mid p)\) under either the static \(q=0\) or Markov \(q=1/4\) task. Contexts and prefix gold labels are computed independently using that arm's exact oracle. Pool generation uses \`random.Random(94000 + 100*s + 10*n + armcode)\`, with armcode 0 for static control and 1 for Markov training. A finite grammar row's likelihood never depends on the neural output. Train/heldout SHA-256 rosters and pool hashes are fixed and saved.

## Models and matched continuation

- Source: reproduce the **GG-08 mixed-depth** endpoint: \`train_cell(seed,steps=400,batch=128)\` then \`continue_training(source,seed,"mixed",training_pools(seed),steps=300)\`, for seeds **11,17**. Exact same native width-32 tied \`ExplicitDecoderBlock\` and unchanged task-context encoding. Retain source and intermediate weight hashes.
- Two independent cloned continuations, both starting from the identical frozen GG-08 mixed endpoint: **static** \(q=0\) and **Markov** \(q=1/4\). Run **240 AdamW steps**, batch **128**, cycling depths 4/6/8 exactly **80 updates per depth**. AdamW LR **0.001**, weight decay **0.001**, betas (0.9,0.999), global gradient clip **1.0**; terminal exact action CE plus **0.2** prefix CE. Torch CPU, one thread. Sample each depth with a fixed \`torch.Generator().manual_seed(97000 + 100*s + armcode)\` and replacement from its registered 8192-example training pool.
- Both arms have equal optimizer steps and **184,320 total sample-visits**. They draw from task-specific natural input likelihoods on the same finite training-support identities; this is a matched-step/visit *domain continuation*, not a guaranteed equal-FLOP or identical-sample-label training comparison.
- Fixed final checkpoint at step 240; no early stopping, no evaluation-based hyperparameter changes. Keep training-loss traces at updates 1,60,120,180,240 and model SHA-256 hashes. The source checkpoint remains unchanged.

## Independent scoring panels

For each depth and four scoring replicate indices 0–3, build panels entirely from the **heldout third** of the full input/context grammar. Each 160-example panel contains:
- **Natural** 64 draws from heldout support, weighted proportional to the exact Markov sequence likelihood, seeded \`random.Random(95000+100*depth+rep)\`.
- **Boundary** 32 draws from the lowest quartile of absolute exact *Markov* log decision margin on heldout support, seeded \`96000+1000*depth+10*rep+1\`.
- **Rare** 32 draws from the lowest quartile of exact Markov observation likelihood on heldout support, seed offset +2.
- **Late reversal** 32 draws from heldout support with a Markov decision after \(n/2\) cues different from its full decision, seed offset +3.

Every cohort's membership is fully determined by the external rational HMM oracle, before neural evaluation. Repeated draws are retained as independent empirical scoring observations from a finite support. All four scoring replicates, the panel SHA hashes, exact gold labels, unique-case counts and the complete training/test split hashes are frozen before any model scoring.

For each of the three arms (**GG-08 source**, **static continuation**, **Markov continuation**) and model seed, compute 12 depth×cohort accuracy cells under Markov exact labels. For the **same scoring inputs** separately compute accuracy against the static \(q=0\) oracle to quantify task-incompatibility/forgetting, and record the exact oracle disagreement fraction. Accuracy is calculated before every physical update.

## Physical SGD and protected-tail scoring

Source report queries are the entire **48 depth-four heldout** input/context identities, newly labeled with the Markov exact oracle. Report membership is separate from both new continuation training pools. For each model arm and query, generate clipped truthful and flipped *Markov* report gradients from the frozen endpoint and perform **two actual PyTorch SGD steps** on independent clones of the same task-model weights, LR **0.05**, global gradient clip **1.0**, no momentum or weight decay. Score raw per-example post-update CE under the *Markov* gold labels across all depths and all four cohorts.

There are **3 arms × 2 seeds × 48 queries × 2 actions = 576 physical SGD steps**, each scored on 1,920 gold observations, giving 1,105,920 per-example CE measurements. Retain all action-conditional raw losses, original weights, source/query hashes and registered training curves.

Apply GG-07's **unchanged** product-TV robust scorer: natural/tail mix \(\lambda\in\{0,1/4,1/2,3/4,1\}\), within-cohort TV radii \(\rho\in\{0,1/64\}\), and depth mixture weights \(\alpha_d\ge1/6\), \(\sum_d\alpha_d=1\). The primary protected score is at \(\lambda=1/2,\rho=1/64\), with sharp lower/upper bounds and sign ambiguity threshold \(10^{-7}\). Retain all status categories: robust harm, benefit, unresolved.

## Registered decisions and interpretation

**Competency transfer gate:** the Markov-continuation model must reach at least **90% exact Markov action accuracy on every one of 12 depth×cohort scoring cells**, for *both* model seeds, and its minimum-cell accuracy must improve by at least **0.05** (five percentage points) over the matched static continuation on the **same Markov** scoring panels. Report every failing cell.

**Protected-harm transfer gate:** both Markov-continuation seeds must first satisfy competency, then each must have at least **24/48** report queries with a robustly positive or negative protected update-harm sign; each must improve by at least **8** robustly signed queries over the static continuation on the same Markov gold. Reuse GG-07/GG-08's strict sign criteria.

A separate **incompatible-grammar readout** reports the Markov model's exact static-task accuracy on the identical input panels, alongside prior static-source performance. Strong Markov performance at the cost of static accuracy is a measured representational tradeoff. It is not a model recognizing which evaluator is watching.

Every scientifically negative result stays in the original score ledger. A GG-09 failure may arise from finite compute, model capacity, contradictory task labels for the same inputs, or generalization difficulty; it is not evidence of spontaneous goal guarding. The physical SGD result is conditional on this toy grammar and source implementation.

## Verification and causal scope

- Separate exact-rational HMM forward filter, independent hidden-state path enumerator on 2–4 cues, exact \(q=0\) reduction to GG-01, full 4/6/8 enumeration disagreement counts, frozen train/test membership checks and heldout challenge identity certificates.
- Exact equality of source weights for both continuations, registered optimizer-step and sample-visit counts, frozen model-source hash after 576 physical updates.
- Independent native one-step SGD parity with GG-04's reference auditor using a new Markov gold \`CuePanel\`; source/treatment/no-update identities.
- Full versioned source-bound raw-loss ledgers and model checkpoints uploaded by the dedicated GitHub Actions workflow.

This is **an independently specified reasoning-task grammar**, evaluated with true physical gradient interventions. No agent is trained to infer its own training status, no reporting-policy weights are secretly self-edited, and no observed behavior licenses a claim of deceptive alignment, simulator bargaining, or automatic goal guarding.
