# LRM-REV-02: learning to recover a posterior through truthful reminders

## Status and provenance

This protocol fixes a public, development-only CPU pilot before its training or development examples are generated. The protocol commit precedes the implementation freeze. The runner requires the exact frozen Git commit and records SHA-256 hashes for every registered source file before fitting a model.

The mathematical source is [Beyond Bayes BB-CQ-05](https://github.com/patrickdugan/beyond-bayes/pull/24), commit `abc89f73d223f00305eb0416a34fe5aee602494d`. The corrected input contract is [LRM-REV-01](PROTOCOL.md), merged in [#205](https://github.com/MoralityLabAI/LoopedTransformers/pull/205), with the independent [EvalMath EVAL-REV-01 reference](https://github.com/MoralityLabAI/EvalMath/pull/53). Existing experiment records and protected populations retain their original status.

The machine-readable settings are [REGISTRATION_LRM_REV02.json](REGISTRATION_LRM_REV02.json). This pilot always reports `ENGINEERING_PILOT_ONLY`. Its preflight status is `PENDING_MATCHED_SHAM_AND_CONFIRMATION_DESIGN`; [the preflight note](PREFLIGHT_LRM_REV02.md) specifies the missing inputs. Four independently trained model roots supply engineering observations. The runner has no confirmation phase.

## Question and information available to the learner

Does repeating the old observations at corrected positions improve a GRU's posterior recovery relative to repeating the same number of truthful observations at random positions?

Use the existing three-regime law with twelve named old bits `X`, six correction positions `S`, and six accepted replacements `Y`. Priors are `(9/20, 7/20, 1/5)`; informative match probabilities are `(19/20, 13/20, 3/20)`. The host chooses `S` independently of outcomes. Old bits in `S` are fair provisional noise, and the remaining old bits and accepted replacements follow the regime likelihood.

The learner receives all twelve named old observations before seeing any correction positions. Accepted `REPLACE` events then reveal `S` and `Y`. At that point its cumulative input is `H=(S,X,Y)`, and the full-history Bayes posterior `p_H` is the reference for every reminder arm and stage. A reminder repeats a value already supplied in `X`.

Each token concatenates a three-way event-type one-hot (`OLD`, `REPLACE`, `REMIND_OLD`), a twelve-way position one-hot, and a two-way bit one-hot. There are seventeen coordinates, with squared input norm three for every event. Public `n=12,j=6` are fixed by the registered task. Source identifiers and private regime labels remain outside model inputs.

## Independent units and source partitions

Model roots are exactly `0,1,2,3`. Each uses independently derived initialization, minibatch, training-source and development-source roles in a new `LRM-REV-02` hash namespace. Every root has 2,048 training episodes and 512 development episodes. The resulting 2,048 development episodes contain repeated measurements nested within four independently trained models.

The generator accepts only `fixture`, `train` and `development`. Fixture examples support code checks before the source freeze and stay outside the pilot partitions. The runtime generates training and development data only after validating the frozen source. Reminder controls derive from separate public source roles, independent of outcomes. Root and example bounds are enforced; unknown phase names are rejected.

Every root is retained in its registered order. A failed or incomplete root remains in the run record. A new architecture, learning schedule or root population requires a prospective amendment that preserves this run.

## Model and training

The model is a one-layer GRU with input size seventeen, hidden size thirty-two and a linear three-class softmax readout: **4,995 trainable parameters**. Use CPU float32, one Torch thread and deterministic execution. The fixed optimizer is Adam with learning rate `0.003`, zero weight decay and batch size sixty-four. Each root takes exactly 800 optimizer steps, sampling its fixed training population through a separate minibatch RNG. The final scheduled checkpoint is scored.

Each training example receives either corrected-slot reminders or a uniform random six-slot reminder set, selected through an outcome-independent source role with probability one half. The choice stays fixed for that example. Tokens within each phase use ascending position order.

The training objective is the weighted sum of squared posterior errors:

```
0.25 * ||a_after_old - P(K | sum(X), n, j)||^2
+ 0.25 * ||a_after_replacements - p_H||^2
+ 0.50 * ||a_after_reminders - p_H||^2.
```

The initial target integrates over the hidden correction schedule. Full-history supervision begins after all six replacements arrive. Development scores have no role in checkpoint selection or training termination.

## Reminder policies and matched exposure

After the original `OLD + REPLACE` stream, clone the same reached hidden state for the candidate and all controls.

| Arm | Continuation |
|---|---|
| Candidate | Six truthful old-bit reminders at the corrected positions `S` |
| Control `b`, for `b=0,1,2,3` | Six truthful old-bit reminders at a uniformly selected six-slot set `T_b` |

Keep every sampled `T_b`, including duplicates and sets that happen to equal `S`. All arms use the same reminder token type and ascending-position order. Their event counts and input norms agree. Record the reached hidden-state displacement at each stage, since input-norm matching does not constrain native displacement.

The random-slot policies are **active controls**: their reminders can improve the model's estimate. They are unsuitable as a fabricated null-sham distribution for a mechanistic-readiness calculation. The question concerns reminder selection. Native-state donor interventions are outside this pilot.

## Endpoint and diagnostics

For an episode and forecast `a`, score

```
excess(a) = ||a - p_H||^2
expected_Brier(a) = excess(a) + 1 - ||p_H||^2.
```

At scoring, convert the CPU float32 logits to float64 before softmax. Compute these scores in float64 from the stored probabilities and exact-rational Bayes targets converted once for display/scoring. Training softmax remains float32. This integrates the private regime label under the declared generator and keeps the endpoint in unhalved three-class Brier units.

The sole primary contrast is fixed at six reminders:

```
D_root = mean_episode(mean_b excess(control_b, r=6) - excess(candidate, r=6)).
```

Positive values favor corrected-slot reminders. Report the mean over four roots and each individual root value. The scientific effect floor is `0.01` Brier. Report its attainment descriptively; this pilot cannot grant a scientific pass or confirmation release. All stages `r=0..6`, the initial forecast and the post-correction forecast are retained as diagnostics. Stages and control sets remain repeated observations within a model root.

The registered engineering competence thresholds are mean initial posterior error at most `0.01` and mean candidate error after six reminders at most `0.02`, both assessed separately for every root. Also report the random-control Bayes excess as available headroom for the `0.01` target. These thresholds describe this instrument's competence and leave the overall evidence verdict unchanged.

Replay the identical original stream from a reset state, once as a single prefix and once split into its OLD and REPLACE phases. Forecast probabilities must agree within `1e-6`; apply the same bound to output logits and reached hidden coordinates. All arms at `r=0` must share the same forecast. Reject nonfinite or unnormalized probabilities and retain the failure record. Scoring uses batch size one. A replay of a different accepted-history token grammar is outside this contract.

## Compute cap and receipts

The entire training/development collection has a 900-second wall-clock cap and a two-GiB resident-memory cap. Run one root at a time on one CPU thread, with GPU allocation disabled. A cap hit preserves completed outcomes and the interrupted root's status. Automatic retries and extra roots are disabled.

Archive source hashes and environment metadata together with training traces, checkpoint hashes and bytes, per-episode inputs, complete probability curves, hidden displacements, runtime and peak RSS. An independent receipt audit recalculates Bayes targets and reported score aggregates from those records. Published results must retain partial runs and failed engineering gates.

A separate checkpoint replay reconstructs the token encoding and GRU forward path from the archived inputs and saved weights. It runs each complete arm from a reset state, obtaining every reminder-stage forecast from a single twenty-four-token pass. Compare these forecasts with the runner's segmented continuation at absolute tolerance `1e-6`; the root-level paired effect must also agree within `1e-6`. This replay uses the existing development records and preserves their evidence status. It loads frozen checkpoints and performs no optimization.

The bounded pilot supplies development evidence for a later design. A future mean-effect gate needs a declared matched-null experiment and a coverage argument for the same paired root-mean statistic, together with a separate confirmation protocol. The current strict EvalMath CLI cannot be populated honestly from active controls and an unallocated confirmation population.
