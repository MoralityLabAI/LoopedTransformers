# NAN-FIT-07: fresh-source pretrained VPD/BPD fitting pilot

This is a bounded development pilot of the real pretrained 3B model. It follows
the complete native BF16/NF4 evaluation and eight SAE fits, and the passing
native-kernel gradient / genuine four-term optimizer engineering fixtures. Those
earlier outcomes are exposed. Do not label random-model checks as pretrained
fits, SVD as learned VPD/BPD, or this fitting pilot as donor-specific transfer.

Freeze all pilot code/config/data/adapter hashes and push before this condition's
first pretrained inference. Keep the original ARG-03 168-case test sealed: no
read, regeneration, exclusion use or inference. No model downloads or changes.

## Cohort and gates

Generate exactly eight new CAL roots from seeds 2026112107..2026112114 and eight
new DEV roots from 2026212107..2026212114 using the already public pointer-root
generator. Each has three source demonstrations and one one-hop query. Add an
explicit answer-copy control for every root: 32 contexts, 16 root groups. Audit
all graph walks independently and reject main-prompt overlap with older public
fixtures. Do not consult protected data for overlap. No model-result filtering,
task repair, changed prompts, truncation or fitted parser.

The worker reads only public prompts, schema and split/control metadata. The host
alone reads gold. Before fitting, run all 32 clean contexts for both native
passes. Report free next-token predictions, not constrained labels or a complete
generation score. The bounded anchored parser grades that single token. This
endpoint is distinct from native prefix-stop and Prime full-final-reply scores.

The predeclared task screen requires at least 6/8 two-loop clean main answers and
8/8 copy answers in EACH split, independently for each backend. Keep every case.
A host-produced hash-bound gate permits that backend's fits; a failed gate is a
preserved negative result, never repaired or silently retried. It does not change
the original failed algebraic gate or authorize BPD donor transfer.

Verify real full-vocabulary head global argmax/selected logits and double-precision
log-normalizer against the existing chunked scorer. Require exact native all-one
closure, including an autograd-enabled rollout, and float32 factor-plus-residual
relative L2 closure <=1e-5. Any unsupported geometry or nonfinite value fails the
condition. Pretrained and reusable-stage parameters remain frozen, with target
parameter hashes before/after each condition.

## Fitting recipe

Initial quartet: BF16/NF4 × T=1 VPD / T=2 BPD, one shared rank-8 factor pair for
physical layer-21 down_proj, dimensions 10752 -> 3072. Preserve backend-effective
rank-8 SVD initialization from NAN-GYM-05 and its hash; verify the actual effective
operator matches before reuse. Do not fit a dense surrogate in place of the
native kernel. The native additive component/residual hook remains authoritative.

Run the actual pinned BPDTrainer at 8f658b789138d73f497b1a2a05f5783745147d40, with
all four terms: faithfulness weight 1e5, minimality, stochastic reconstruction
weight .5, and a two-inner-step persistent adversary weight .5. Exactly eight
joint optimizer steps, B1, first-to-last CAL main root once each; no copy/DEV
fitting. No warmup or best-checkpoint selection. Save initialization and final
factors, controller, means, optimizer/PGD diagnostics and every failed attempt.
Use the fixed seed and optimizer settings in config.json.

The controller is the genuine state_only pointwise network: projection width 32,
zero extra attention blocks, RMS normalization and the framework's output map.
It observes native context-bearing preweight states; no gold, pass clock or
future module inputs. Pointwise evaluation permits exact replay on cached
last-token states. T=1 uses zero ablation/no sphere, mean minimality/zero temporal
switch, and single-pass adversary scope. T=2 uses pass-conditional mean/sphere,
pass-max minimality and the shared-batch-and-pass adversary. These are declared
method conditions, not an isolated causal effect of recurrence.

Recompute exact equal-root CAL means from all token positions before initial
diagnostics and after the final step; use current factors and the actual native
kernel outputs. No DEV mean estimation. Stock EMA updates still operate during
training. Record both the training EMA and the post-fit exact CAL statistics.
Rotation seeds are source_seed plus CAL-root index. Preserve the initial SVD
residual-neutralized pathway on all fresh main contexts before fitting.

For training fragility normalization, measure the library's same-pass full-head
KL response to four initial-SVD component-0 activation-RMS-matched read rotations
per CAL root/pass at scale .25. Keep raw per-root values; freeze their root mean
before optimization. T=2 uses these tolerances, T=1 retains its defined VPD unit
weights while reporting its calibration. This estimates a training normalizer,
not a 95th-percentile sham envelope: eight roots cannot support a 95%/95% tolerance
bound (59 required). Do not count rotations/passes as additional roots.

## Fixed diagnostics and SAE cross-reference

After the final step, keep factors/controller frozen. Evaluate all 16 fresh main
prefixes with all-one, component-only, static CAL per-pass/component mean,
clean-trajectory-informed and actual reached-state masks. Static masks average
CAL roots and token positions equally within root; no DEV averaging. Stock
training uses clean-informed schedules. The online diagnostic calls the learned
controller INSIDE the native module on the state actually reached after earlier
edits. Record reached-input/mask differences, actual physical write displacements,
full-vocabulary per-pass KL and free-token predictions. T=1 online and clean
controller inputs must agree. Routing modes are not physically dose matched and
are fidelity diagnostics, not a causal mediation comparison.
All-one includes the full residual. Other diagnostic modes set the residual mask
to zero: in T=2 pass-mean ablation this replaces it with its CAL mean rather than
zero. Thus "component_only" means the framework's residual-neutralized pathway,
not a rank-8 reconstruction of the entire native operator.

Probe the frozen layer-21 SAE dictionaries and all registered NAN-GYM-05 contexts
against the actual trained output directions, initial SVD and norm-matched Haar
controls in physical coordinates. Down_proj output adds immediately to the
observed residual_after_layer, BEFORE final loop RMSNorm. Do not bridge directly
to normalized loop boundaries. Report all features, loops and source groups;
never equate independently fitted BF16/NF4 feature indices. T=1 application at
loop 2 is explicitly out-of-training-depth. Keep routing derivatives local.

The initial quartet does not finish the rank/site campaign. Rank-16, o_proj's
intervening-MLP bridge, other layer matches, clock/visible gaps, physically matched
donor/dark/random controls and genuine causal transfer remain further stages.
No o_proj-to-SAE identity bridge or positive mechanism is inferred here.

## Evidence and resources

Unit: source-generator world, with prompts, phases, masks, rank and backend as
repeated measures; one pretrained checkpoint. n_units=8 CAL + 8 DEV, n_cases=32
clean contexts/backend plus fixed within-root diagnostics. Primary causal contrast,
physical causal dose and matched-sham correctness floor: not yet calibrated.
Future donor-specific recovery minimum proposed for planning: 0.20 absolute
correctness over a matched random-dark control, requiring separate calibration.
Planned/actual formal inferential comparisons=0. Power/resolution cannot support
confirmation. Readout exposure and all candidate conditions are disclosed. Verdict:
development fitting/instrument pilot only; no causal or equivalence claim.

Run one large model at a time after the registered Prime workers exit; NF4 then
BF16. Keep CPU audits/probes/reporting concurrent. CUDA fraction .75; CPU head and
embeddings, full-vocabulary chunks 8192, native two-pass reference, no KV cache.
Immutable CPU weight reuse and a shared 2-GiB added-snapshot cap protect host RAM.
Bound each backend worker to 7200 seconds between stages/steps. Preserve partial
checkpoints and failure receipts. Do not restart on an observation timeout or
blindly retry an errored fit. A stopped/incomplete Prime job does not authorize a
second simultaneous model; verify actual model process termination first.

Publish aggregates, source hashes, fit/closure/memory receipts and complete failure
counts. All raw outputs, per-record codes, routing text, tokens and hidden vectors
remain local. No paid/cloud job, altered checkpoint, or automatic unseal.
