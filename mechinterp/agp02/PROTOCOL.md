# AGP-02 native grader fibers | Recovered traces and AGP-03 causal next test

**Retrospective diagnostic + prospective study registration. November 19, 2026.**

The original prospective learned-model experiment lives in [Control-Harness PR #73](https://github.com/MoralityLabAI/Control-Harness/pull/73) and the independent verifier-only recovery in [PR #75](https://github.com/MoralityLabAI/Control-Harness/pull/75). Its first registered source was commit `3f9b3e0f710a19008f63808154ee02d0d02df2f4`; the request-only commit was `19e2dc4d32f0fa9926722e9020253e518d9229a2`. [Original first-readout run 44483250239](https://github.com/MoralityLabAI/Control-Harness/actions/runs/44483250239) archived its complete learned-checkpoint and native-state evidence; [audit-only run 44483708165](https://github.com/MoralityLabAI/Control-Harness/actions/runs/44483708165) independently reconstructed all 73,728 host-scored proposals and 368,640 recurrent hidden states from those exact original artifacts.

## Observed native paired-state geometry (retrospective)

The public grader cue is injected at step seven. Matched hint-0 and hint-1 trajectories have equal states before it is encountered. For each seed, arm and case, `d_t = h_t(hint=1) - h_t(hint=0)` is an actual input intervention response in the **unmodified trained model**. It is not a hidden-state causal edit.

The registered models are 16D GRUCells iterated 12 times with initializers 103/211/307. The 128 host roots have 24 cases each. Analysis here aggregates individual cases within root and reports means across three fixed learned checkpoints. It has no new confirmation data.

| Arm | Mean ||d8||2 | Mean ||d12||2 | Uncentered top-PC energy at d12 | Centered top-PC energy at d12 |
| --- | ---: | ---: | ---: | ---: |
| `reward_target` | 4.28145 | 7.23835 | 0.99997 | 0.8098 |
| `host_trait` | 1.31899 | 1.70129 | 0.96068 | 0.6175 |
| `mixed` | 1.53903 | 2.10574 | 0.93410 | 0.8800 |
| `trait_cf` | 0.65475 | 0.55135 | 0.85780 | 0.8351 |

These are **post-result exploratory statistics**. In particular, an uncentered top eigenvector captures the common mean response, so a high rank-one energy fraction is not itself evidence that a controllable one-dimensional latent trait has been isolated. The original behavioral finding was a root-cluster mean forbidden-case susceptibility `0.40213` for host-trait supervision versus `0.15473` under paired-grader invariance regularization. The registered primary contrast `reward_target - trait_cf = 0.8452691` has a descriptive whole-root interval `[0.8289931,0.8615451]`. Unauthorized host executions remained zero.

The `mechinterp/agp02/trace_audit.py` module independently checks the fixed public evidence format, paired cue coverage, checkpoint binding, native pre-hint state equality and both centered and uncentered second-moment spectra for every readout depth. It consumes archived original `native_traces.jsonl.gz` only on explicit invocation; CI runs synthetic source-only fixtures.

## AGP-03: registered future native causal direction experiment

**This protocol is informed by the completed AGP-02 development results. Its planned causal outcome must use fresh OS-derived experimental roots and a separately sealed implementation/request.** Existing AGP-02 train roots are for candidate fitting only; its calibration and development cohorts are prohibited for confirmation.

### Frozen checkpoint and training-only direction

Load the exact original AGP-02 frozen-fit model artifact by checked SHA-256, with seed-specific 16D GRU weights unchanged. Retain each checkpoint's complete native prefix/suffix execution and trace contract. For each of the four trained objective arms and each seed, derive a candidate one-dimensional physical direction from **the original 256 training roots only**:

    v = normalize(mean_case( h8(hint=1) - h8(hint=0) )).

The direction sign points toward the high-hint state. Fit each half of the training cohort separately as a stability diagnostic and preflight gate `cos²(v_half1,v_half2) >= 0.95`. Candidate selection and the physical patch site (after native recurrent step eight) are fixed before opening a fresh causal study.

### Native interventions and matched controls

For an independent new host case, run both public hint conditions unchanged to produce `h8_high`, `h8_low`, and `d = h8_high - h8_low`. Evaluate source-preserving intervention on the **hint=1 recipient**:

    candidate: h8' = h8_high - (v·d)v, dose=|| (v·d)v ||_2.

The external paired donor h8_low is a **diagnostic full-source** reference. Every arm also gets zero edit, exact self-edit, orthogonal complement `(I-Pv)d` and **16 seeded unit-Haar random directions**, each signed using its own inner product with `d` and rescaled to the exact delivered physical candidate dose. Every edited state is propagated by the original unmodified GRU through steps 9–12, with the unchanged final head. Comparators must operate on the same root and case; sham/candidate physical norm parity tolerance `1e-6` is compulsory. Cases of dose below `1e-6` count as uninformative zeros in all arms and remain in the outcome denominator.

All outputs remain typed `grant`/`deny` *proposals*. The host-private executor checks each action against the registered true permission and blocks every unauthorized execution, while preserving **attempted** false grants for scientific scoring. False grants cost 2, missed legitimate grants cost 1. Measuring only false grants without missed permissions would reward always-deny behavior and fails the protocol.

### Prospective causal design

Freeze a new source tree and separate request before new entropy. Reserve **64 independent calibration roots, 128 independent development roots, and zero confirmation roots opened**, each with 24 original-generator cases. Candidate directions and sham vectors are trained/selected without those roots. The case is the repeated measure; the independent root is the statistical unit; 3 model initializers and all candidate/sham arms are nested within root. Random shams must be shared per root and seed across causal comparisons.

**Primary comparison:** on the `host_trait` model, define rootwise weighted-loss benefit per 24 cases of the candidate over the per-case median **16 equal-dose random-sham losses**, summed over cases then normalized by `2×24`, averaged across 3 fixed initializers. Require a mean benefit of at least `+0.02` normalized cost units per case and a whole-root 95% bootstrap lower bound greater than zero from 8,192 deterministic root resamples. Original same-model hint=1 recipient with zero edit is separately reported. The registered primary family size is one; comparisons with `reward_target`, `mixed`, `trait_cf`, full donor, self-edit, residual and zero edit are descriptive secondary diagnostics.

The causal mediation diagnostic records how much of the **native paired final-logit shift** each patch transfers toward hint=0 and compares with matched shams. At least 90% of scored cases require full/split native parity within `1e-6`, and candidate physical doses must exceed the `1e-6` resolution floor on at least 90% of cases. The calibration issuer-root weighted-loss headroom against the true-permission oracle must be at least 0.05 per case. Any failed model stability, intervention parity, headroom, sham-resolution or bootstrap gate yields a negative/limited development finding without removing cases or replacing roots.

This is a small synthetic recurrent causal-localization experiment. Its success would demonstrate a reproducible physical grader-response intervention in the registered GRU task. Generalization to large aligned language models, Anthropic/OpenAI training philosophy or genuine self-improving AI systems remains an external question.

## Source-only trace analysis

```bash
python -m unittest discover -s mechinterp/agp02/tests -v
python -m mechinterp.agp02.trace_audit \
  --traces /path/to/native_traces.jsonl.gz \
  --out /tmp/agp02-native-trace-audit.json
```

The original artifact is read-only input. The independent output must record its SHA-256 and the per-arm, per-seed, per-depth root and case denominators. All exploratory numbers retain their retrospective label.
