# PEF-09 | Ungated, label-free interaction-direction ablation

Registration: 2026-11-15. This file is committed as a standalone prerequisite before any PEF-09 heldout ablation run. PEF-07 and PEF-08 results were already known and informed this design. This is a **new conditional intervention study**, not retrospective preregistration for the earlier results.

## Parent, frozen artifacts and splits

Based on PEF-08 draft PR #75 head `cf708b432c07515e6ba7e28ce5000347df14859f`, which itself extends PEF-07 PR #67 (head `d9c2087cbe37f4787e3e56ef9d64ab2eb148f214`). Use the original PEF-07 450-step six-model checkpoints from the local archive `PEF08_frozen_coalition_causal_evidence_20261115.zip`, SHA256 `713520836f49e34d07a36c9ca11200b2cdf6afac8fcfa4c020b20d94b3e7322b`. The original sealed ledger SHA256 is `1233c3de26d64318a19b18e10abe4530cdc8e5f2dadc9027d8798a993c099714`; the 12-bit context split manifest SHA256 is `849d039e47dec472b0b4835d24928e9a547e262bf84fd7dfed16741df9911ac1`.

All six frozen checkpoint byte SHA256s, parameter digests, metadata and source-manifest hashes must match their original PEF-07 evidence ledger. Source contexts remain 160/48/48 for training, probe, heldout. The *probe* 48 context groups (768 records) are exclusively used for direction selection and for fitting the fixed h14 task ridge decoder. The *heldout* 48 context groups (768 records) are exclusively used for the final ablation and outcome scoring. No training on heldout or further edits to checkpoint weights.

## Feature selection independent of target readout

For each frozen model seed in {103,131,167}, task in {coalition, threshold}, and recurrent visit t in {4,8,12}, calculate natural hidden memory `h_t` for all 16 focal assignments in each probe context.

For pair p in {(0,1),(2,3)}, and each of the 4 assignments b to the *other* two focal bits, compute the four-corner vector `d_{p,b}(context,t) = h_t(b+e_i+e_j)-h_t(b+e_i)-h_t(b+e_j)+h_t(b)`. Stack the resulting 48*4=192 vectors into matrix `D_p(t)` with shape [192,32]. Take its **uncentered**, rank-1 leading right singular vector `v_p(t)`. Fix sign so the maximal-absolute coordinate is positive, ties by index. If largest singular value <1e-10, record degeneracy and use canonical coordinate e0; count this as failure for pair selectivity. This direction uses only hidden activations and the two predeclared bit-pair identities, **not** the classification head weights, model labels, heldout examples, or PEF-08's previously selected top-8 channels.

Define `mu_t` as the arithmetic mean of the 768 natural probe states at visit t. Primary intervention **ungated**: for every heldout example regardless of focal bits, `h'_t = h_t - ((h_t - mu_t) dot v_p(t)) v_p(t)`. The same `mu_t` and `v_p(t)` apply to all 16 focal masks, including masks with neither bit active. Continue the frozen recurrent operator on natural input bits to h12; at h14 continue with the previously registered bits (0,1). The original trained native h12 logits and the once-probe-fitted h14 ridge readout are unchanged.

## Independent comparators

(1) **Eight energy-matched random directions** for each task/seed/visit/pair: generate 256 32-dimensional i.i.d. standard-normal unit vectors using `torch.Generator(device="cpu").manual_seed(9009+1000*seed+10*t+2*pair_index+100000*[task=="threshold"])` and fixed FP64 tensors. For each candidate u compute probe projection energy `E[(h_t-mu_t) dot u]^2`, and choose the eight with the closest absolute log-energy difference to the target v; tie by candidate index. Do not rescale either ablation. Record each selected random vector, selected indices, and energy ratio.

(2) **Cross-pair control**: use the other intended pair's independently extracted v, which has equal one-dimensional rank but a different bit-pair provenance.

(3) **Global-PC control**: take the top right singular vector of the centered 768x32 natural probe state matrix at visit t, with the same sign convention. This is an ungated, label-free generic high-variance direction.

(4) **Identity and full-space algebraic sanity checks** in unit tests. Zero-rank projection must leave all scores bit-identical. Full orthonormal basis projection must send every state to `mu_t`. Projectors must be symmetric, idempotent, and independent of focal masks.

All feature directions and null selections are determined on the probe contexts **before heldout inference**. The causal analysis can read heldout natural states only after the directions, random vectors, and h14 probe ridge are finalized, with hashes of those frozen selections recorded.

## Outcomes and preregistered analysis

For each of 6 models x 3 visit times x 2 target pairs x 11 conditions (target, other-pair, global-PC, eight matched-random), score on all 768 heldout histories:

- native h12 score correct /768, balanced class accuracy, errors by all 16 focal masks, mean absolute native logit change;
- four-corner native-logit interaction magnitude `M_p`, averaged over **all four other-bit bases and all 48 contexts**, for both intended pairs; calculate `A_p=1-M_p^{patched}/max(M_p^{baseline},1e-10)` and `A_other` with the same definition;
- directional selectivity score `S_p = A_p - A_other`, which must be reported alongside both raw interaction magnitudes and absolute error counts;
- fixed h14 ridge correct/768 after natural continuation (0,1), fitted once on unpatched probe states; h14 pair-interaction effects recorded with the h14 ridge margin scale;
- per-run heldout projection energy, probe-selected energy, 32-vector direction and SHA256 receipt, for all controls.

The primary diagnostic is the **heldout loop-8 selectivity excess** `S(target v_p) - median[S(eight matched-random directions)]`. Declare a seed a diagnostic success only when **both intended pairs** have excess strictly greater than +0.15 and at least 16/768 native h12 classification errors after the target-direction ablation. Count successful coalition seeds among three; the registered 2/3 cohort decision is `criterion_met` at >=2. Report loop 4 and 12 independently, full per-seed results and threshold-control results, even if all criteria fail. Criterion is diagnostic, because the PEF-07 models and tasks were already known and the direction choice may recover generic interaction geometry.

Because full-space projections can have very large effects, the global-PC control is a perturbation magnitude comparator, not a pair-matched adversary; report perturbation energy. The eight random controls are matched on probe **energy**, not automatically on downstream output sensitivity. Since directions are learned from the probe group, heldout tests still assess transfer across context IDs in this fixed synthetic task family, not transfer across unseen operators.

## Scientific limitations

The target direction is extracted from the known target pair's four-corner hidden activation, so it is an *interaction-conditioned feature selection* rather than a completely blind unsupervised discovery. Crucially, applying its projector involves no bit-pair condition and no classifier-head information. This tests whether an independently chosen, global linear direction matters to the frozen model's actual computation. Rank-one projection can perturb unrelated signal; its effect is neither a unique circuit proof nor sufficient to show that ablated information disappears from h14 representation. All results remain about the synthetic 12-bit tasks. No claims about SAEs, Lean reasoning improvement, or pretrained models without separate experiments.

Reproduction: `python -m pytest -q research/pef09/test_pef09.py` and `python -m research.pef09.pef09 --evidence /path/to/registered --out /tmp/pef09`. Requiring archived weights for full replay is an explicit external-artifact dependency. CI for source-only invariants may run separately without original checkpoints.
