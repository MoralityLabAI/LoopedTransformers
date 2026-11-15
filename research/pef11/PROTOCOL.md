# PEF-11 | Fixed-noise and quantization frontier of ablation-angle recovery

Registration: 2026-11-15, **before any PEF-11 heldout perturbation was scored**. PEF-07 through PEF-10 results were already known; PEF-10 found ridge recovery with random tangents at 0.25°, while the native head stayed collapsed. This is a new conditional study and cannot retroactively promote earlier hypotheses.

## Inputs and trust

Fork PEF-10 PR #81 from exact head `159148716a0106e557ec7f543b9b45fb57a6f71a`. Reuse six untouched PEF-07 trained checkpoints, their sealed SHA256 training ledger `1233c3de26d64318a19b18e10abe4530cdc8e5f2dadc9027d8798a993c099714`, the PEF-09 frozen selection ledger `f206d88f5f2ea9703edc385d47bcf6fd6cd11a6cb4a3ece6edeae230dd734157`, and the PEF-10 *already probe-fitted* angle/ridge ledger `3ef6ce605540a5009f6896cf02a938c200bde23c7a5104d378aace7388430177`. Each original checkpoint's byte and parameter SHA256 is checked before scoring. PEF-10 ledger references must match PEF-07 and PEF-09; individual direction hashes and ridge hashes must be checked. Inputs are the original 160/48/48 context-disjoint groups with 16 focal masks each. PEF-11 uses no new training. Probe states define all scales and noise geometry. Heldout is used only after freezing condition selection and all deterministic noise seeds.

## Frozen condition grid

At visit h8, select `tangent in {pair01,pair23,pca2,random}`, `theta in degrees {0,0.25,1,4,16}`, and the **positive** rotation orientation for theta>0 (the zero angle is sign=0). This gives 20 conditions per model and 120 across 6 models. Each direction, its probe center, unit tangent and original *unperturbed post-ablation* fitted h12 and h14 ridge coefficients are read verbatim from PEF-10's frozen ledger, with equality/hash checks. Nothing is chosen based on PEF-10 heldout per-angle accuracy; this grid includes even prior failure cases. Independently measure an **unablated natural h8 state** under identical perturbations as a general capability control; use the original native h12 head and the original frozen PEF-09 h14 ridge for that control.

For each condition, write `g` (global PC), `u` (tangent), `mu` (h8 probe mean), `v=cos(theta)g+sin(theta)u` (removed direction), and `w=-sin(theta)g+cos(theta)u` (retained two-plane direction). All are unit and `v·w≈0`. The frozen scale is `s=sqrt(mean_probe[((h8-mu)·g)^2])` (strictly positive). A deterministic off-plane direction `o` is made from one standard-normal FP64 vector with seed `110110+seed+10000*[task=="threshold"]+100*tangent_index+angle_index`, orthogonalized against g and u; regenerate once using seed+100000 if degenerate. `o` and w have equal norm. This off-plane control has equal injected L2 perturbation norm on every example.

## Perturbations before natural replay

Start with the exact ungated PEF-10 ablated h8 state `a=h8-((h8-mu)·v)v`; for the natural control start with the unablated h8. The same transformation applies to **all 16 focal patterns**, regardless of class or bit identities. Perturbations are applied at h8 **before** re-running the original recurrent steps 9–12 and h13/14 continuation (0,1).

1. **Additive Gaussian, paired controls:** fixed `relative sigma` in `{0,0.001,0.003,0.01,0.03,0.1}`; absolute sigma = relative sigma × s. Each nonzero level has three independent replicates r=0,1,2. Generate a 768-long FP64 standard-normal vector `z` with CPU seed `111100000+seed*10000+1000*[task=="threshold"]+100*tangent_index+10*angle_index+replicate`. Reuse this *identical per-example z* for the transverse disturbance `a+sigma*z*w`, matched off-plane disturbance `a+sigma*z*o`, and natural-state control `h8+sigma*z*w`. The single zero-noise reference is unmodified. Report each replicate and the mean, and the realized norm of perturbations.
2. **Retained-coordinate quantization:** step sizes relative to s `{0,0.003,0.01,0.03,0.1}`. Quantize exactly the retained transverse scalar `q=(a-mu)·w` to `step×round(q/step)` (PyTorch nearest-even), leaving other h8 coordinates unchanged; zero step is identity. This is **one-dimensional affine coordinate quantization**, not a full 32-channel hardware format. Run the matched natural-state control using the same frozen w, mu and step.
3. **Per-channel quantization:** same five relative steps, with each channel's absolute step `relative × clamp(probe_h8_std[c], min=s/1000)` (population std), each centered on frozen mu[c]. Apply `Q(h8[c])=mu[c]+step[c]×round((h8[c]-mu[c])/step[c])` to the entire **post-ablation** h8 state. The matched natural state uses the same steps and centers. This is a second, explicitly axis-dependent hardware-motivated comparison. Zero step is exact identity.

For all perturbed branches, score the original PEF-10 h12/h14 ridge decoders **without refitting on noise or quantized states**. Also report original native h12, previously frozen h14 decoder, natural-control native h12 and natural-control frozen h14. Every decoded score is evaluated against all 768 heldout labels, including per-class recall and error counts for all 16 focal patterns. No output-head changes, no test-time tuning.

## Registered primary decisions

The *random-tangent* case at 0.25° is the small-angle reference; the same tangent at 4° is the larger-angle comparison. Both use the positive sign. For each coalition seed:

- **Fragility at small angle:** clean h12 newly fitted ridge correct >=720/768, AND at relative Gaussian transverse sigma 0.01 the **mean of three noisy heldout correct counts** <= clean - 96, AND natural-control native h12 at sigma 0.01 remains >=720/768 on average. Quantization at retained-coordinate relative step 0.03 must produce at least 96 fewer correct than clean. Noisy evidence uses the fixed per-sample standard-normal draws; the natural control is an independent readout.
- **Robustness at larger angle:** at 4° and the same random tangent, Gaussian transverse sigma 0.01 mean correct >=720/768 and retained-coordinate quantization at step 0.03 correct >=720/768.

Count the coalition seeds passing both. `criterion_met` requires >=2 of 3. Every arm, including threshold controls, must be reported. A separate directional robustness diagnostic compares transverse against energy-matched off-plane Gaussian under the **same z**, reporting the paired accuracy difference at sigma 0.01 for each seed/angle and counts of seeds with greater transverse damage. A separate noise threshold table lists for each angle the largest tested relative sigma and quantization step meeting 720/768, with no interpolation or posthoc threshold tuning.

## Analysis limits

Gaussian error depends on only three fixed random replicas and is descriptive; it is not a confidence interval over possible architectures. Center/PC/tangent directions and readout are frozen from the earlier PEF-10 probe ledger. Quantized/transverse readout is fixed to the noise-free probe training distribution; the experiment measures *deployment without noisy retraining*. The off-plane direction is matched on per-example perturbation norm, not on model Jacobian sensitivity or baseline hidden variance. Per-channel quantization tests state representations at h8, not an actual low-bit hardware kernel. Fixed native h12 results help distinguish accessibility to a new decoder from functional recovery by the original head. These finite synthetic Boolean tasks say nothing directly about Lean, language models or safety behavior.

## Reproduce

```sh
python -m pytest -q research/pef11/test_pef11.py
python -m research.pef11.pef11 --evidence /path/to/registered \
  --selection-ledger /path/to/frozen_selection_ledger.json \
  --angle-ledger /path/to/frozen_angle_and_ridge_ledger.json --out /tmp/pef11
```

CI unit tests need no external checkpoints. Local Python 3.13 CPU findings and hosted Python 3.12 verification must be labeled separately.