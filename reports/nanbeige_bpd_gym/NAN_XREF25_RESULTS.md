# Audited CAL-only SAE towers and learned component cross-references

Both NF4 and BF16 passed the separate strict XREF25 independent audit at fitted depths1 and2. Each cell rechecks all1706 cached vectors,873472 sparse-code values,129024 correlation entries and18 physical geometry cases. Original XREF19/XREF24 failures remain failed; the decoder orientation and geometry dtype are repaired only in the separately registered XREF25 condition. No tolerance was widened, worker/model replayed, or grade added.

Some registered strata reconstruct worse with the trained SAE than its random-initial control: negative improvement appears in12 of the48 backend/depth/dataset/layer range rows. Those rows repeat dictionaries across depths. Across all cached-factor strata,34528 of516096 registered correlation entries are defined; the rest remain null under the original root-count/constant-coordinate rules.

The same eight fixed CAL-only SAE dictionaries (four layers per backend,128 features/top4) are reused across fitted depths and datasets. Learned components/controllers had16 real CAL-only optimizer steps at each depth. Cached applications are observational: component reconstruction and centered routed energy do not equal the actual sphere-rescaled online write or establish task mediation.

Unit:reused source root/family on one checkpoint. GYM05 has89 root groups/644 phase vectors, COMP14 has24 roots/480 vectors, CTX08 five families/582 vectors. These counts repeat across fits and layers; they are not pooled independent population replications.

## SAE reconstruction across every registered stratum

Each range below is the minimum to maximum of the EXISTING equal-root group means across all registered CAL/DEV, loop and task/control strata in that row. It is not an overall root mean, uncertainty interval or new hypothesis test. Strata remain separate in the scalar CSV; all128-feature activity vectors and128x8 correlations remain in the full source JSON. Negative improvements are retained.

| Backend | Fit depth | Dataset | Layer | Strata | Phase vectors | Trained normalized MSE range | Random-initial MSE range | Improvement range |
|---|---:|---|---:|---:|---:|---|---|---|
| bf16 | 1 | gym05 | 5 | 28 | 644 | 0.012323 to 16.841 | 0.78231 to 17.036 | 0.088229 to 1.1779 |
| bf16 | 1 | gym05 | 11 | 28 | 644 | 0.05624 to 10.55 | 0.7726 to 10.811 | -0.16637 to 1.0449 |
| bf16 | 1 | gym05 | 17 | 28 | 644 | 0.14063 to 8.2387 | 0.7904 to 8.3724 | 0.10181 to 0.97615 |
| bf16 | 1 | gym05 | 21 | 28 | 644 | 0.09507 to 9.5535 | 0.75218 to 9.9613 | 0.25355 to 0.99839 |
| bf16 | 1 | comp14 | 5 | 20 | 480 | 0.021816 to 13.533 | 0.79069 to 13.687 | 0.1543 to 1.0535 |
| bf16 | 1 | comp14 | 11 | 20 | 480 | 0.086896 to 9.4527 | 0.77741 to 10.062 | 0.39762 to 1.2599 |
| bf16 | 1 | comp14 | 17 | 20 | 480 | 0.26173 to 6.1633 | 0.86969 to 6.2958 | 0.13253 to 1.1646 |
| bf16 | 1 | comp14 | 21 | 20 | 480 | 0.16329 to 4.3218 | 0.80424 to 4.9118 | 0.51573 to 1.1318 |
| bf16 | 1 | ctx08 | 5 | 78 | 582 | 2.3758 to 8.0444 | 3.1942 to 8.4405 | 0.39614 to 1.1042 |
| bf16 | 1 | ctx08 | 11 | 78 | 582 | 2.3619 to 5.0743 | 2.9174 to 5.8982 | 0.34037 to 0.93245 |
| bf16 | 1 | ctx08 | 17 | 78 | 582 | 1.352 to 4.6415 | 1.8191 to 4.9709 | 0.29668 to 1.0111 |
| bf16 | 1 | ctx08 | 21 | 78 | 582 | 1.6287 to 3.4922 | 2.2463 to 4.7912 | 0.56628 to 1.5278 |
| bf16 | 2 | gym05 | 5 | 28 | 644 | 0.012323 to 16.841 | 0.78231 to 17.036 | 0.088229 to 1.1779 |
| bf16 | 2 | gym05 | 11 | 28 | 644 | 0.05624 to 10.55 | 0.7726 to 10.811 | -0.16637 to 1.0449 |
| bf16 | 2 | gym05 | 17 | 28 | 644 | 0.14063 to 8.2387 | 0.7904 to 8.3724 | 0.10181 to 0.97615 |
| bf16 | 2 | gym05 | 21 | 28 | 644 | 0.09507 to 9.5535 | 0.75218 to 9.9613 | 0.25355 to 0.99839 |
| bf16 | 2 | comp14 | 5 | 20 | 480 | 0.021816 to 13.533 | 0.79069 to 13.687 | 0.1543 to 1.0535 |
| bf16 | 2 | comp14 | 11 | 20 | 480 | 0.086896 to 9.4527 | 0.77741 to 10.062 | 0.39762 to 1.2599 |
| bf16 | 2 | comp14 | 17 | 20 | 480 | 0.26173 to 6.1633 | 0.86969 to 6.2958 | 0.13253 to 1.1646 |
| bf16 | 2 | comp14 | 21 | 20 | 480 | 0.16329 to 4.3218 | 0.80424 to 4.9118 | 0.51573 to 1.1318 |
| bf16 | 2 | ctx08 | 5 | 78 | 582 | 2.3758 to 8.0444 | 3.1942 to 8.4405 | 0.39614 to 1.1042 |
| bf16 | 2 | ctx08 | 11 | 78 | 582 | 2.3619 to 5.0743 | 2.9174 to 5.8982 | 0.34037 to 0.93245 |
| bf16 | 2 | ctx08 | 17 | 78 | 582 | 1.352 to 4.6415 | 1.8191 to 4.9709 | 0.29668 to 1.0111 |
| bf16 | 2 | ctx08 | 21 | 78 | 582 | 1.6287 to 3.4922 | 2.2463 to 4.7912 | 0.56628 to 1.5278 |
| nf4 | 1 | gym05 | 5 | 28 | 644 | 0.012964 to 16.619 | 0.77778 to 16.829 | -0.16029 to 1.0118 |
| nf4 | 1 | gym05 | 11 | 28 | 644 | 0.065722 to 11.653 | 0.79556 to 11.898 | -0.15147 to 1.0092 |
| nf4 | 1 | gym05 | 17 | 28 | 644 | 0.1454 to 8.3627 | 0.79898 to 8.3412 | -0.025991 to 0.9177 |
| nf4 | 1 | gym05 | 21 | 28 | 644 | 0.10022 to 8.2701 | 0.74889 to 8.5202 | 0.16384 to 0.9764 |
| nf4 | 1 | comp14 | 5 | 20 | 480 | 0.019147 to 13.008 | 0.78643 to 12.904 | -0.1038 to 1.0777 |
| nf4 | 1 | comp14 | 11 | 20 | 480 | 0.09438 to 10.325 | 0.79743 to 10.429 | 0.10441 to 1.264 |
| nf4 | 1 | comp14 | 17 | 20 | 480 | 0.24941 to 6.0354 | 0.8582 to 6.1163 | 0.080954 to 1.0724 |
| nf4 | 1 | comp14 | 21 | 20 | 480 | 0.14723 to 4.0076 | 0.7727 to 4.4116 | 0.40396 to 1.0077 |
| nf4 | 1 | ctx08 | 5 | 78 | 582 | 2.834 to 8.6836 | 3.617 to 8.5584 | -0.20528 to 0.82536 |
| nf4 | 1 | ctx08 | 11 | 78 | 582 | 2.5238 to 5.6959 | 3.0075 to 6.4425 | 0.27884 to 0.86057 |
| nf4 | 1 | ctx08 | 17 | 78 | 582 | 1.3696 to 4.3452 | 1.7444 to 4.8617 | 0.31064 to 0.98703 |
| nf4 | 1 | ctx08 | 21 | 78 | 582 | 1.4095 to 2.8686 | 1.8408 to 3.7994 | 0.4313 to 1.1781 |
| nf4 | 2 | gym05 | 5 | 28 | 644 | 0.012964 to 16.619 | 0.77778 to 16.829 | -0.16029 to 1.0118 |
| nf4 | 2 | gym05 | 11 | 28 | 644 | 0.065722 to 11.653 | 0.79556 to 11.898 | -0.15147 to 1.0092 |
| nf4 | 2 | gym05 | 17 | 28 | 644 | 0.1454 to 8.3627 | 0.79898 to 8.3412 | -0.025991 to 0.9177 |
| nf4 | 2 | gym05 | 21 | 28 | 644 | 0.10022 to 8.2701 | 0.74889 to 8.5202 | 0.16384 to 0.9764 |
| nf4 | 2 | comp14 | 5 | 20 | 480 | 0.019147 to 13.008 | 0.78643 to 12.904 | -0.1038 to 1.0777 |
| nf4 | 2 | comp14 | 11 | 20 | 480 | 0.09438 to 10.325 | 0.79743 to 10.429 | 0.10441 to 1.264 |
| nf4 | 2 | comp14 | 17 | 20 | 480 | 0.24941 to 6.0354 | 0.8582 to 6.1163 | 0.080954 to 1.0724 |
| nf4 | 2 | comp14 | 21 | 20 | 480 | 0.14723 to 4.0076 | 0.7727 to 4.4116 | 0.40396 to 1.0077 |
| nf4 | 2 | ctx08 | 5 | 78 | 582 | 2.834 to 8.6836 | 3.617 to 8.5584 | -0.20528 to 0.82536 |
| nf4 | 2 | ctx08 | 11 | 78 | 582 | 2.5238 to 5.6959 | 3.0075 to 6.4425 | 0.27884 to 0.86057 |
| nf4 | 2 | ctx08 | 17 | 78 | 582 | 1.3696 to 4.3452 | 1.7444 to 4.8617 | 0.31064 to 0.98703 |
| nf4 | 2 | ctx08 | 21 | 78 | 582 | 1.4095 to 2.8686 | 1.8408 to 3.7994 | 0.4313 to 1.1781 |

## Cached component reconstruction and undefined correlations

Ranges again retain all existing CAL/DEV/loop/control strata. Error is pure component prediction versus native physical module output at the cached clean input. Haar factors have matched input-column/output-row norms. Correlation counts are matrix entries, not independent tests: fewer than3 roots or constant coordinates stay null. T1 applications at loop2 are outside fit depth.

| Backend | Fit depth | Dataset | Strata | Learned relative L2 range | Initial SVD range | Matched-norm Haar range | Defined / all correlation entries | Out-of-depth strata |
|---|---:|---|---:|---|---|---|---:|---:|
| bf16 | 1 | gym05 | 28 | 0.13222 to 0.4285 | 0.13691 to 0.42978 | 1.0006 to 1.0034 | 1808 / 28672 | 14 |
| bf16 | 1 | comp14 | 20 | 0.13251 to 0.40492 | 0.13729 to 0.40647 | 1.0006 to 1.0019 | 1888 / 20480 | 10 |
| bf16 | 1 | ctx08 | 78 | 0.13375 to 0.40344 | 0.13863 to 0.40511 | 1.0005 to 1.0019 | 4136 / 79872 | 39 |
| bf16 | 2 | gym05 | 28 | 0.1371 to 0.42886 | 0.13691 to 0.42978 | 0.99963 to 1.004 | 1808 / 28672 | 0 |
| bf16 | 2 | comp14 | 20 | 0.13756 to 0.40493 | 0.13729 to 0.40647 | 0.99975 to 1.0014 | 1888 / 20480 | 0 |
| bf16 | 2 | ctx08 | 78 | 0.13885 to 0.40373 | 0.13863 to 0.40511 | 0.99969 to 1.0004 | 4136 / 79872 | 0 |
| nf4 | 1 | gym05 | 28 | 0.13118 to 0.4124 | 0.13566 to 0.41547 | 1.0006 to 1.0031 | 1824 / 28672 | 14 |
| nf4 | 1 | comp14 | 20 | 0.13195 to 0.39548 | 0.1366 to 0.39873 | 1.0006 to 1.0017 | 2184 / 20480 | 10 |
| nf4 | 1 | ctx08 | 78 | 0.13197 to 0.39077 | 0.1366 to 0.39429 | 1.0006 to 1.0022 | 5424 / 79872 | 39 |
| nf4 | 2 | gym05 | 28 | 0.1361 to 0.41411 | 0.13566 to 0.41547 | 0.99959 to 1.0037 | 1824 / 28672 | 0 |
| nf4 | 2 | comp14 | 20 | 0.1371 to 0.39719 | 0.1366 to 0.39873 | 0.99972 to 1.001 | 2184 / 20480 | 0 |
| nf4 | 2 | ctx08 | 78 | 0.13705 to 0.39281 | 0.1366 to 0.39429 | 0.99962 to 1.0005 | 5424 / 79872 | 0 |

## Fixed physical weight geometry

Only the layer21 down_proj output-to-residual identity-coordinate bridge is used. The same weight matrices repeat across three datasets/two application loops; all72 registered cases map to12 labelled backend/depth/basis groups. Their numerical arrays have10 distinct fingerprints: initial SVD controls also repeat across fitted depths. Every matrix entry was checked equal before grouping repetitions. Mean absolute cosines are explicitly post hoc display summaries over ALL128x8 entries, without selected features/components, significance tests or causal interpretation. The legacy unique-geometry CSV name refers to unique labelled coefficient keys, not independent numerical bases. Signs and every sensitivity remain in that CSV. Independently fitted backend SAE indices are not subtracted; the finite diagonal/permutation gauge check does not identify arbitrary rotated component directions.

| Backend | Fit depth | Basis | Repeated registered cases | Mean absolute encoder cosine | Mean absolute decoder cosine | Zero encoder / decoder / output rows |
|---|---:|---|---:|---:|---:|---:|
| bf16 | 1 | initial_SVD_not_trained | 6 | 0.0141995 | 0.160957 | 0 / 0 / 0 |
| bf16 | 1 | learned_VPD_T1 | 6 | 0.0141884 | 0.161217 | 0 / 0 / 0 |
| bf16 | 1 | matched_norm_Haar_not_trained | 6 | 0.0143852 | 0.0138096 | 0 / 0 / 0 |
| bf16 | 2 | initial_SVD_not_trained | 6 | 0.0141995 | 0.160957 | 0 / 0 / 0 |
| bf16 | 2 | learned_BPD_T2 | 6 | 0.0141965 | 0.160928 | 0 / 0 / 0 |
| bf16 | 2 | matched_norm_Haar_not_trained | 6 | 0.0146438 | 0.0154826 | 0 / 0 / 0 |
| nf4 | 1 | initial_SVD_not_trained | 6 | 0.013603 | 0.165745 | 0 / 0 / 0 |
| nf4 | 1 | learned_VPD_T1 | 6 | 0.0136064 | 0.166045 | 0 / 0 / 0 |
| nf4 | 1 | matched_norm_Haar_not_trained | 6 | 0.0146906 | 0.0140881 | 0 / 0 / 0 |
| nf4 | 2 | initial_SVD_not_trained | 6 | 0.013603 | 0.165745 | 0 / 0 / 0 |
| nf4 | 2 | learned_BPD_T2 | 6 | 0.0136091 | 0.165642 | 0 / 0 / 0 |
| nf4 | 2 | matched_norm_Haar_not_trained | 6 | 0.0142969 | 0.015651 | 0 / 0 / 0 |

## Provenance and limits

Immutable source: `fefa31be596ac37a0bdf5e2000e8a5efd79c23f4`, SHA-256 `207bc4acbd6e583bda8ff1ed0bfe1f85053883e18a172c36a1a1075516e7a0fa`. Fit checkpoint/16-step qualification is cross-bound to `f419a7a1a75cb16d4cb6c9748a2927c249244dcd`.

All outcomes and prior repair failures were exposed before this presentation. Physical neural dose:none in this cached analysis. Scientific causal floor, matched-sham floor and causal power remain uncalibrated; planned/actual formal comparisons0, verdict REDESIGN_OR_PILOT_ONLY. More untouched roots, a prospectively calibrated causal contrast and an independent sealed readout would be needed for confirmation. No original failed algebraic/FIT07 gate is changed, and no donor transfer or equivalence is claimed. The168-case test stays sealed; the continuous goal remains active.

[Full immutable audit JSON](https://github.com/MoralityLabAI/LoopedTransformers/blob/fefa31be596ac37a0bdf5e2000e8a5efd79c23f4/reports/nanbeige_bpd_gym/nan_xref25_xref19_progress.json) · [All scalar strata](nan_xref25_scalar_strata.csv) · [All unique weight entries](nan_xref25_unique_weight_geometry.csv) · [All72 geometry-case bindings](nan_xref25_geometry_coverage.csv) · [Presentation receipt](nan_xref25_results_receipt.json) · [Separate repair](NAN_XREF25_XREF19_PROGRESS.md) · [Online score/dose comparison](NAN_FIT16_IO18_RESULTS.md)
