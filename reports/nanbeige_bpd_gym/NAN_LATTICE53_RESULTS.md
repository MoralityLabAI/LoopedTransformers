# LATTICE53 physical limits of low-dose BPD routing edits

At 0.25% residual L2, exact rational certificates excluded **248 of the
250** learned or CAL-static directions that failed DOSE51's angular gate.
For each certificate, every finite BF16 displacement within the registered
3% radius band has direction cosine strictly below 0.98. The bound
applies to arbitrary reachable edits, including nonlinear and
coordinate-selective searches.

| Precision | Signed targets | Ray passed the angular gate | Certified impossible | Unresolved angle failures | Roots with all eight targets impossible |
| --- | ---: | ---: | ---: | ---: | ---: |
| NF4 | 128 | 0 | 128 | 0 | 16/16 |
| BF16 | 128 | 6 | 120 | 2 | 15/16 |

All 64 learned and 64 static NF4 targets were certified. In the BF16
lane, 60/64 targets were certified for each arm. The two unresolved
failures were learned targets. Both sets of edits are stored in BF16
residual arithmetic. NF4 and BF16 refer to the inference lanes that
produced the clean residuals and learned banks.

The checker re-created every physical edit in the hash-bound DOSE51
archive before computing the bound. A float64 grid picked a candidate
scale. An exact rational pass independently bracketed each probe
coordinate by adjacent BF16 values and proved the strict inequality at
both endpoints of the allowed radius interval. Feasible and infeasible
two-coordinate fixtures, including signed zero and asymmetric spacing
around ±1, exercised the implementation.

For this route family, another search at the same native boundary and
dose cannot recover coverage on the certified rows. A follow-up
comparison can calibrate rounded direction feasibility on CAL before
spending GPU time, then apply the current-head and matched-sham gates
to the resulting intervention. The separately studied SPECTRAL45
margin-null directions are a candidate calibration family.

The independent unit is the source root across 16 exposed DEV roots.
These numerical certificates concern finite reachability. Future
behavior and adaptive-route utility remain unmeasured by this study.
The host audit used about 60.0 MiB peak sampled private RAM and zero
pretrained forwards. [Proof and protocol](../../experiments/nan_lattice53/PROTOCOL.md)
· [Aggregate](nan_lattice53_results.json)
