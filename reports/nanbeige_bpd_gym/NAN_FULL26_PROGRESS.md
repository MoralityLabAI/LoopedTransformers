# NAN-FULL-26: full development trajectories and batched A/B

Frozen scope: 1,549 baseline contexts per backend: GYM05 322, COMP14 240, CTX08 291, CTX21 624, STORY23 72. After full baseline generation: 216 clean/intervention pairs per backend (all 24 story worlds and 48 focused competence worlds × three routes).

Native full-prefix generation, two recurrent loops, full vocabulary greedy sampling. Native clean row A and edited row B share one real decoder batch; one model is resident. Capacity-only preflight governs batch two versus one for baseline contexts. A/B requires batch two; no serial substitution. All 40 physical vectors are saved at every generated prefix, with exact BF16 storage. Fixed SAE/random controls and independent BPD mask/write audits stream on one CPU thread.

This is a new trace-prompt and batched execution condition. Earlier one-token scores are not its matched control. Cross-precision accuracy differences include batch-shape differences wherever capacity admission differs; geometry requires identical histories and shapes. Generated prefixes without a closed audited episode retain unknown grades and are not replayed. All 168 original test cases remain sealed. Repeated roots, tokens, routes and features are correlated measures. No transfer, equivalence or causal confirmation is claimed; effect floors, dose/sham calibration and power remain uncalibrated. Verdict: REDESIGN_OR_PILOT_ONLY.

NF4: unrun_or_live_no_terminal_claim. Closed baseline unknown/1549; A/B unknown/216.
BF16: unrun_or_live_no_terminal_claim. Closed baseline unknown/1549; A/B unknown/216.

|Backend|Stage|Split|Lane|Mode|Closed|Roots|Format|Correct|Prefixes|
|---|---|---|---|---|---:|---:|---:|---:|---:|
