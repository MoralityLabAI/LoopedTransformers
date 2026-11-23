# LBF-13: budget-conditioned rewrite utility results
Date 2026-11-24. Disposition: ENGINEERING_PASS; UTILITY_AND_MEMORY_SCREENS_MISSED. Development only.

[Registration](LBF_13_BUDGET_FRESCO_REGISTRATION.md) · [Passing CPU/Lean CI](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45282924682) · Tested head 689713c890acf110446bb812f9af7dd1861c20f5.
[Evidence archive](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45282924682/artifacts/14870887173), SHA-256 63136b1ce15e58e62a2b4839afc1c8f90cc89a55500954cc70efe83a1f6dfa00, expires 2026-12-07. Raw boundary tensor ledger SHA-256 fdc58335d301031821db89afb0199e767d3e162e949c7f4946606e5f48dbdc9c.

## Custody
All 15 engineering tests passed. 32 certified training references, 320 witness frames, 1,692 forced candidate costs, 362 censored at 128, 224 independent shorter-budget checks and 24 axiom-free forced continuation audit certificates. 92,369 train oracle successor checks including prefix audits. Both seeds match initialization and sampled root/frame/budget schedules across the witness and utility arms. 193 axiom-free policy success certificates and all 24 calibration reference certificates. Ordinary/OOD outcomes sealed. 184 distinct decision probes with full continuous tensors and same-decision zero/roll controls. No live Lean tactic authority claimed.

| Work budget | Mixed menus | All positive | All negative | Positive candidates |
|---:|---:|---:|---:|---:|
|1|39|0|281|39|
|4|96|0|224|140|
|8|134|21|165|377|
|16|147|57|116|652|
|32|133|99|88|864|
|64|109|154|57|1104|
|128|86|207|27|1330|

## Frozen search, 128 successor cap, path cap 16
| Policy | Certified closures | Work | Expanded | Model calls |
|---|---:|---:|---:|---:|
| A* |11/24|2031|427|0|
| Bidirectional BFS |17/24|1896|403|0|
| Uniform frontier, five streams |55/120|10963|2319|0|
| Witness frontier |22/48|4338|928|928|
| Utility frontier |22/48|4350|922|922|
| Utility zero |22/48|4350|922|922|
| Utility roll |22/48|4350|922|922|
| Utility sibling-only |22/48|4106|853|853|

Both seeds close 11/24 under every learned arm. Per-root utility normal contrasts versus A*, uniform, witness, sibling, zero and roll: 24 ties, 0 pp, interval [0,0]. Versus bidir: -25 pp, descriptive paired 95% interval [-41.67,-8.33], 0 wins/6 losses/18 ties. Both screens missed. Utility sibling ordering saves 244 considered successors versus utility frontier over 48 episodes; closure remains unchanged. Exact bounded restart labels do not measure marginal utility of an already populated search frontier. Full bank tensor archive supports future diagnostic analysis; this pilot alone does not isolate which part of the architecture causes the null search effect.

## Research decision
Move supervision to decisions native to bidirectional search. Strong symbolic bidirectional search should remain a control. Use the public pair of frontier layers, costs and meeting evidence, with forced direction choices followed by a fixed continuation policy. A separate fresh suite is required. No further tuning on this exposed cohort and no architecture promotion.
