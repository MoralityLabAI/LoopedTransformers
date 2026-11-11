# Rapid gym ledger

Exploratory: dev = mathlib-c3 `val` states, Lean-certified. Not a registered test.

## By spec (seeds pooled)

| spec | runs | dev close@8 | dev+auto close@8 | val bits/byte | fresh |
|---|---|---|---|---|---|
| `["tlm-flat-XS", {}, 0.00025, 2000]` | 4 | 0.239 | 0.506 | 2.5404 | 256000 |
| `["tlm-flat-XS", {}, 0.001, 2000]` | 4 | 0.233 | 0.510 | 1.8285 | 256000 |
| `["tlm-trm-XS", {"trm_sup": 1}, 0.0005, 2000]` | 2 | 0.231 | 0.504 | 2.8834 | 256000 |
| `["tlm-trm-XS", {}, 0.0005, 2000]` | 1 | 0.227 | 0.504 | 2.8915 | 16000 |
| `["tlm-trm-XS", {"trm_sup": 4}, 0.0005, 2000]` | 1 | 0.227 | 0.504 | 2.8444 | 64000 |
| `["tlm-trm-XS", {"trm_T": 1, "trm_n": 2, "trm_sup": 4}, 0.0005, 2000]` | 1 | 0.227 | 0.504 | 2.8591 | 64000 |
| `["tlm-flat-XS", {}, 0.0005, 2000]` | 3 | 0.213 | 0.507 | 2.0996 | 256000 |
| `["tlm-loop-XS", {}, 0.0005, 2000]` | 1 | 0.210 | 0.513 | 2.0891 | 256000 |
| `["tlm-flat-XS", {}, 0.0005, 1000]` | 1 | 0.168 | 0.513 | 2.5665 | 128000 |

## Knob effects (mutant - parent, same round, budget and seed)

| knob | n | mean delta close@8 |
|---|---|---|
| lr=0.00025 | 1 | +0.0504 |
| lr=0.001 | 1 | +0.0000 |

## Rows

| round | run | dev close@1 | dev close@8 | +auto close@8 | auto close@8 | val bits/byte | train min |
|---|---|---|---|---|---|---|---|
| 1 | r01_flatXS_lr0.0005_k1_s17 | 0.160 | 0.168 | 0.513 | 0.521 | 2.5665 | 8.2 |
| 2 | r02_flatXS_lr0.0005_k2_s29 | 0.202 | 0.202 | 0.504 | 0.521 | 2.0855 | 19.0 |
| 2 | r02_flatXS_lr0.00025_k2_s29 | 0.227 | 0.252 | 0.504 | 0.521 | 2.5535 | 23.6 |
| 2 | r02_flatXS_lr0.001_k2_s29 | 0.151 | 0.202 | 0.504 | 0.521 | 1.8260 | 12.3 |
| 3 | r03_flatXS_lr0.00025_k2_s43 | 0.202 | 0.244 | 0.504 | 0.521 | 2.5372 | 12.8 |
| 3 | r03_flatXS_lr0.001_k2_s43 | 0.210 | 0.269 | 0.513 | 0.521 | 1.8203 | 12.6 |
| 3 | r03_flatXS_lr0.0005_k2_s43 | 0.210 | 0.252 | 0.513 | 0.521 | 2.0909 | 11.9 |
| 4 | r04_flatXS_lr0.001_k2_s59 | 0.202 | 0.244 | 0.513 | 0.521 | 1.8393 | 11.9 |
| 4 | r04_flatXS_lr0.0005_k2_s59 | 0.176 | 0.185 | 0.504 | 0.521 | 2.1225 | 11.9 |
| 4 | r04_flatXS_lr0.00025_k2_s59 | 0.210 | 0.252 | 0.513 | 0.521 | 2.5389 | 16.1 |
| 4 | r04_loopXS_lr0.0005_k2_s59 | 0.176 | 0.210 | 0.513 | 0.521 | 2.0891 | 1.4 |
| 4 | r04_trmXS_lr0.0005_k2_s59 | 0.227 | 0.227 | 0.504 | 0.521 | 2.8915 | 45.2 |
| 4 | r04_trmXS_sup4_lr0.0005_k2_s59 | 0.227 | 0.227 | 0.504 | 0.521 | 2.8444 | 43.5 |
| 4 | r04_trmXS_sup1_lr0.0005_k2_s59 | 0.227 | 0.235 | 0.504 | 0.521 | 2.9084 | 42.5 |
| 4 | r04_trmXS_T1_n2_sup4_lr0.0005_k2_s59 | 0.227 | 0.227 | 0.504 | 0.521 | 2.8591 | 11.6 |
| 5 | r05_flatXS_lr0.00025_k2_s71 | 0.202 | 0.210 | 0.504 | 0.521 | 2.5322 | 2.8 |
| 5 | r05_flatXS_lr0.001_k2_s71 | 0.151 | 0.218 | 0.513 | 0.521 | 1.8283 | 2.8 |
| 5 | r05_trmXS_sup1_lr0.0005_k2_s71 | 0.227 | 0.227 | 0.504 | 0.521 | 2.8584 | 42.3 |

## Not scored

- round 1: r01_loopXS_lr0.0005_k1_s17 (training did not finish)
- round 1: r01_trmXS_lr0.0005_k1_s17 (training did not finish)
- round 1: r01_trmXS_sup4_lr0.0005_k1_s17 (training did not finish)
- round 1: r01_trmXS_sup1_lr0.0005_k1_s17 (training did not finish)
- round 1: r01_trmXS_T1_n2_sup4_lr0.0005_k1_s17 (training did not finish)
