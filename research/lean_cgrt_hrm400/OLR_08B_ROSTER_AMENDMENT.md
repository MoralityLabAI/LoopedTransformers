# OLR-08B prospective roster amendment

2026-11-26, written before the OLR-08B calibration freeze and before any OLR-08B prospective outcome.

## Reason

A parallel OLR-08B implementation, registered on research/olr-obligation-ledger-20261123 at
commit 9612b6fa, froze its own registry at 21:23 local time. That was after this branch committed
its outcome-blind roster at 21:01 (6a2c83ce). Its run on this machine (C:\olr08b\full, contract schema
olr08b.cpu.v1) registered calibration roots in 7 of the 64 files in
[olr08b/calibration_roster.json](olr08b/calibration_roster.json) and has already collected native
outcomes on them:

| File | Registered roots | Menu outcomes | S1 outcomes | S2 outcomes |
| --- | --- | --- | --- | --- |
| src/Init/Data/Array/Bootstrap.lean | 13 | 13 | 7 | 7 |
| src/Init/Data/Array/Lex/Lemmas.lean | 17 | 17 | 12 | 12 |
| src/Init/Data/Int/Bitwise/Lemmas.lean | 8 | 8 | 6 | 0 |
| src/Init/Data/Int/DivMod/Bootstrap.lean | 18 | 18 | 11 | 0 |
| src/Init/Data/Int/DivMod/Lemmas.lean | 18 | 18 | 10 | 0 |
| src/Init/Data/Range/Lemmas.lean | 8 | 8 | 7 | 0 |
| src/Init/Data/Vector/Basic.lean | 13 | 13 | 2 | 0 |

Counts were read from that run's checkpoint directories at 23:05 local time. The parallel
registry SHA-256, its calibration goal hashes and its declaration names are recorded in
[results/olr08b_pr399/preoutcome/parallel_cohort.json](results/olr08b_pr399/preoutcome/parallel_cohort.json).
None of this campaign's outcomes were consulted. The amendment only reacts to another cohort's
collected outcomes.

## Amendment

The prospective calibration roster is
[olr08b/calibration_roster_amended.json](olr08b/calibration_roster_amended.json): the registered 64
files minus the 7 above, 57 files (SHA-256
e57fa6d7ff847ab879d534bcd87b00ce264dbf3f54681a87cbda84e3c913824e). The original roster
(2e3c7f9d…) is unchanged. The selection rules, exclusions, feasibility thresholds (at least 120
valid menu-hard roots in at least 12 file clusters, at least 10 T-human audits, at least 95%
human closure) and headroom gate are unchanged.

After freezing, the registry is also audited against the parallel cohort's calibration goals and
declaration names. A collision is reported and registered before any outcome is collected.

The parallel registry also lists the 275 original OLR-08A evaluation roots, with identical IDs. This
campaign opens none of them.
