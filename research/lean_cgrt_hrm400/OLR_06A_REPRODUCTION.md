# OLR-06A: independent reproduction of Fable's bidirectional parity pilot

2026-11-24. Synthetic feasibility evidence only, no Lean. Source: [archived referee script](referee_olr06/ledger_pilot.py) and [original logs](referee_olr06/review_logs.txt). Exact uploaded script SHA-256 b25157ace05b48a81ce5627e3bdf959d90f690f689eb964fdae38cfc5868366f. The archived script and logs have Git blob hashes 597293eba7f72d58dc18030c3ca57c13049c2cb3 and 3b82ec25fd86f9591204a8c7e70625e5b1cd04a9, matching the uploaded files.

## Independent local reproduction

Unchanged supplied script; command:

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONHASHSEED=0 python -u ledger_pilot.py bi 0,1,2 2000

Python 3.13.5, PyTorch 2.10.0+cpu, Linux x86_64, one Torch thread. This is independent reproduction across a different software environment, not an assertion of identical checkpoint bytes. The original referee used 2.5.1+cu124 on CPU. Model parameters 33123, original generator and sampler unchanged. The training loss printed by the original script is the final minibatch loss; no model archive is produced by that script.

| Seed | DEV T8 | Deep T8 | Deep T16 | Deep T32 | T8 deep unknown | T32 deep unknown | Fit time (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1.000 | .999 | .999 | .995 | 1.000 | 1.000 | 59 |
| 1 | 1.000 | .998 | .994 | .965 | 1.000 | 1.000 | 58 |
| 2 | .999 | .993 | .986 | .972 | .991 | .952 | 61 |
| Mean of logged rounded scores | .9997 | .9967 | .9930 | .9773 | .9970 | .9840 | |

Observed deep-panel hop>=5 accuracy at T8: .975/.915/.823. At T16: 1.000/1.000/.932. At T32: .998/.946/.921. The original Fable deep-panel mean active-slot accuracies were .9957/.9917/.9900 at T8/16/32. The depth transfer at 8 and 16 visits replicates; longer iteration degrades some seeds. The maximum known path length and unknown-region metrics must be retained rather than selecting the best depth after looking at the held-out panel.

Local log SHA-256 47fb82106b6730087906e4b5644cad3242d91f4b60fa7eb6b789277010542e10. No full seven-arm independent results are claimed here. A pinned CPU seven-arm CI matrix has been registered in [olr06a-referee.yml](../../.github/workflows/olr06a-referee.yml); its artifacts are the separate record of that run.

## Scientific status

The exact signed-component solver still solves every consistent graph and identifies every unanchored component without learning. This confirms the feasibility of learned directional graph transport, while leaving no accuracy headroom for a learned parity solver over the explicit reference. OLR-06B therefore moves to paid acquisition of missing facts and compares to same-information Bellman planning. No native proof or 5M architecture claim follows.
