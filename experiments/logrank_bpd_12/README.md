# LOGRANK-BPD-12: adaptive verifier-budget allocation

Development only. Known quotient coordinates, uniform hidden state, binary-symmetric independent observation noise (known rate). Initial ranks 5/8, total widths 8/11. Sixteen training worlds and 24 disjoint evaluation worlds. A ridge controller imitates one-step exact target Brier gain on 32,640 candidate-scoring examples from training trajectories (correlated within worlds). The 16 features use BP moments, algebraic span membership and public query weights. No hidden target labels or exact posterior enter its selection rule. No native neural model, protected cohort or causal claim. See PROTOCOL.md.

## Reproduce

```sh
cd experiments/logrank_bpd_12
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python pilot.py train
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python pilot.py 100 106
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python pilot.py 106 112
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python audit.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python lookahead.py
```

Each arm obtains exactly 12 additional calls, no direct target observations. Candidate masks may repeat an initial observation but cannot be acquired twice. All final predictions use the exact posterior, including BP/learned acquisition arms; this isolates query selection from decoding. BP selection uses GF(2) elimination rather than enumerating posterior states. CPU selection times exclude posterior scoring/updating. No claim of speed advantage at these small widths.

## Results (24 worlds per noise, weighted Brier loss; lower is better)

| Policy | .15 noise | .35 noise | .15 mean rank gain | .35 mean rank gain |
|---|---:|---:|---:|---:|
| random | .101036 | .231167 | 2.917 | 2.917 |
| old_only | .145293 | .241760 | 0 | 0 |
| rank_first | .079858 | .244734 | 3 | 3 |
| entropy | .084571 | .238134 | 3 | 3 |
| exact_gain | .059990 | .245337 | 2.875 | 1.583 |
| bp_gain | .072830 | .238937 | 2.917 | 2.167 |
| learned | .080652 | .241671 | 2.917 | .958 |
| two_step (adaptive 12B) | .060439 | .241570 | 2.833 | 1.458 |

Initial mean Brier .183042/.237604. At .15 noise exact one-step and BP acquisition outperform random descriptively. Learned imitation does not improve on the BP heuristic. At .35 noise random has the lowest realized loss. Exact and learned policies reduce model-implied posterior risk to .227876/.227117, versus .236734 for random; realized error and conditional expected risk need not order the same way on 24 worlds. No statistical superiority claim. Each noise rate reuses worlds and policies share the same initial observations and step-indexed noise uniforms. Independent unit is world.

## Lookahead (12B, exposed-root follow-up)

For parity moments mu(q)=E[(-1)^(q dot h)|observations] and rho=1-2p, the one-step gain is a weighted sum over targets q of rho^2 (mu(q xor a)-mu(q)mu(a))^2/[4(1-rho^2 mu(a)^2)]. If q is outside the current row span S, mu(q)=0; a can inform q only if q xor a is in S (necessary, not sufficient). In-span probes cannot inform outside targets, and an out-of-span probe cannot inform in-span targets immediately. Several individually uninformative observations may jointly identify a target. Example: uniform h1,h2, target h1 xor h2, candidate observations h1,h2. Neither observation alone reduces Brier risk; together their expected gain is rho^4/4. For rho>0 the gain is strictly positive. Greedy one-step can select an irrelevant tied candidate when the budget is two. The two-step controller computes immediate gain plus expected best next-step gain under the two possible observations. It replans each step, and uses one-step at the final call. 12B reuses exposed worlds and is not independently confirmed; no consistent improvement over one-step or random was established.

## Audit and scope

Main 12: PASS, 336 trajectories, 139,776 posterior probabilities independently recomputed by integer likelihood weights (max abs discrepancy 4.996e-16); 4,032 chosen conditional gains recomputed, 2,880 classical choices checked independently, 1,152 BP/learned choices replayed with the original feature code. Rank, masks, shared noise draws, score counts, coverage and disjoint training/evaluation seeds verified. No independent BP feature implementation or ridge training replay. 12B is a separate follow-up run, outside this independent audit. No repository CI claimed.

Source, raw model/trajectory receipts, audit, 12B trajectories and root-level summary are preserved in the experimental bundle. Receipt SHA-256:
- receipt_100_106.json: a063392b22e324bc4d9a85a78e5820e443c2238cab6d6df64e46952097b87f1a
- receipt_106_112.json: 079a1039bd50e1af926a0e248983073e446e4a4d48ec1fb0155d5378067cc9dc
- audit.json: be51c9002a3318981723ce214caef2fc4b9a1f2993f6ac51b18f070fd3960f71
- lookahead.json: 3900baf015857053e5504e70c0e2268aa02a540e52bba1febb4737c607c85773

## Next

Freeze a new prospective development protocol on larger and diverse verifier families. Compare random, BP, exact small-state oracle, and structured lookahead with actual end-to-end BP decoding rather than exact posterior scoring. Include hidden/uncertain basis maps, unequal verifier costs and confidence-gated stopping. Learn policy only where structured methods leave measurable headroom. Use EvalMath independent-root power/resolution planning before causal claims.
