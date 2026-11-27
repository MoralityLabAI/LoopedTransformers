# LOGRANK-BPD-18: budgeted adaptive cutset refinement

Development only. 27 fresh worlds, three per dimension (12/24/48) and family (relabeled chain, relabeled variable-hub chain, global random triples). Each world has 2n unary and 3n triple observations, four unobserved pair targets, uniform hidden state, paired .15/.35 BSC noise and known likelihood. Hub roots have one, two or three hubs; their identities are offline, never supplied to the controller. All five arms receive identical observations. 54 conditions and 270 solver runs. Protocol committed before main execution. No causal claim, native BPD/MMS model training, protected data or confirmation. Three independent worlds per stratum; noise conditions and queries reuse worlds.

## Controller and soundness

Condition each nonzero target q on both parity signs by pivot substitution, leaving a common variable topology. Maintain one disjoint cutset partition per query, with the same cell masks on both event sides. Initial cell bounds use the likelihood-weighted independent-factor bound from 16. Actions either refine a leaf with scope 3/5/7 or partition it by one/two additional degree-selected cut coordinates and apply scope 3 to each child. Cut depth at most four. For each side at a node v, retain the old valid enclosure I_v. When refining, intersect it with the new enclosure. When splitting, set

    I_v(new) = I_v(old) intersect [sum_child L_child, sum_child U_child].

Every child represents a disjoint assignment cell. Summed child enclosures are valid because the true integer likelihood weight is additive. The intersection remains valid; recursion through ancestors preserves a nested interval for each event-side weight, even if a child kernel is looser than the parent. For positive weights, the event interval [L1/(L1+U0), U1/(U1+L0)] consequently narrows monotonically. A sign is issued only at a certified lower confidence of .9. These are model-conditional posterior guarantees, not guarantees of individual outcomes.

Predict work and peak live numeric table slots from public masks before an action. Planning reproduces deterministic minimum-neighbor ordering, greedy mini-bucket packing and all scope sizes. Each action must fit the remaining 900n join-entry access budget, 45n peak lower+upper factor-table slot limit, and 128 kernel calls (including initial independent bounds). No partially executed action. Join-entry count is the number of assignments times group factors visited, not CPU instructions or arbitrary-precision bit operations. The slot ledger includes initial factor tables, the old bucket and new tables; it excludes temporary Python lists, other objects, independent-factor preprocessing, planning and receipt storage. Per-slot payload estimate uses the Python int object size for 2^n max(u,v)^m plus one pointer; it is a conservative bound for counted numeric table payload, not a measured total RAM cap. The actual solver stats and independent scope audit reproduce the ledger. Adaptive priority uses predicted no-split plans first; otherwise estimated split reduction, weighted by current query interval width and cell upper-weight uncertainty per planned work. It does not optimize true future information gain. Static controls use fixed scope 3/5 or two degree/random cuts with scope 3; each processes query targets in public order, skipping infeasible actions. All share the same budgets, not matched actual wall time.

## Results

Exact query counts across nine worlds per family at either noise level:

| Family | Scope 3 (.15/.35) | Scope 5 | Degree cut2 | Random cut2 (.15/.35) | Adaptive |
|---|---:|---:|---:|---:|---:|
| Chain | 9/9 | 36 | 36 | 18/18 | 36 |
| Hub | 3/4 | 31 | 31 | 10/11 | 34 |
| Dense | 0/0 | 0 | 0 | 0/0 | 0 |

At .15 adaptive certifies 33 chain, 32 hub and 2 dense answers (67/108), all correct in this sample. Degree cut2 certifies 33 chain and 33 hub (66/108), all correct. Scope5 certifies 63/108. No arm issues a certificate at .35. Adaptive yields 140 exact query/noise cases across both rates, versus 134 each for scope5 and degree2. Mean interval widths at .15 on chain/hub/dense: adaptive 0/.05556/.93778; scope5 0/.09235/1; degree2 0/.02463/1. Degree2 can give narrower nonexact intervals despite fewer exact cases. Across 54 paired world/noise conditions, adaptive width improves/ties/worsens relative to scope5 on 10/44/0, and relative to degree2 on 9/44/1 (tolerance 1e-12; repeated noise is not a new independent root). No statistical superiority claim.

48D, .15 noise (three worlds per family, 12 query decisions):

| Family/arm | Mean width | Exact | Certified | Mean counted work | Peak slots | Mean ms |
|---|---:|---:|---:|---:|---:|---:|
| Chain scope5 | 0 | 12 | 11 | 11029 | 932 | 71.0 |
| Chain adaptive | 0 | 12 | 11 | 11029 | 932 | 133.9 |
| Hub scope5 | .195829 | 8 | 8 | 24808 | 1788 | 85.9 |
| Hub degree2 | .073791 | 8 | 10 | 27051 | 1078 | 243.1 |
| Hub adaptive | .166667 | 10 | 9 | 25703 | 1788 | 248.2 |

Adaptive correctly selects one-coordinate cuts on the 48D one-hub world and two-coordinate cuts on the two-hub world (all queries exact). In the three-hub world it solves two queries with scope 7; remaining budget is insufficient to complete the other two. All 48D dense arms execute zero table refinements: a post-run same-source diagnostic checks 72 initial plans (includes random control), all exceed 2160 slots, 12 also exceed 43200 work. Initial independent-factor intervals remain effectively vacuous. This is budget rejection, not a completed dense elimination comparison; no general impossibility result.

Over all 54 conditions per arm, mean work for scope5/degree2/adaptive is 6880/13223/9795 entries. Mean elapsed time 30.0/73.7/89.2 ms respectively, with adaptive plan construction alone averaging 43.4 ms. Four concurrent fresh forkserver workers, times descriptive. Adaptive uses 97 cut actions out of 189 actions, 24 scope-3, 57 scope-5 and 11 scope-7 no-cut actions. All 270 runs respect all three caps. Adaptive counted payload upper bound peaks at 307536 bytes at 48D; process RSS roughly 80-81.4 MiB (includes interpreter, worker and receipts). No measured process-RAM improvement. The controller has a small exactness gain but overhead is substantial; fixed scope5 remains the strong speed baseline for low-width graphs.

## Verification

Independent absolute-mask factor replay and set-scope accounting PASS: 6244 weight bounds and budget replays; 821 monotone actions, 437 disjoint splits, 1080 complete frontier partitions and event intervals; 2278 exact cell enclosures by direct 12D summation, 840 exact final interval enclosures over 42 conditions. Exact references use exhaustive sums for 12D and independent chronological frontier DP with offline ordering for larger structured worlds. Verified static/degree cut choices and action structural constraints, not independent maximization of adaptive priority or every rejected plan. No independent timing replay. Raw receipts retain all actions, kernels, grouping traces and every failed result. Post-run diagnostic is separately labeled and performs no additional likelihood evaluations. Git includes source, summary and digests; full raw receipt and logs in bundle.

## Reproduce

Python 3.13, standard library only, Linux resource/forkserver.

```
python pilot18.py
python audit18.py
python diagnostic18.py
```

Pinned pilot16.py/audit16.py copies from 16. Next: use a single low-width elimination pass to recover reusable messages for many target marginals; avoid re-solving both sides of every target. Compare cached separator-message contractions and direct conditioned re-elimination under a real workspace byte budget and wall-time matched controls. For dense graphs, stream initial factors to reduce workspace, test bit-complexity-aware budgets, and preserve certified abstention. No native integration, CI, or Lean compilation claimed.
