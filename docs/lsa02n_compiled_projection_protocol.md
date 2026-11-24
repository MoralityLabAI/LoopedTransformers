# LSA-02N: compiled local potentials on the hard equality-search frontier

Registered before fitting and running a fresh cohort. CPU only. Parent: LSA-02M native result, PR #338. No sealed test access.

## Question

LSA-02M recovered a learned/sham separation, but pair-projection greedy closed all 32 systems by 256 expansions with far lower runtime. Four visits and off-path supervision showed no primary-endpoint advantage. Test whether the local geometry can be extended symbolically or captured by a cheap train-only learned additive potential. The research target is bounded proof-search efficiency, not unrestricted proof coverage or recurrence.

## Prospective roster and custody

Regenerate/authenticate the original suite, LSA-02L development manifest and LSA-02M development manifest (SHA-256 be142147c008a06e55bb01967a90cf27b27b128984d5f4d59cae147b64bb69b3). Exclude their entire equation-system alpha-lineage census. Freeze 32 new systems using seed 2026112404, cycling word lengths 8,9,10, reachable shortest-distance band 10–11. Use the original generator and its cap. The roster is an intentionally hard development distribution; no comparison of raw closure percentages across cohorts. Do not parse any old held-out rows for evaluation.

Regenerate the original native training proofs and LSA-02M's certified 994 ordered pairs, including 760 additional native proofs. Authenticate the exact pair-file SHA-256 24ade5f3b9830c2539f60397003a2ec3bc30dd0b39ee1c4d0d7ff59d18f4b4a5. Only these 64 training lineages supply fitting examples. Keep native proof evidence separate from exact grammar distance labels.

## Projection geometry

For a width-k contiguous window, construct a relaxed reversible graph on the 4^k local words. A rewrite fully inside the window matches both symbols; a rewrite crossing its boundary matches only the overlapping symbol and can choose the outside symbol freely. Any real rewrite projects to a relaxed transition or identity. If q_i is the shortest relaxed distance to the witness window, then max(max_i q_i, ceil(sum_i q_i/(k+1))) is admissible, since a two-symbol rewrite touches at most k+1 windows. At k=3 take the maximum of this bound and the pair bound. This is a relaxation, not a full-word search oracle.

Public arms: original Hamming greedy, bidirectional BFS, exact prior Hamming + 0.25*pair bound, Hamming + 1.0*pair bound, Hamming + 0.25*max(pair,triple bound), and Hamming + 1.0*max(pair,triple bound). All greedy variants share the same heap, g-based reopening, tie-break, goal-on-pop and 15-rewrite path cap. Report setup work (local states, rule checks, abstract expansions and edges) separately; include it in planner wall time.

## Compiled learned potential

Use the previous 11 alpha-invariant pair features. Let phi(w) be their average over valid adjacent windows (linear ablation); primary phi additionally includes all 66 degree-two monomials, including squares, within each window. No cross-window products or IDs. The feature is still an average of local functions. On each certified harder/easier pair, fit the residual distance gap relative to Hamming + 0.25*pair bound. Solve deterministic float64 ridge least squares with objective mean squared error + 0.01*||beta||^2, no intercept. Weight all ordered pairs equally. Train primary and linear ablation once, duplicate each solve and compare exact coefficient digests. Two shams (107,109) permute residual targets within pair kind and word-length strata, with identical feature matrix and solver. No tuning, checkpoint selection or fitting on development data.

For each theorem, compile beta dot local feature to an integer lookup table at each window and pair state (round to 1/256). Center the average against the witness word, round the difference to 1/256, and clamp it to [-2,2]. Add the correction to the exact pair prior. The compiled model and sham have no per-state tensor/model inference. Zero coefficients reproduce the pair prior exactly. Record setup coefficient products and state scoring calls; include all table compilation/encoding in timing. Report training residual prediction diagnostics descriptively.

## Native evaluation

Run independently at 16/32/64/256/1024 full-word expansion caps on the frozen roster. Count expanded words, enumerated edges, scoring states, projection setup, coefficient products, planning time and logical native tactics. Exact BFS ceiling must certify all 32 roots. Freeze proposed paths before native replay. Every distinct successful root/path must pass native Lean kernel, exact state replay and independent axiom-free compilation. Deduplicate physical certification work while retaining per-algorithm logical costs. Report depth band, per-root and per-sham seed results; edge thresholds are retrospective filters within expansion caps. Joint expansion+tactic counts are illustrative.

Primary developmental gate at 64 expansions: primary quadratic compiled potential exceeds strongest aggregate public arm by >=3/32, root-paired 95% bootstrap lower bound >0 (strongest public recomputed in each sample), and exceeds mean sham by >=2/32 and both sham seeds individually. Linear comparison is attribution only. A practical compute claim additionally requires lower measured planner wall time at a matched or higher closure level, including table setup. Record wall time on this worker as descriptive, not a universal runtime claim. If public search saturates, report the ceiling and move to a more informative task grammar; no changing the registered gate. No architecture promotion or sealed confirmation.
