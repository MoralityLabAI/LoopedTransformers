# Second-round sandbox referee audit: LBF-18/19

**Verdict: major revision; preserve the sound abstract interpretation result.** Workshop submission is plausible after an executable-refinement clarification and corrected proof-search comparison; the current evidence does not establish a recurrent-reasoning or general Lean wall-time advantage.

Reviewed merged commit `f56b91b0746322860fe26de702c2a94ebb9b2b17`, original LBF-19 archive `c7a1d417b394fcb8cb64b05270f63e6a537d0c21`, freeze `2629bf784d97166f85ab786dba01edd76c7f27d5` and first independent referee commit `8102fb2190e3c0d30a5a7fad93694d4eaa0af6bd`. This is a new retrospective ChatGPT sandbox audit, not a claim of a Fable replication. It does not modify any prior registration or result.

## Original objections, reassessed

| Original issue | Assessment | Evidence and remaining work |
| --- | --- | --- |
| M1 unsound suppression | Resolved for formal operator; executable refinement partial | LBF18IncrementalAndCollisions.lean lines 830–846 and 1042–1169 prove both guarded transfers, including the modulus-free mask variant. Reported enriched 64/100 to 0/100 false-negative systems and 192/192 Python/Rust path parity. Need a general correspondence theorem from concrete Lean definitions to local Python/Rust code, plus explicit input validation. |
| M2 accounting | Improved, central economic issue outstanding | LBF-18 reports unique enumeration and clean timing. LBF-19 measures live tactic cost but enforces checks that are unnecessary for its exact syntactic rewrite domain. See deferred audit below. |
| M3 controls | Partial | LBF-18 adds exact_mm/absint_mm and symbolic readout, but no learned Hamming-only control. LBF-19 registered count-mode roster incomplete. Graph bridge missing from H1 live contrast. |
| M4 learned recurrence | Resolved as negative result on tested models | LBF-18 symbolic readout 28.73 vs recurrent 29.15 work, root interval [-0.69,+1.91] for recurrent minus symbolic, conditional on 3 frozen seeds. LBF-19 dev comparators and Track R remain negative; no matching direct structural-prior control. |
| M5 bank attribution | Partial | Joint factorial measured inference dependence with interactions and intervals; retrained factorial still absent. |
| M6 generalization | Improved, limited | 64 LBF-18 sealed roots and 24 beacon-derived harder LBF-19 roots; 28/72 live primary episodes close. Only 3 frozen seeds for search; 5 for Track R; restricted generators. |
| M7 formal coverage | Improved, partial | Boolean support is proof-relevant. LBF-18 theorems establish abstract operator soundness, not general executable equivalence. zero_preserving_gate_support is a Nat product lemma, not a theorem of float32 cell arithmetic. LBF-19 Complete1/chain_sound prove soundness under abstract completion premises, not that Python completion always meets them. |
| M8 custody | Improved, partial | LBF-19 frozen code blobs identical at registration, results and merge (five modules plus registered test). Freeze commit is unsigned and its author/committer time alone is not an independent server timestamp. The seed formula has two inconsistent descriptions. |

## New major objection: verifier placement is an experimental treatment

The fragment represents a string as a composition of arbitrary unary functions on A and each rule as an unconditional equation h_i : forall z, lhs(z)=rhs(z). Every exact symbolic match yields a valid proof step: instantiate the suffix, take symmetry if needed, and apply congruence for the unchanged prefix. Compose equality proofs along the resulting path. Thus intermediate native tactic calls are optional in this domain, and final proof checking is sufficient.

**Independent experiment.** A second implementation of layer-synchronous bidirectional BFS, in the cloud JavaScript sandbox, read all 24 archived cases directly from the pinned GitHub source. All 24 solved; all 96 archived counter values (expanded, visited, work, path length for each root) matched; all paths independently replayed. Mean search 0.8693 s, median 0.5725 s, max 3.675 s; mean final path 18 steps; mean visited 107039.29, expanded 50627.83, work 576828.58. Timing is specific to the sandbox and should not be compared directly to a Windows native run. This establishes an exact symbolic/deferred alternative. The independent sandbox did not have Lean installed. The subsequent Linux CI audit compiled all 24 regenerated explicit congrArg/calc proofs axiom-free in Lean 4.19.0, with 0.5515 seconds mean search and 7.4577 seconds for the complete 24-proof batch. It also checked 144 archived live episode records, replayed all 28 successful paths, and matched all 13 original certificate hashes after CRLF reconstruction. See CLOUD_RESULTS.md and the passing audit artifact.

The 2695 s projection uses approximately 107039 live discovery checks at 25.18 ms per check. It is conditional on eager checking of every novel search state. The alternative needs one final checked certificate per root, containing an average of 18 equality steps. A fair same-machine comparison must include both eager and deferred variants for each search controller, and distinguish proof acquisition latency from search abort latency.

## Confirmatory analysis

- H1 in the frozen registration compares with rootwise max(geom_mm, graph_bridge); analysis silently falls back to geom_mm if graph_bridge is unavailable. The actual archive contains no count-mode confirmation and no live graph_bridge. Primary 28/72, any-seed 13/24, all-three 7/24; 0/24 geom_mm. This restricted contrast is encouraging. A subsequent independent count-mode graph_bridge replay on the same 24 roots also closed 0/24 at B=512 (mean 307.33 novel calls), and geom_mm matched its reported stratum-specific call means. This fills the missing symbolic comparator retrospectively and supports the closure contrast, while the originally planned live/count-mode campaign remains incomplete. Exact one-sided sign-flip for the 13 positive nonzero root differences is 2^-13 = 0.00012207.
- H2 and H3: primary non-speculative discovery calls are bounded by its 512 charged successor budget, whereas the alternatives always hit a 2000-call cap. The lower call totals are mechanically implied by the stopping rules in these realized trajectories. They are resource comparisons of different stopping policies, not estimates of required calls to successful proof. H2 also requires H1 to pass. Recompute time-to-proof curves under shared call and time caps; no post-hoc substitution of a different H1 comparator.
- H2's reported wall-clock comparison on 13 any-seed-closed roots averages across all primary seeds on those roots, including failures. Comparator runs are censored. In lbf19_harness.py lines 194–241, wall timing starts after NativeSession initialization and stops before final certification; it omits both costs. Relabel the measured wall time as search-loop time; measure total proof delivery separately.
- The 28 theorems are in 13 standalone files. Archived compilation time is mean per file, not mean per proof. Zero mismatches are evidence for the restricted rewrite correspondence, not proof that intermediate checking adds utility.
- Frozen LBF-19 source files are unchanged at merge, including original test hashes. CI checks 18 pytest tests and 18 tree-automata theorem declarations. The earlier 144-episode campaign has not been independently replayed with its learned checkpoints in this sandbox.

## Beacon

The official quicknet chain parameters give genesis 1692803367, 3-second period (https://docs.drand.love/blog/2023/10/16/quicknet-is-live/). Round 34258635 was scheduled 2026-11-25 04:01:09 UTC. The unsigned registration commit lists author/committer time 04:00:51 UTC, 18 seconds earlier; this is compatible with the declared chronology, but does not prove when GitHub received it. PR #358 was created after the beacon. The raw relay endpoint was not accessible to this session, so the BLS signature and external publication ordering remain unverified.

Actual code `seed_int = int(a.seed_hex[:16],16)` takes prefix 996d4ba4053ae1d4 (11055575828098900436). SHA-256 of the ASCII hex string begins 99e78e60e7422dd5; SHA-256 of decoded signature bytes begins 24e46a35a95cff9d. These are distinct algorithms. The archived cohort records the signature prefix; correct the prose and use signed external freeze plus a single canonical seed derivation for future studies.

## Track T: bounded weighted completion

The new ground-rule construction and proof sketch are in `GRADED_GROUND.md`; its independent sandbox fixture produced 100 systems, 4676 term checks, 434 beyond-horizon negatives, 0 false negatives and 0 false positives for truncated distances at K=5. Maximum 19 states, 3 saturation rounds. This is a post-hoc ground-rule pilot, not a general result for left-linear variable rules. Original Python experiment's after-results planted run and H5 failure should remain visible.

## Track R: prior versus recurrence

The six trained arms over five seeds yield 30 reported fits, with masked variants reusing weights. The archived learned_transfer directory contains a summary and three figures, no individual checkpoint files or manifests. The training loop creates models, evaluates them and archives aggregate/per-seed metrics; shard serialization stores result objects. No checkpoint digest or training trace was found. DataGoblin should ingest a study-level discovery record and 30 run-receipt references with checkpoint identities unknown, not fabricate 30 distinct SHA-256 specimen records. Keep public test metrics in evaluation custody unless explicitly admitted in a separate research-only lane.

The prior's AP advantage is partly explained by structural hull masking; after shared post-hoc masking the non-prior recurrent and direct arms nearly tie. Hard-sound readout remains the full symbolic hull. The missing factorial is recurrent/direct × prior/no-prior under comparable active compute and identical masks, plus a symbolic-hull readout, with root and seed uncertainty. In-hull novel-cell AP, word-level misses and measured runtime against production Rust are required.

## Paper and next experiments

See `PAPER_OUTLINE.md` and `LBF20_PLAN.md`. Preserve the modest formal contribution and the conditional empirical result for cheap exact rewriting. The strongest defensible economic claim requires a domain in which native tactic outcomes reveal successor information unavailable to a cheap exact symbolic model. Run fresh confirmation only after fixing the comparison and freeze protocol.
