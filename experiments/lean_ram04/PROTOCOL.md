# RAM-04 protocol, frozen before collection and prospective observation

Preserve RAM-02 a3e7be130334ecf86b8f0befc912f4864418d385 and RAM-03
c1265ea1718680be4827c2528ca0b39d5603d6b1, their worktrees, labels and all
checkpoints. Use a separate RAM-04 branch and D:\Research Engine\RAM-04.
Reuse the existing Python 3.13.7 / PyTorch 2.6.0 and Lean 4.35.0 installations.
Caps remain 6 GiB process-tree RSS, 2.5 GiB CUDA allocation, 2 GiB available
RAM/commit and 512 MiB dedicated GPU headroom. No torch.compile or upgrades.

## Data and representation

Re-probe the original 27 FC-12 TRAIN roots with the RAM-04 grammar, retaining
the five RAM-03 whole-theorem development groups. Do not import or relabel the
old 961 or 4,973 observations. Add explicitly authored statement-only curricula
for exact/application, rewriting and existential/conjunction/disjunction
elimination. Hold out complete template groups, not random renamed instances.
Curriculum source bodies are placeholders removed before probing; no expert
proof prefix is supplied. These authored tasks never count as original MathCan
proof gains. Report natural versus authored support separately.

Collect up to 8 descendant states per natural root and 4 per authored root,
depth 4, maximum 8 goals. Probe every registered candidate, including failures.
Prefer successors reached by the three target families and retain multi-goal
states. Independently replay every observation and its parent in fresh sessions,
including structured local-context metadata and clean Lean process exit.
Timeouts are unknown and must not become negative labels. Before fitting,
require at least 8 positive and 8 negative examples of each target family in
both optimization and development. If this gate fails, report the gap and make
only training/development-driven revisions before prospective observation.

Use Lean-provided accessible local names, proposition status, head constructor
and first function-domain type. Candidate arguments use only this visible
information; no target/reference theorem or goal-solving oracle. Keep RAM-03
closing/definition/induction tactics. Register up to 12 exact candidates (bare,
symmetry/projection, one/two local applications), 8 rw candidates (orientations
and local rewriting), 6 rcases candidates (And/Exists/Or). Other families retain
their old caps. Named introductions make new hypotheses accessible.
Native, frequency and neural policies always receive the identical pool.

Model inputs: active context and conclusion, complete pending goal contexts,
candidate/family, observed goal count, referenced local types and textual
candidate-local/goal overlap. Never use theorem/file/template ID, source origin,
prefix, future state, verdict or outcome. Token hashing is lossy and disclosed.

## Training and selection

Train loop32/loop64 (4 shared passes) and flat64, seeds 17/29/43. Effective batch
64 from microbatch 16 x accumulation 4. Each target family contributes 16 rows,
8 positive and 8 negative; the remaining 16 draw 8 single- and 8 multi-goal
inputs. Sample only optimization data. AdamW .002/.01, clip 1, maximum 1,600
updates, validate every 100, patience 4. Weighted CE plus .2 SmoothL1 goal delta.
Select checkpoints and one seed per architecture using the mean of five dev
group losses: single, multi, exact, rw, rcases. Compute class weights on training
only; preserve natural/authored and family metrics. Exclude exact canonical
state overlap across splits. Check for conflicting labels on identical features.
Calibrate identical loop64 CPU/CUDA updates (10 warmup, 30 timed); require
1.15x speedup to use CUDA. Save full optimizer/RNG best and last weights.

## Frozen prospective comparison

Only declaration names were inspected for new target selection. Select source
positions 2, 4 and 6 in the four alphabetically first unused MathCan modules:
FoundryProbe, FundedHooks, OpcodeRefinement and TOEConeMatching. All 12 names
are frozen in config.json. Their goals, bodies and reference outcomes remain
unobserved until model/grammar/search and dev-selected weight hashes are frozen.
No old FC-13 or RAM-03 target is added to the primary score.

Keep RAM-03 persistent bounded DFS, saved-root replay, visible-state deduplication,
maximum 8 goals, no cross-policy cache. Primary: 30 seconds, 48 logical prefix
evaluations, 8 transitions; native, frequency, loop32, loop64, flat64. Secondary:
60 seconds, 128 evaluations, 16 transitions; native and loop64. Per-request limit
3 seconds and 100,000 heartbeats; at most two server restarts. Startup (10-second
maximum), complete final compile and independent byte-identical replay are all
charged. Reserve .5 seconds for shutdown. No late certificate gets credit.
If an operational episode prevents initial observations, preserve every attempt;
any recovery reruns the entire affected budget comparison for both/all policies,
never just failed cases and never with larger timeouts. Disclose its cause or
that the cause is unknown. Do not silently remove outages from the evidence.

Each accepted proof compiles the complete original theorem separately, rejects
sorry/sorryAx and every dependency except propext, Classical.choice, Quot.sound,
and passes fresh exact-source replay. Only these certificates establish coverage.
Report matched native-only/model-only sets, complete costs, all nine fits, family
precision/recall, source support, resource peaks, failures and negative results.
No gain is promised. Publish a separate PR based on RAM-03; never merge.
