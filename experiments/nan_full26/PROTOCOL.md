# NAN-FULL-26 frozen development gym

Run every context from the existing gym rosters: GYM05 322 (including its 32
local routing cases), COMP14 240, CTX08 291, CTX21 624 and native STORY23 72:
1,549 baseline trajectories per precision. Source order is deterministic:
STORY23 first, then the original order within GYM05, COMP14, CTX08, CTX21.
No new sources are selected using model outcomes. This is a new trace-prompt
condition on exposed development roots. Earlier one-token scores are not its
matched control. Local routing source text remains local; only hashes enter Git.

STORY23 keeps its bounded intermediate-state/path/final packet. Other tasks
quote the original task, explicitly replace its reply-format instructions,
and request a bounded JSON trace array plus final answer and END marker.
No decoding constraint forces the grammar or final answer. Grid tasks have
256 generated tokens; other generic tasks 128; STORY23 64. Stop at the first
complete anchored packet, EOS, character cap, or token cap. No prose rescue,
answer extraction from malformed replies, or retry. Trace text is an output
to inspect, not evidence that it faithfully describes internal computation.

After ALL baseline trajectories, run all 24 story main worlds and all 48
focused competence worlds through three A/B conditions: component only,
fixed CAL mean mask, and reached-state controller. Thus 216 pairs per backend,
432 additional trajectories. A is clean native; B is the actual free-running
intervention, using the qualified independent IO18 T2 checkpoint for that same
precision. Each pair initially shares a prompt; it can diverge token by token.
Both rows execute one real decoder GEMM batch through one resident model.
This is not inherited run_pairs' serial loop over B1 rows. After one reply
ends, its partner continues as B1, with the changed batch shape recorded;
no dummy continuation is generated to inflate GPU utilization.

Native two-pass recurrence, no KV cache. All 22 NF4 decoder layers and the
unchanged BF16 full-vocabulary head reside on CUDA. The BF16 control retains
four decoder layers plus its reusable stage and the same BF16 head. Token
embedding weights remain on CPU; exact token embedding activations are reused
on GPU. This costs no changed hidden-state arithmetic. CUDA fraction is .90.
One large model at a time. No weights, packages, external model service or
cloud accelerator are downloaded. Snapshot revision is
451ed48c3273ecef7ea8faaa43c31ce529763bb1.

Before scored episodes, check B1 old/new native code closure at all 40 sites
and all logits, B1 full-head chunk parity, B2 clean/clean row identity, and B2
all-one BPD closure. Record B1 versus B2 numerical differences; batch two is
a new arithmetic condition, not a bit-equivalence claim. Run the three fixed
capacity probes: B2×4608, B2×9216, B1×9216, in that order, with repeated source
token IDs and native mode. A capacity-only probe may fail and be recorded;
only previously passed shapes admit scored inputs. No case is truncated or
replayed after a scored OOM. A/B has no serial substitute if B2 is unavailable.
Finite limits: seven days of loaded-model execution per backend, six hours
per pair, 160 GiB artifacts per backend, 20 GiB free-disk floor. Any failure,
budget stop, incomplete trajectory and remaining unrun cases stay explicit.

At EVERY generated prefix save all 40 actual-token vectors: both recurrent
passes × layers 5,11,17,21 × residual-after-layer, o_proj input/output,
down_proj input/output. BF16 bits are stored losslessly as uint16. Save each
normalized boundary, prefix/history hash, batch row/shape, token/readout,
actual BPD mask, and native output at that row's actual reached input.
Hooks buffer tiny GPU vectors and transfer them together, avoiding per-site
CPU synchronizations. Write-once closed episodes signal the host reader.

The CPU lane runs alongside generation after older host lanes have ended.
It uses the eight existing CAL-only SAE towers, all 128 features/top4 and
their fixed random-initial controls. No dictionary/factor refitting. It
independently verifies actual masks and local writes with the previous fixed
tolerances, all state hashes and prefix coverage, and separately grades
packet format, final answer and story checkpoints/path. Per-prefix recurrence
and codes stay local. Sanitized coverage/grade/resource milestones are pushed.
CUDA event intervals and forward wall times are separately reported; event
intervals are not kernel occupancy. Report real B2 fraction, padding, peak
allocator bytes and actual generated-token throughput, including the long
BF16 staging costs, instead of claiming utilization from one snapshot.

The existing CTX21, STORY23 and XREF22/25 attempts run to their frozen
terminals first. The new coordinator captures their authenticated Windows
process handles immediately, including later-created STORY23 host identity.
It never stops a valid predecessor or treats an observation timeout as exit.
It starts one full-gym model worker and one streaming CPU audit process.
All numerical processes use one CPU thread, Below Normal priority, with
thread limits set before numerical imports. The separate math process is
never stopped, reprioritized or used for inference.

Independent roots are the evidence units; worlds, tokens, loops, routes and
features are repeated measures. No fitted cross-precision alignment. Any
future physical cross-precision comparison must match prompt/history AND
execution shape; free-running trajectories after divergence cannot be paired
as identical-input geometry. Cross-precision accuracy differences include
batch-shape differences wherever capacity admission differs. No formal inference, donor transfer, equivalence
or causal confirmation: effect floor, sham/dose matching and power remain
uncalibrated. Verdict remains REDESIGN_OR_PILOT_ONLY. The original 168-case
test is never opened, including for filtering or exclusions. Finishing this
finite queue is a milestone; the user's continuous Nanbeige goal stays active.
