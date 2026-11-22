# NAN-STREAM-36 cached multi-evaluation scheduling

A single loaded checkpoint serves up to two independent READOUT34
instances in one physical prefill and decode stream. Keep the one-pass
and two-pass cache requirements separate. Pair by depth, split, context
event count, arm and readout, then by nearest frozen prompt length. Order
jobs by the earlier original slot. Stop and prune rows independently
while preserving their original padded physical time axis. Keep every
registered slot and require the existing pinned prompt hashes and strict
final scorer. No correctness outcome is used by the packer.

This is a separately planned execution condition. READOUT34's verified B1
runtime and the active RAM28/BF31/CACHE32 queue retain their own numerical
policies. A fake cache fixture checks the production packing and stop
driver with two rows, unequal prompt lengths and a real survivor-index
transition. Device preflight must separately measure a real B2 canvas for
each depth, row pruning, physical position and independent same-input
B1/B2 score parity, along with CUDA time, host time, RAM, VRAM,
temperature and clocks. Count row throughput and total work, rather than
inferring speed from percentage GPU activity. The planned B2 gate is a
single-model device condition before scoring. At this stage the packer
and driver make no pretrained or GPU calls.

## Verified software and pending device measurements

All 672 slots pair at identical prefill lengths, for 336 B2 jobs per
precision with zero planned right padding. The longest reserved physical
canvas is 3,463 positions. A separate tiny real DynamicCache fixture
deliberately added a padding hole and then pruned the completed row.
At each depth the synthetic full-vocabulary logits agreed between B2
and standalone B1 both before and after pruning. That fixture used a
maximum of 652.2 MiB sampled private RAM on CPU. The published
shape and cache receipts bind the two fixtures. Actual 3050 throughput,
numerical comparability and device admission remain pending.

[Schedule and stop check](SCHEDULING_VERIFICATION.json) ·
[CPU cache and head check](CACHE_VERIFICATION.json)

The device preflight procedure now runs a CAL pair at each depth and a
deliberately padded pair through prefill, incremental decode, and
row pruning. It compares the entire vocabulary against B1
within a fixed envelope of 0.0625 absolute plus 1% relative logit
difference and requires the same greedy choices. Capacity is a separate
two-row test through the scheduled 3,463 positions. On tiny CPU weights
the production procedure passed with exact logits, 38 resource checks,
and two deliberately rejected readout differences. The 3050 gate still
requires a dedicated admitted run.
[Production preflight CPU receipt](DEVICE_PREFLIGHT_CPU_VERIFICATION.json)
