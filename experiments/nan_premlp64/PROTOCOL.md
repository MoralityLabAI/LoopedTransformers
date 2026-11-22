# NAN-PREMLP-64 direct residual capture

CARRY63 found that most of the focused answer-margin gain was already
present in the last layer's effective carry. This instrument records the
direct post-attention residual and both operands of its native add, then
records the MLP read, write and pre-edit layer output. It uses the existing
genuine two-row last-valid-token chunked prefill and phase-boundary gate.

In each phase capture the input to attention RMSNorm, the attention
output projection, the input to pre-MLP RMSNorm, the MLP down-projection
input and output, and the output of the native layer before its boundary
patch. Each hook clones one vector per eligible row and returns no
replacement. The live packet must reconstruct both residual additions
in BF16. An approved phase-1 boundary patch remains downstream of the
second recorded addition and is reflected in the reached head and the
phase-2 physical inputs. The reference row must close in both phases.
Layer normalization and attention computation are unchanged.

For a pair of rows at a fixed physical phase, split the output difference
into the incoming attention residual, the attention projection write,
the MLP projection write, and the two measured residual-add rounding
terms. Use the actual operands to compute each rounding vector. Replay
the fixed two-token head on 32 BF16 coalitions for exact rational
five-group Shapley weighting. The separate continuous accounting reports
each numerator term, inverse-RMS change and head-rounding remainder.
These readout allocations describe the captured endpoints. Causal
evidence comes from qualified, signed delayed interventions with the
same companion checks and matched shams.

Commission on the snapshot's actual Nanbeige decoder classes, using
three random BF16 CPU layers with a width of 64 and no weight shard load.
Exercise resident and reused staged-layer paths. Tie receipt checks to
the pinned checkpoint code, the exact native compare and the registered
CPU resource limits. The production adapter can be imported without
touching the running workers. Device and pretrained verification need a
separate guarded registration after the existing lane. Root-level
pretrained evidence and formal comparisons are pending. Keep ARG-03
sealed, and keep individual captures local.
