# Storage-backed BF16 builder: CPU integration

The prospective production assembler now passes the actual pinned-class CPU
integration. This is **MOCK_ONLY engineering**, using a random small model,
synthetic factors and synthetic CI weights. It supplies no new competence,
recurrence or causal result. No pretrained forward or CUDA context was used.

The builder starts from a complete BF16 meta architecture, binds the CPU
embedding/head and BPD reference weight read-only, materializes four resident
layers and the boundary norm, and allocates one reusable stage. All other
161 of 201 parameters stay meta; each stage copy maps one source tensor and
releases its ordinary tensor owner afterward. There is no complete CPU decoder
bank. Resident and stage RoPE buffers are initialized before execution.

The unchanged integration criteria passed all 22 captures and 396 stage
copies: native B1, batched B1 and ragged B2; all 40 captured states per row;
boundary states and full tiny-vocabulary logits; actual BPD hooks for all-one,
component-only, static-CAL-mean and reached-state routes. All-one closure and
the clean A row were exact, while nontrivial edited B rows changed. The scoped
cleanup retained no unused duplicate observer vectors after export and changed
no exported native tensors, states or logits. All four route comparisons were
byte exact against the dense CPU reference.

Maximum sampled private commitment was **712.699 MiB**; minimum system
available RAM was **5.833 GiB**, across 98 samples.
The one-thread Below Normal process used a 0.75 GiB private cap, 5 GiB start
admission and 4 GiB reserve, with two-second self-only checks. It exited
successfully. These samples do not forecast full-checkpoint CUDA memory.

The future BF31 namespace is deliberately incomplete and **unregistered**.
Its loader was checked to reject the missing source manifest before importing
Torch. No BF16 supervisor or pretrained control has started. A complete source
and resource freeze, original qualified factor binding, actual checkpoint
stage-copy checks, CUDA closure and padded capacity admission remain required.
The existing 1,549 baseline / 216 true B2 A/B scope is retained. BF16 waits for
the current NF4 model, canonical host analysis and small audit to end.

The earlier successful loop integration and its original failed RoPE fixture
remain separate and preserved. The frozen live NF4 condition was not edited.
The original 168-case test stays sealed. The continuous goal remains active;
no formal comparison, mediation or equivalence is claimed.

[Assembler](../../mechinterp/nanbeige_bpd_gym/lazy_bf16_backend.py) ·
[CPU harness](../../mechinterp/nanbeige_bpd_gym/lazy_bf16_integration.py) ·
[Sanitized receipt](lazy_bf16_backend_integration.json) ·
[Prior loop integration and preserved failure](LAZY_BF16_LOOP_INTEGRATION.md) ·
[Live full NF4 condition](NAN_RAM28_PROGRESS.md)
