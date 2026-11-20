# NAN-RECOMP-09: memory-bounded native BF16 BPD fit

Updated 2026-11-20T11:00:14.890478+00:00

Evidence status: **queued**. Independently checked baseline contexts: 0/32. Actual optimizer steps completed: 0/8.

The original BF16 T=2 FIT07 attempt hit its 2-GiB added saved-tensor CPU cap before an optimizer step. This new condition retains that cap, the same exposed source roster, CAL order, initial SVD, four stock loss terms, eight-step final-checkpoint rule, null normalization and fixed diagnostics. It recomputes frozen blocks during backward, retains version-guarded references to immutable resident weights, and keeps the unchanged BF16 head on GPU. The selected layer-21 hook is excluded from recomputation so pass counters and reached-state routing remain correct.

BF16 previously passed the frozen screen: CAL 7/8, DEV 7/8, copy 16/16. This is not a new competence test. NF4 remains excluded because its original DEV screen was 5/8. The original failed run and all its partial artifacts remain intact.

Before fitting, all 32 new-execution baseline readouts must equal the original and every vocabulary logit must agree with the CPU head on the first source at both loops. Real masked factor gradients must be finite/nonzero. Random 22-layer/width-64 BF16/NF4 engineering already matched forward states, factor gradients and two stock optimizer steps per path under matched global/trainer/adversary RNG. That engineering evidence is separate from the real fit.

CPU threads=1, Below Normal priority, CUDA fraction=.85, four BF16 decoder layers plus the reusable stage and resident head, 7200-second between-stage budget. One large model at a time. No DEV/copy fitting, source selection, rank selection, causal claim, equivalence or donor transfer. Eight CAL/eight DEV source roots remain repeated measures on one checkpoint. The original 168-case test stays sealed.

[Protocol](../../experiments/nan_recomp09/PROTOCOL.md) · [Source seal](../../experiments/nan_recomp09/SOURCE_MANIFEST.json) · [Aggregate evidence](nan_recomp09_progress.json)
