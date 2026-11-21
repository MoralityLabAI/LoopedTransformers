# MLP27: candidate bounded activation workspace

CTX21 NF4 completed and audited all 312 registered 4K contexts, then stopped
on the first 8K forward while evaluating the MLP's gate/up activation product.
That original 85%-allocator condition remains `capacity_stopped_audited`:
312 of 624 contexts and 12,480 vectors checked; all 312 missing 8K outcomes remain
unknown. [Original report and failure](NAN_CTX21_PROGRESS.md) binds terminal
aggregate `b2b1d9befbeac4466d3059060f722b4e3a920aee`.

This candidate materializes the full intermediate tensor in bounded token
chunks of gate/up projections, then calls the physical down projection once
on that full tensor. It preserves the down-projection hook's token coordinates
and one logical BPD pass invocation per layer. It changes gate/up GEMM shapes
and is a new numerical condition; it is not a patch to a frozen worker.
No CTX21, STORY23 or FULL26 worker imports it, and no pretrained model ran with it.

Twelve CPU cases executed the actual pinned `NanbeigeMLP` class with tiny
synthetic weights, FP32/BF16, batches 1/2, 7/65/513 tokens and 32/256-token chunks.
The first fixture setup omitted the class's `mlp_bias` configuration flag;
that setup error was corrected before numerical comparisons. Both supported
chunk sizes were then tested, including the proposed 256-token default.

| Engineering endpoint | Result |
|---|---:|
| Synthetic CPU cases passed | 12 |
| Maximum FP32 output absolute difference | 3.725290298461914e-08 |
| Maximum BF16 output absolute difference | 0 |
| Down projection sees complete intermediate tensor | All cases |
| Physical down-projection hook calls | One per MLP call |
| Synthetic edited B row leaves A unchanged | All B2 cases |
| Disabling/removing adapter restores native outputs | Bit exact |
| Training/unsupported tensor-parallel branch | Rejected |
| Pretrained model forwards / CUDA allocations | 0 / 0 |

The comparison tolerances were rtol=.02/atol=.01, fixed in the CPU fixture
before execution. The tiny observed differences are engineering measurements,
not meaningful causal thresholds. CPU BF16 parity does not establish native
CUDA NF4 parity. Actual checkpoint closure, BPD masks/writes and padded as
well as unpadded capacity must be measured in a prospectively frozen new
condition before scored use. Original failures, sources and thresholds stay intact.

For illustration only, with batch 1, 8192 tokens, 10752 intermediate channels,
BF16 and 256-token chunks, the large gate/up/product activation storage changes
from up to 3 full 168-MiB tensors to one 168-MiB tensor plus three 5.25-MiB chunks:
504 MiB versus 183.75 MiB. This is an activation-storage bound excluding target
weights, attention, normalization, hooks, allocator reserve and GEMM workspaces;
it is not an observed total CUDA peak or proof that 8K fits on the 3050.

[Machine receipt](nan_mlp27_cpu_engineering.json) ·
[Candidate adapter](../../mechinterp/nanbeige_bpd_gym/mlp_workspace27.py) ·
[Synthetic CPU reproduction](../../mechinterp/nanbeige_bpd_gym/check_mlp_workspace27.py).
Only one CPU thread at Below Normal priority was used while matched BF16
loaded and began its unchanged registered evaluation. The continuous goal
remains active; the original 168-case test stays sealed.
