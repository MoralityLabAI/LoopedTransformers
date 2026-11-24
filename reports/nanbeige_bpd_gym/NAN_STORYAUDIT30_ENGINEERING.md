# Story-audit full runner: synthetic engineering check

**MOCK_ONLY — these are software checks, not Nanbeige grades.** The unchanged
frozen STORYAUDIT30 main function completed on six deliberately constructed
episodes. No model was loaded, no pretrained forward ran, and no real generated
text or state file was inspected. The original NF4 worker remained live under
its same creation-bound process handle after the check.

The fixture covered one public CAL root, both worlds and all three semantic
arms. Four responses used the source oracle's canonical packet. One response
ended immediately at EOS, and one used a well-formed packet from the wrong
counterfactual world. All state arrays were synthetic zeros, explicitly marked
as mock artifacts in an ignored local directory. In particular, synthetic
host-gold replies to erased controls are not evidence that their inputs supply
the answer.

The actual runner performed its original source verification, package checks,
independent source oracle, pinned re-tokenization, packet grading, stopping and
history checks, archive validation, native-write identity, batch/padding checks,
missing-outcome aggregation and private Markdown/JSON generation. Only output
paths were rebound inside this isolated process; no frozen source was edited.

| Synthetic check | Observed result |
|---|---|
| Closed mock episodes / registered slots | 6 / 72; 66 unknown |
| Format-valid mock packets | 5 / 6 |
| Final-correct / jointly correct mock packets | 4 / 6 for each endpoint |
| Empty EOS | Preserved as malformed and incorrect |
| Well-formed wrong-world packet | Format accepted; final and joint correctness rejected |
| Prefixes and physical vectors checked | 261 prefixes; 10,440 uint16 BF16 vectors |
| Variable batch shape | Real reader checked B2-to-B1 synthetic EOS tails and padding bindings |
| DEV outcomes | All unknown; the fixture supplies no DEV observations |
| Generated aggregate hashes | All match the runner's immutable completion receipt |

All **72 public host-reference packets** also round-trip through the pinned
tokenizer and original grammar in **51–52 tokens**, within the unchanged
64-token budget. This is a syntax/output-budget check using known oracle
packets, including erased-control gold. It does not establish source-conditioned
model competence, greedy generation, faithful reasoning or correctness of any
real trace.

Maximum sampled private RAM for the entire construction and audit was
**131.965 MiB**, with at least **6.449 GiB system RAM available**. The inner
audit observed the same private maximum and at least **6.950 GiB available**.
One numerical thread and Below Normal priority; no Torch import or CUDA
allocation. The self-only watchdog retained the 0.75 GiB private cap, 4 GiB
system reserve and 5 GiB admission requirement. Two-second samples are not
continuous peak measurements. System counters reflect a shared machine.

The frozen audit seal remained
`8671e772f7166facca90ecd5fa74683a0908270fe7051e06b3c6fc45da482143`;
all frozen audit/dependency bindings were verified again afterward. Real
storyworld grading still waits for the registered block to close. The full
RAM28 scope, later canonical SAE/BPD audit, BF16 deferral, original failures
and absent final parameter verification remain unchanged. The original
168-case test stays sealed and the continuous research goal remains active.

Reproduce using the existing pinned tokenizer environment and local registered
source/assets; use a new ignored output directory each time:

```powershell
& $GPU_PYTHON -X utf8 -B -u -m mechinterp.nanbeige_bpd_gym.storyaudit30_engineering `
  --snapshot $PINNED_SNAPSHOT `
  --out-dir "runs\storyaudit30_main_MOCK_ONLY_fresh"
```

[Engineering harness](../../mechinterp/nanbeige_bpd_gym/storyaudit30_engineering.py) ·
[Sanitized receipt](nan_storyaudit30_engineering.json) ·
[Early-audit plan](STORYAUDIT30_PLAN.md) ·
[Frozen protocol](../../experiments/nan_storyaudit30/PROTOCOL.md)
