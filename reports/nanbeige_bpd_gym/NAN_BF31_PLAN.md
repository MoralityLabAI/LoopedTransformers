# BF16 full control under the host-RAM policy

The next full BF16 control is frozen before inference. Its source seal is
`15525236bbae38f664480bb17c17c77665887159ce8b6e8e31ff02b1d0ed944e`. It waits for the current RAM28 NF4 model,
canonical host and bounded story audit to exit; it starts no second model.

The full source roster is unchanged: **1,549 baselines and 216 actual batched
A/B pairs**. The loader keeps four decoder layers and the BF16 head on GPU,
uses one reusable GPU stage and file-backed CPU embeddings/reference weights,
and leaves other decoder parameters on meta. It verifies every source-to-stage
parameter byte before the first forward. Native/head/row/all-one closure and
B1x9216 plus padded B2x4608 capacity remain required. No old failed gate is
reopened or missing outcome replaced.

RAM admission is **5 GiB available**. Between captures: **2 GiB reserve and
6 GiB private cap**. The authenticated supervisor's self-worker stop is
**1 GiB reserve or 6.5 GiB private**. One numerical thread, Below Normal;
substantial SAE/BPD host work follows model exit with a 1.5 GiB private cap.
The independent math process is untouched.

The actual pinned-class small-model assembler previously passed 22 CPU
captures and four BPD routes byte exactly. The completed BF31 host/report path
passed **24 CPU checks**, independently graded **11 synthetic BF16 episodes**,
used **11 synthetic NF4 audits** from their separate location, checked **14
complete-batch geometry prefixes** and **six A/B groups**, and retained all
**3,962 coverage slots** with **3,940 unknown** in that fixture. Maximum sampled
private RAM was **359.805 MiB**, minimum system available RAM **5.403 GiB**.
This is **MOCK_ONLY engineering**, with no pretrained forward or CUDA, and
supplies no new real-model score or full-CUDA memory forecast.

The live NF4 condition keeps its original source seal, artifacts, grades and
normal receipt or absence. Future geometry requires identical complete audited
decoder-batch inputs, including companion histories/modes/order/padding.
Missing companions and unequal tails remain excluded and counted. Root means
precede descriptive aggregation. All scientific floors and failures remain;
formal comparisons 0, REDESIGN_OR_PILOT_ONLY. The 168-case test stays sealed and
the continuous goal stays active.

Prepublication registration encountered an unrelated omitted file in the
sparse BPD checkout and then a fresh log-directory initialization issue. Both
unpublished attempts and source bytes are retained locally. The corrected
condition binds the actual pinned framework files and passed the full fixture
again before publication; there were no BF31 pretrained outcomes or changes
to scientific criteria.

[Protocol and all safeguards](../../experiments/nan_bf31/PROTOCOL.md) ·
[Source seal](../../experiments/nan_bf31/SOURCE_MANIFEST.json) ·
[Sanitized synthetic engineering receipt](../../experiments/nan_bf31/ENGINEERING.json) ·
[Complete current coverage and unknowns](NAN_BF31_PROGRESS.md) ·
[Live NF4 condition](NAN_RAM28_PROGRESS.md) ·
[Validated assembler](LAZY_BF16_BACKEND_INTEGRATION.md)
