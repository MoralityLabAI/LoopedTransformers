# NAN-FIT-07 fresh-source fitting pilot

The source-only development pilot is frozen before its first pretrained
inference. Source manifest SHA256:
`8890f915d7b1ba05dc611dfb62e2c1d40abc2cc5bb71351e7b338f640b6f104e`.
No pretrained fit result is available at registration.

There are eight new CAL roots, eight new DEV roots and one explicit copy control
per root: 32 contexts per backend. Independent source checks cover all 64 graph
walks; no main prompt overlaps the previous public cohort. Tokenizer-only
validation passed with a maximum of 381 tokens. The original 168-case test was
not read, regenerated or used as an exclusion set.

The registered clean screen requires at least 6/8 main and 8/8 copy answers in
each split, independently for BF16 and NF4. This screen grades the first free
token of a two-loop native prefix. It preserves all cases and cannot authorize
donor transfer or change the original failed algebraic competence gate.

If a backend passes, it runs eight actual stock BPDTrainer steps for each of
T=1 VPD and T=2 BPD at layer-21 down_proj, rank 8. All four loss terms remain
active. Fixed final-checkpoint diagnostics compare component, static CAL mean,
clean-informed and reached-state routing with all-one closure, full-vocabulary
KL, physical write changes and unhooked teacher outputs. The learned factors
are then cross-referenced against the existing layer-21 SAE, initial SVD and
norm-matched Haar directions on every registered native context.

Engineering evidence uses random width-64 models, not pretrained Nanbeige:

| Backend | Depth | Actual optimizer steps | Worker path | CUDA peak allocated |
|---|---:|---:|---|---:|
| BF16 | 2 | 8 | Passed | 83,595,264 bytes |
| NF4 | 2 | 8 | Passed | 84,315,136 bytes |
| BF16 | 1 | 8 | Passed, including strict boundary closure | 77,579,264 bytes |

Each receipt binds the exact worker source used for that engineering version.
Four CPU contracts passed, including source correctness, case-preserving gate,
worker oracle isolation and NumPy replay of the actual pointwise controller.
The GPU queue waits for the actual Prime supervisor and model processes to
terminate before starting NF4, then BF16. CPU probes can run concurrently.

This is a fitting and measurement pilot with eight independent DEV roots,
zero formal inferential comparisons and no matched causal-dose/sham calibration.
Routing modes have different physical doses. It supports no causal transfer,
equivalence or mechanism confirmation claim. Rank 16, other layers, o_proj's
intervening-MLP bridge and donor-specific causal controls remain later work.

[Frozen protocol](../../experiments/nan_fit07/PROTOCOL.md),
[source registration](../../experiments/nan_fit07/SOURCE_MANIFEST.json),
[engineering receipts](../../experiments/nan_fit07/receipts).
