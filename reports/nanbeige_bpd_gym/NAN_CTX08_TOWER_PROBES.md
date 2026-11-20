# NAN-CTX-08 fixed SAE tower and learned VPD probes

Updated 2026-11-20T11:18:51.966790+00:00

These CPU probes apply the already fitted NAN-GYM-05 dictionaries and their random initial controls to every completed context at layers 5/11/17/21 and both loops. Each backend requires all 291 contexts, including all world pairs, positions, lengths, tasks and copy controls. No refitting or outcome selection is performed. The operation family was in the frozen CTX08 protocol; this implementation was written after NF4 behavior and partial BF16 behavior were exposed. It is an observational development analysis.

| Backend | Complete contexts | Fixed dictionaries | Learned VPD T1 asset |
|---|---:|---:|---|
| NF4 | 291 | 4, each with initial control | missing |
| BF16 | 291 | 4, each with initial control | passed |

All 128 feature activity fractions, normalized reconstruction errors, physical errors, loop/world displacements and initial-control counterparts are retained in the aggregate JSON for every registered group. Values are averaged within source root before averaging roots. The four DEV families are the units; the hundreds of variants, loops and lengths are repeated measures.

The learned-factor producer is FIT07 only. Its completed BF16 T1 checkpoint is applied to cached layer-21 down_proj inputs, with original SVD and matched-norm Haar controls and every SAE-feature/controller-energy root correlation. Loop 2 is outside T1 fitting depth. NF4 has no eligible fitted asset; FIT07 T2 failed. RECOMP09 is a separate execution amendment and is excluded from this original producer set.

Paired BF16/NF4 physical residual and loop-displacement geometry: complete at all four layers. Feature-index codes from independently fitted backend dictionaries are never subtracted. High cosine does not imply equal displacement or a shared causal mechanism. Residuals precede final loop RMSNorm; only down_proj has the immediate residual identity bridge.

No model inference, online intervention, causal test, equivalence claim or donor transfer is performed. Raw states, per-record codes, controller masks and IDs remain local. The original 168-case test stays sealed.

[Behavior and capacity](NAN_CTX08_PROGRESS.md) · [All aggregate probes](nan_ctx08_tower_probes.json) · [Original protocol](../../experiments/nan_ctx08/PROTOCOL.md)
