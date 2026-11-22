# NEC-03V: original historical source archive attestation and 32D decoder recovery

**Status:** the original source weight provenance has been independently authenticated. Native/SAE activation caches and per-case TopK coefficients remain a separate gate. The scientific status is REDESIGN_OR_PILOT_ONLY.

## Immutable source archive identities

Original NEC03T necessity run: 44913605626. Original NEC03U root report and audit SHA256 pins remain in NEC03U_ARCHIVE_LOCK.json. The NEC03T report binds each arm to its historical native model weights, SAE checkpoint and independently audited source refit report.

| Original root | Source-model/SAE run | Source artifact ID | NEC03T artifact ID | Reset-native artifact ID |
| --- | ---: | ---: | ---: | ---: |
| dev5-000 | 44899750508 | 14705980493 | 14710962168 | 14705405217 |
| dev5-001 | 44899750508 | 14705925546 | 14710947309 | 14705385295 |
| dev5-002 | 44899750508 | 14704954383 | 14711101362 | 14704951781 |
| dev5-003 | 44899750508 | 14705281999 | 14710449539 | 14704318584 |
| dev5-004 | 44900628360 | 14705279661 | 14710478043 | included in source archive |
| dev5-005 | 44900628360 | 14706270227 | 14710554274 | included in source archive |
| dev5-006 | 44900628360 | 14706290195 | 14711161360 | included in source archive |
| dev5-007 | 44900628360 | 14706002643 | 14710609069 | included in source archive |

The standard-library-only attestor, nec03v_source_attest.py, enforces original world 0..7, registered source group 24..31, exactly 256 old NEC03T cases per arm, fixed-zero ineligible roots 3 and 5, original NEC03U byte-pinned NEC03T source and audit report, independently audited original NEC03O-A/B source receipts, original native reset1800 checkpoint byte hashes, original source-site roster hashes, and original SAE source checkpoint hashes where present.

Successful hosted provenance check: https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44983678150 . Eight original root jobs and the synthetic tamper tests passed with zero model forwards and zero protected-world reads.

## CPU-only frozen decoder-column reconstruction

The follow-up CPU implementation, nec03v_decoder_columns.py, loads only the original native/SAE checkpoint tensor dictionaries from those authenticated original bytes. Torch is used on CPU with weights_only=True. No native or SAE model object is instantiated, run, trained or evaluated.

For registered visit v, frozen native down-projection W_down of shape 32x64, frozen SAE decoder W_dec of shape 64x128, normalization scale s of size 64, and source-hub mask m_v of size 64, the projected output vector for original feature i is

    decoder32[v][i] = W_down @ (m_v * s * W_dec[:,i]).

The result is 128x32 projected columns per registered visit, one set per historical model arm. A separate implementation in nec03v_decoder_columns_audit.py recomputes every scalar as the direct sum of 64 terms, checking the recorded result against the same byte-authenticated original source checkpoints. This reconstructs *weights and physical geometry*, without evaluating the original teacher or producing task labels/logits.

The original root3/5 source sites and ineligible no-source cases stay in the fixed denominator. The CPU workflow is .github/workflows/nec03v-original-decoder-columns.yml, producing original-source physical-geometry receipts. The artifacts remain scientific mechanism *inputs*, rather than evidence of new necessity effects.

## Reproduce from the original archives

After restoring the matching original artifacts above into distinct source, NEC03T and reset directories:

    python -m mechinterp.neuron_communities.nec03v_source_attest --world-index 0 --source-root /data/source0 --nec03t-root /data/t0 --reset-root /data/reset0 --out /data/source-origin.json
    python -m mechinterp.neuron_communities.nec03v_decoder_columns --world-index 0 --source-root /data/source0 --nec03t-root /data/t0 --reset-root /data/reset0 --out /data/decoder32.json
    python -m mechinterp.neuron_communities.nec03v_decoder_columns_audit --world-index 0 --source-root /data/source0 --nec03t-root /data/t0 --reset-root /data/reset0 --columns /data/decoder32.json --out /data/decoder32-audit.json

For original root 4..7, the reset-native checkpoint is already included in the original source artifact, so omit the reset-root argument.

## Remaining source-only activation dependency

The original NEC03T case data store active feature *IDs* and aggregate output doses. The new coherent four-feature group study requires exactly 1,280 original source-fit active-code sets and 256 original source case positive TopK activation coefficient vectors per arm and visit.

Original archived source checkpoint tensors alone do not supply those per-example quantities. A separate pre-execution locked extraction path must establish provenance of the original clean 64D GELU inputs and their fixed source membership. If genuinely archived source activations are available, SAE code encoding from them is read-only CPU arithmetic. If no archived activations exist, collecting them from the frozen original native model requires **new source-only forward passes**, with its own separately registered access policy, original-source-only data manifest, frozen implementation bytes and independent verifier. None are authorized by this archive-attestation stage.

No new recruited world, development-evaluation, reserved, five-input embargoed or old four-input confirmation scoring is permitted. The original 69 source-eligible recruited roots still have zero new causal-sham outcomes, and the 59-root tolerance plan remains suspended. FSI-01 stays REDESIGN_OR_PILOT_ONLY.
