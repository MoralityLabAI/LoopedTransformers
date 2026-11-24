# Actual fixed SAE/BPD asset loading, CPU engineering

The unchanged CACHE32 fixed-asset loader successfully loaded the **actual**
CAL-trained dictionaries and qualified IO18 T2 BPD/controller banks for both
NF4 and BF16. This extends the earlier tiny synthetic host fixtures to real
asset files, without reading episodes or task gold or loading a model.

| Per precision | Verified loading evidence |
|---|---|
| SAE layers | 5,11,17,21;128 features at each layer |
| Fixed controls | Four matching random-initial dictionaries |
| Total dictionary arrays | 25,464,832 bytes across eight dictionaries |
| BPD/controller bank | 5,489,302 array bytes;rank8,two native phases |
| Existing qualified BPD fit | 16 actual optimizer updates; no new fitting |
| Asset pins | 11 hashes checked per precision |

Both loader results were real assets with `MOCK_ONLY=False`. Physical
3072-wide SAE coordinates,10752-by8 reads,8-by3072 writes, two-phase buffers,
finite values and positive SAE scales passed. The original target-byte
bindings remain NF4 `8e80e7d7…7663b3` and BF16 `cf3db880…d09186`.
This checks stored asset/receipt bindings; it does not establish a new model's
terminal integrity or numerical fidelity.

The separately frozen source was pushed at384cca69b84d3edee310eedf38de98bc4e15d7eb
before the one-shot attempt. Its seal is
`9984aa36785c5a0efd47a4824f137ffc1c69b1203867b44929d9c33297c3afc3`.
The unchanged STORYAUDIT30 guard ran with config/output globals redirected
only inside this engineering process. Limits were5 GiB available at admission,
4 GiB reserve,0.75 GiB private, one numerical thread, Below Normal priority,
two-second self-only watch and120-second loaded wall limit. Banks were loaded
serially and released between precisions. No frozen source or running model
was modified.

Across216 resource samples, maximum sampled private commitment was
**58.086 MiB** and minimum available system RAM was **4.999 GiB**. No Torch or
CUDA initialized; no model call, episode/gold read, SAE encoding, routing,
refit or task grade occurred. Per-array guards supplemented periodic samples;
the measurement is not a continuous memory-peak certificate.

This is asset-loading engineering with zero new source roots and no causal
contrast or physical dose. Causal/sham effect floors remain uncalibrated,
formal comparisons0. The complete matched cached CUDA and independent episode
audits remain required; the production post-model serialized host lease is
unchanged. Original168-case ARG03 test sealed; continuous goal active.

[Full shape/dtype/byte/hash metadata and resource provenance](nan_assets33_engineering.json) ·
[Prospective protocol](../../experiments/nan_assets33/PROTOCOL.md) ·
[Frozen cached comparison](NAN_CACHE32_PLAN.md)
