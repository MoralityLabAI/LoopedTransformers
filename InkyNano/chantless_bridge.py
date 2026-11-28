"""Expert-support-only import of Chantless v3 for the published InkyNano bank."""
import argparse
import base64
import hashlib
import json
import math
from pathlib import Path
import zlib

LAYERS, EXPERTS, TOP_K = 24, 128, 8
MODEL = "thinkingmachines/Inkling-2-Mini"
REVISION = "ed40e7da625f594fc9c4990f5f66be24873c22d7"


def read(path):
    raw = Path(path).read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def decode(spec):
    if isinstance(spec, list):
        ids = tuple(spec)
        if any(type(i) is not int or not 0 <= i < EXPERTS for i in ids):
            raise ValueError("invalid expert ID")
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate expert ID")
        return tuple(sorted(ids))
    if not isinstance(spec, dict) or spec.get("n") != EXPERTS:
        raise ValueError("expected a 128-bit expert bitmap")
    raw = base64.b64decode(spec["b64"], validate=True)
    enc = spec.get("enc", "packbits")
    if enc == "packbits+zlib":
        d = zlib.decompressobj()
        raw = d.decompress(raw, 17)
        if not d.eof or d.unused_data or d.unconsumed_tail:
            raise ValueError("invalid compressed bitmap")
    elif enc != "packbits":
        raise ValueError("unknown bitmap encoding")
    if len(raw) != 16:
        raise ValueError("wrong bitmap byte length")
    ids = tuple(i for i in range(EXPERTS) if raw[i // 8] & (128 >> (i % 8)))
    if type(spec.get("count")) is not int or spec["count"] != len(ids):
        raise ValueError("bitmap count mismatch")
    return ids


def selection(mask):
    source = mask.get("source", {})
    if mask.get("schema") != "chantless.mask/v3" or (source.get("repo"), source.get("revision")) != (MODEL, REVISION):
        raise ValueError("unexpected mask schema or source checkpoint")
    support = mask.get("experts") or {}
    if not isinstance(support, dict) or set(support) - {str(i) for i in range(LAYERS)}:
        raise ValueError("invalid physical layer IDs")
    rows = [decode(support[str(i)]) if str(i) in support else tuple(range(EXPERTS)) for i in range(LAYERS)]
    if len({len(row) for row in rows}) != 1 or len(rows[0]) < TOP_K:
        raise ValueError("need uniform support with at least eight experts per layer")
    return rows


def plan(mask, manifest, mask_sha, manifest_sha):
    if (manifest.get("source_model"), manifest.get("source_revision")) != (MODEL, REVISION):
        raise ValueError("bank checkpoint mismatch")
    if (manifest.get("bundle_format") != "gguf_v3_standalone_tensor_bundles"
            or manifest.get("num_layers") != LAYERS or manifest.get("total_experts") != LAYERS * EXPERTS):
        raise ValueError("expected the published full InkyNano bank")
    rows = selection(mask)
    payload = 0
    for layer, ids in enumerate(rows):
        for eid in ids:
            entry = manifest["experts"][str(layer)][str(eid)]
            n = entry["span_bytes"]
            if type(n) is not int or n <= 0:
                raise ValueError("invalid packed expert size")
            payload += n
    backbone = manifest["backbone"]["bytes"]
    if type(backbone) is not int or backbone <= 0:
        raise ValueError("invalid backbone size")
    return {"schema": "inkynano.chantless-support/v1", "source_model": MODEL,
            "source_revision": REVISION, "chantless_mask_sha256": mask_sha,
            "bank_manifest_sha256": manifest_sha, "resident_by_layer": rows,
            "capacity_per_layer": len(rows[0]), "active_per_token": TOP_K,
            "expert_span_bytes": payload, "backbone_bundle_bytes": backbone,
            "estimated_bundle_bytes": payload + backbone,
            "scope": "expert support only; original bank precision, FFN width and vocabulary"}


def route(scores, bias, allowed, resident=(), scale=8.0):
    """Reference for masked native ranking; original physical IDs, no cache substitution."""
    if len(scores) != EXPERTS or len(bias) != EXPERTS:
        raise ValueError("expected full native router coordinates")
    if any(not math.isfinite(s) or not 0 <= s <= 1 for s in scores) or any(not math.isfinite(b) for b in bias):
        raise ValueError("invalid sigmoid score or ranking bias")
    allowed = tuple(allowed)
    if len(allowed) < TOP_K or tuple(sorted(set(allowed))) != allowed or any(type(i) is not int or not 0 <= i < EXPERTS for i in allowed):
        raise ValueError("invalid support")
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("invalid scale")
    ranking = [s + b for s, b in zip(scores, bias)]
    if not all(map(math.isfinite, ranking)):
        raise ValueError("nonfinite rank")
    ids = tuple(sorted(allowed, key=lambda i: (-ranking[i], i))[:TOP_K])
    mass = sum(scores[i] for i in ids)
    if mass <= 0:
        raise ValueError("zero mixture mass")
    return {"ids": ids, "local_ids": tuple(allowed.index(i) for i in ids),
            "weights": tuple(scale * scores[i] / (mass + 1e-20) for i in ids),
            "miss_ids": tuple(i for i in ids if i not in resident)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mask", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    mask, mask_sha = read(a.mask)
    manifest, manifest_sha = read(a.manifest)
    result = plan(mask, manifest, mask_sha, manifest_sha)
    with a.out.open("x", encoding="utf8") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps({k: v for k, v in result.items() if k != "resident_by_layer"}, indent=2))


if __name__ == "__main__":
    main()
