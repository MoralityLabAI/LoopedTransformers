"""Compose the published InkyNano bundles for the upstream Inkling2 Loop runtime."""
import argparse
import hashlib
import re
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np

from gym_manifest import sha
from matrix_gguf import LAYERS, TensorCopy, plan_array, write_packed_parts
from stage1_data import read_json, write_json
from stage1_memory import GIB, MemoryGuard


def within(root, path):
    file = (root / path).resolve()
    if not file.is_relative_to(root):
        raise ValueError("bundle path escapes its root")
    return file


def stage_backbone(gguf, model, root, manifest, selected, controls, guard):
    import torch

    source = within(root, manifest["path"])
    if source.stat().st_size != manifest["bytes"] or sha(source) != manifest["sha256"]:
        raise ValueError("backbone checksum mismatch")

    with source.open("rb") as backing, controls.open("wb") as converted:
        for tensor in manifest["tensors"]:
            guard.check()
            original = tensor["name"]
            filtered = model.filter_tensors((original, None))
            if not filtered:
                raise ValueError(f"unexpected tensor excluded from the text backbone: {original}")
            name = filtered[0]
            layer = re.search(r"\.layers\.(\d+)\.", name)
            bid = int(layer[1]) if layer else None
            shape = tuple(tensor["shape"])
            quant = gguf.GGMLQuantizationType[tensor["ggml_type"]]
            if quant in (gguf.GGMLQuantizationType.Q4_K, gguf.GGMLQuantizationType.Q6_K):
                bare = model.tensor_map.get_name(name)
                mapped = bare + ".weight" if bare is not None else model.map_tensor_name(name)
                disk_shape = gguf.quant_shape_to_byte_shape(shape, quant)
                yield TensorCopy(mapped, disk_shape, np.dtype("u1"), quant,
                                 ((source, tensor["file_offset"], tensor["nbytes"]),))
                continue
            if quant not in (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.BF16):
                raise ValueError(f"unsupported precision for backbone control: {original}")
            backing.seek(tensor["file_offset"])
            data = backing.read(tensor["nbytes"])
            if hashlib.sha256(data).hexdigest() != tensor["sha256"]:
                raise ValueError(f"control checksum mismatch: {original}")
            if quant == gguf.GGMLQuantizationType.BF16:
                # A BF16 value has an exact representation in F32.
                values = (np.frombuffer(data, dtype="<u2").astype(np.uint32) << 16).view(np.float32)
            else:
                values = np.frombuffer(data, dtype="<f4").copy()
            if bid is not None and (name.endswith(".mlp.gate.weight") or
                                    name.endswith(".mlp.gate.e_score_correction_bias")):
                values = values.reshape(shape)[selected[bid]]
            else:
                values = values.reshape(shape)
            for mapped, out in model.modify_tensors(torch.from_numpy(values), name, bid):
                array = out.detach().contiguous().numpy().astype("<f4", copy=False)
                offset = converted.tell()
                array.tofile(converted)
                yield TensorCopy(mapped, array.shape, array.dtype, gguf.GGMLQuantizationType.F32,
                                 ((controls, offset, array.nbytes),))


def stage_expert_layer(gguf, root, experts, layer, ids, guard):
    spans = {"gate_up": [], "down": []}
    layouts = {}
    for expert in ids:
        guard.check()
        entry = experts[str(layer)][str(expert)]
        source = within(root, entry["shard"])
        origin = 0
        if not source.is_file():
            if expert < 8:
                source = within(root, f"fixed-eight/layer{layer:02d}.bin")
                first = 0
            elif expert < 12:
                source = within(root, f"spans-008-012/layer{layer:02d}.bin")
                first = 8
            else:
                raise FileNotFoundError(source)
            origin = experts[str(layer)][str(first)]["span_file_offset"]
        begin = entry["span_file_offset"]
        with source.open("rb") as weights:
            weights.seek(begin - origin)
            blob = weights.read(entry["span_bytes"])
        if len(blob) != entry["span_bytes"]:
            raise ValueError(f"short isolated expert span for {layer}/{expert}")
        for projection in ("gate", "up", "down"):
            tensor = entry["tensors"][projection]
            offset, size = tensor["file_offset"] - begin, tensor["nbytes"]
            data = blob[offset:offset + size]
            if offset < 0 or len(data) != size or hashlib.sha256(data).hexdigest() != tensor["sha256"]:
                raise ValueError(f"expert checksum mismatch: {tensor['name']}")
            quant = gguf.GGMLQuantizationType[tensor["ggml_type"]]
            dims = tuple(tensor["shape"])
            if tuple(tensor["logical_shape"]) != dims:
                raise ValueError("padded expert projections require a separate layout")
            packed_shape = gguf.quant_shape_to_byte_shape(dims, quant)
            name = "down" if projection == "down" else "gate_up"
            if layouts.setdefault(name, (quant, packed_shape)) != (quant, packed_shape):
                raise ValueError("incompatible expert projections within a physical layer")
            spans[name].append((source, tensor["file_offset"] - origin, size))
    for name, (quant, packed_shape) in layouts.items():
        shape = (len(ids), packed_shape[0] * (2 if name == "gate_up" else 1), packed_shape[1])
        yield TensorCopy(f"blk.{layer}.ffn_{name}_exps.weight", shape, np.dtype("u1"),
                         quant, tuple(spans[name]))


def compose(args, guard=None):
    guard = guard or MemoryGuard()
    root = args.bank.resolve()
    path = root / "manifest.json"
    manifest = read_json(path)
    if (manifest.get("bundle_format") != "gguf_v3_standalone_tensor_bundles" or
            manifest.get("num_layers") != LAYERS or manifest.get("total_experts") != LAYERS * 128):
        raise ValueError("unexpected published bank layout")
    manifest_sha = sha(path)
    if args.manifest_sha256 and args.manifest_sha256 != manifest_sha:
        raise ValueError("unexpected weight-bank manifest checksum")
    selected = (np.broadcast_to(np.arange(128), (LAYERS, 128)) if args.unrestricted
                else plan_array(read_json(args.selection)["resident_by_layer"], tuple(range(8, 129))) if args.selection
                else np.broadcast_to(np.arange(8), (LAYERS, 8)))
    sys.path[:0] = [str(args.llama_source.resolve() / "gguf-py"), str(args.llama_source.resolve())]
    import gguf
    from conversion.inkling2 import Inkling2LoopModel

    args.out.parent.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    model = Inkling2LoopModel(args.source_metadata, gguf.LlamaFileType.MOSTLY_Q4_K_M, args.out)
    model.hparams["n_routed_experts"] = selected.shape[1]
    model.set_gguf_parameters()
    model.set_vocab()
    model.gguf_writer.add_name("InkyNano mixed quantized text")
    model.gguf_writer.add_string("inkynano.bank_manifest_sha256", manifest_sha)
    with TemporaryDirectory(prefix="inkynano-controls-", dir=args.out.parent) as scratch:
        controls = Path(scratch) / "controls.bin"
        parts = list(stage_backbone(gguf, model, root, manifest["backbone"], selected, controls, guard))
        if set(model._queries) != {"q", "kv"}:
            raise ValueError("incomplete cross-pass query controls")
        for layer, ids in enumerate(selected):
            parts.extend(stage_expert_layer(gguf, root, manifest["experts"], layer, ids, guard))
        size = write_packed_parts(model.gguf_writer, args.out, parts, args.max_model_bytes, guard)
    model_sha = sha(args.out)
    guard.check()
    receipt = {"bank_manifest_sha256": manifest_sha, "model_sha256": model_sha,
               "capacity_per_layer": selected.shape[1],
               "resident_by_layer": selected.tolist(), "bytes": size, "tensors": len(parts),
               "compose_seconds": perf_counter() - start, "memory": guard.receipt()}
    write_json(args.out.with_suffix(".receipt.json"), receipt)
    print(receipt["bytes"], "packed bytes", receipt["tensors"], "tensors")
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bank", type=Path, required=True)
    p.add_argument("--llama-source", type=Path, required=True)
    p.add_argument("--source-metadata", type=Path, required=True)
    p.add_argument("--manifest-sha256")
    scopes = p.add_mutually_exclusive_group()
    scopes.add_argument("--selection", type=Path)
    scopes.add_argument("--unrestricted", action="store_true")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-model-bytes", type=int, default=4 * GIB)
    compose(p.parse_args())


if __name__ == "__main__":
    main()
