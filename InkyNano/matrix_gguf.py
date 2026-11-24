"""Pack one immutable, layerwise expert selection into a llama.cpp Inkling GGUF."""
import argparse
from contextlib import ExitStack
from dataclasses import dataclass
import re
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

from stage1_memory import MemoryGuard

ARCH = "inkling2_loop"
LAYERS, EXPERTS = 24, 128
BANK = re.compile(r"^blk\.(\d+)\.(ffn_(?:gate_inp|gate_up_exps|down_exps|up_exps|gate_exps)|exp_probs_b)\.(\w+)$")
REQUIRED = {"ffn_gate_inp", "ffn_gate_up_exps", "ffn_down_exps"}


@dataclass(frozen=True)
class TensorCopy:
    name: str
    shape: tuple
    dtype: np.dtype
    quant: object
    spans: tuple

    @property
    def size(self):
        return sum(size for _, _, size in self.spans)


def tensor_copy(source, tensor, indices=None, expert_count=EXPERTS):
    if indices is None:
        return TensorCopy(tensor.name, tensor.data.shape, tensor.data.dtype, tensor.tensor_type,
                          ((source, tensor.data_offset, tensor.n_bytes),))
    stride, remainder = divmod(tensor.n_bytes, expert_count)
    if remainder:
        raise ValueError(f"expert slices are not equally sized: {tensor.name}")
    return TensorCopy(tensor.name, (len(indices), *tensor.data.shape[1:]), tensor.data.dtype,
                      tensor.tensor_type,
                      tuple((source, tensor.data_offset + int(i) * stride, stride) for i in indices))


def plan_array(layers, capacities=(8, 16, 32)):
    selected = np.asarray(layers)
    if (selected.ndim != 2 or selected.shape[0] != LAYERS or selected.shape[1] not in capacities
            or not np.issubdtype(selected.dtype, np.integer)
            or np.any(selected < 0) or np.any(selected >= EXPERTS)
            or np.any(np.diff(selected, axis=1) <= 0)):
        raise ValueError(f"need 24 increasing, in-range lists of expert IDs at capacity {capacities}")
    return selected


@dataclass(frozen=True)
class PublishedComposition:
    model_sha256: str
    manifest_sha256: str
    available: tuple

    @classmethod
    def read(cls, path):
        from stage1_data import read_json

        receipt = read_json(path)
        model_hash = receipt["model_sha256"]
        bank_hash = receipt["bank_manifest_sha256"]
        ids = plan_array(receipt["resident_by_layer"], tuple(range(8, EXPERTS + 1)))
        if (receipt["capacity_per_layer"] != ids.shape[1] or
                any(not re.fullmatch(r"[0-9a-f]{64}", digest) for digest in (model_hash, bank_hash))):
            raise ValueError("invalid published composition provenance")
        return cls(model_hash, bank_hash, tuple(map(tuple, ids.tolist())))

    def expand_mass(self, phases):
        for phase in phases.values():
            mass = np.asarray(phase["mass"], dtype=np.float64)
            width = len(self.available[0])
            if mass.shape != (LAYERS, EXPERTS) or np.any(mass[:, width:]):
                raise ValueError("native routing contains an ID outside the composed teacher")
            expanded = np.zeros_like(mass)
            np.put_along_axis(expanded, np.asarray(self.available), mass[:, :width], axis=1)
            phase["mass"] = expanded.tolist()


def validate_inkling_header(reader, expert_count=EXPERTS):
    fields = reader.fields
    expect = {"general.architecture": ARCH, f"{ARCH}.block_count": LAYERS,
              f"{ARCH}.expert_count": expert_count, f"{ARCH}.expert_used_count": 8}
    for name, value in expect.items():
        if name not in fields or fields[name].contents() != value:
            raise ValueError(f"incompatible Inkling header field {name}")
    if any(key.startswith("split.") for key in fields):
        raise ValueError("prompt packing requires a single-file source GGUF")


def pack(gguf, source, target, layers, max_bytes, guard=None, composition=None):
    selection = plan_array(layers)
    capacity = selection.shape[1]
    expert_count = EXPERTS
    if composition:
        expert_count = len(composition.available[0])
        selection = np.asarray([[ids.index(expert) for expert in layer]
                                for ids, layer in zip(composition.available, selection, strict=True)])
    guard = guard or MemoryGuard()
    started = perf_counter()
    reader = gguf.GGUFReader(source, "r")
    validate_inkling_header(reader, expert_count)
    if composition and reader.fields["inkynano.bank_manifest_sha256"].contents() != composition.manifest_sha256:
        raise ValueError("composed GGUF identifies a different published bank")
    parts = []
    names = {layer: set() for layer in range(LAYERS)}
    expert_bytes = 0
    for tensor in reader.tensors:
        original = tensor.data.shape
        match = BANK.fullmatch(tensor.name)
        indices = None
        if match:
            layer, bank, suffix = int(match[1]), match[2], match[3]
            if not 0 <= layer < LAYERS:
                raise ValueError(f"unexpected physical layer {tensor.name}")
            if original[0] == expert_count:
                indices = selection[layer]
                if suffix == "weight":
                    names[layer].add(bank)
            elif (suffix == "weight" and bank in REQUIRED) or suffix == "bias":
                raise ValueError(f"invalid axis for expert tensor {tensor.name}: {original}")
        elif re.search(r"(?:_exps|exp_probs_b|ffn_gate_inp)", tensor.name):
            raise ValueError(f"unknown layout for expert tensor {tensor.name}")
        part = tensor_copy(source, tensor, indices, expert_count)
        if indices is not None and "_exps" in tensor.name and tensor.name.endswith(".weight"):
            expert_bytes += part.size
        parts.append(part)
    if any(not REQUIRED.issubset(found) for found in names.values()):
        raise ValueError("missing native fused expert or router tensor")
    size = write_compact(gguf, reader, target, capacity, parts, max_bytes, guard)
    return {"expert_weight_bytes": expert_bytes, "compact_gguf_bytes": size,
            "pack_seconds": perf_counter() - started}


def write_compact(gguf, reader, target, capacity, parts, max_bytes, guard):
    writer = gguf.GGUFWriter(None, ARCH, endianess=reader.endianess)
    for key, field in reader.fields.items():
        if key == "general.architecture" or key.startswith("GGUF.") or not field.types:
            continue
        value = capacity if key == f"{ARCH}.expert_count" else field.contents()
        if key == "general.alignment":
            writer.data_alignment = value
        writer.add_key_value(key, value, field.types[0],
                             field.types[1] if len(field.types) > 1 else None)
    return write_packed_parts(writer, target, parts, max_bytes, guard)


def write_packed_parts(writer, target, parts, max_bytes, guard):
    temporary = target.with_suffix(target.suffix + ".part")
    if target.exists():
        raise FileExistsError(target)
    if temporary.exists():
        raise FileExistsError(temporary)
    try:
        for part in parts:
            writer.add_tensor_info(part.name, part.shape, part.dtype, part.size, part.quant)
        writer.write_header_to_file(path=temporary)
        writer.write_kv_data_to_file()
        writer.write_ti_data_to_file()
        writer.close()
        align = writer.data_alignment
        header = temporary.stat().st_size
        expected = header + (-header % align) + sum(part.size + (-part.size % align) for part in parts)
        if expected > max_bytes:
            raise MemoryError("compact GGUF file exceeds RAM model budget")
        with temporary.open("ab") as output, ExitStack() as input_file:
            output.write(bytes(-header % align))
            current_source = None
            for part in parts:
                for source, offset, size in part.spans:
                    if source != current_source:
                        input_file.close()
                        weights = input_file.enter_context(source.open("rb"))
                        current_source = source
                    weights.seek(offset)
                    while size:
                        chunk = weights.read(min(size, 4 << 20))
                        if not chunk:
                            raise IOError(f"short tensor read: {part.name}")
                        output.write(chunk)
                        size -= len(chunk)
                    guard.check()
                output.write(bytes(-part.size % align))
        if temporary.stat().st_size != expected:
            raise IOError("compact GGUF size differs from planned byte spans")
        temporary.replace(target)
    except BaseException:
        writer.close()
        temporary.unlink(missing_ok=True)
        raise
    return expected


def main():
    import json

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gguf-python", type=Path, required=True)
    sources = p.add_mutually_exclusive_group(required=True)
    sources.add_argument("--expert-bundle", type=Path)
    sources.add_argument("--model", type=Path)
    p.add_argument("--selection", type=Path, required=True, help="one plans.json entry")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-model-bytes", type=int, required=True)
    a = p.parse_args()
    sys.path.insert(0, str(a.gguf_python))
    import gguf

    selection = json.loads(a.selection.read_text(encoding="utf8"))
    layers = selection["resident_by_layer"]
    if a.expert_bundle:
        from matrix_expert_bank import ExpertBank

        bank = ExpertBank(a.expert_bundle)
        if selection.get("expert_manifest_sha256") not in (None, bank.manifest_sha256):
            raise ValueError("selection identifies a different isolated expert manifest")
        print(bank.pack(gguf, a.out, layers, a.max_model_bytes, MemoryGuard()))
    else:
        print(pack(gguf, a.model, a.out, layers, a.max_model_bytes))


if __name__ == "__main__":
    main()
