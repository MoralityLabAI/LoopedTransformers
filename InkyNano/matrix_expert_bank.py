"""Assemble compact ranks from an indexed set of already isolated packed experts."""
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from gym_manifest import sha
from matrix_gguf import (BANK, EXPERTS, LAYERS, TensorCopy, plan_array, tensor_copy,
                         validate_inkling_header, write_compact)
from stage1_data import read_json

FORMAT = "inky-isolated-gguf-v1"
SHA256 = re.compile(r"[0-9a-f]{64}")
PROJECTIONS = ("ffn_gate_up_exps", "ffn_down_exps")


@dataclass(frozen=True)
class Artifact:
    path: Path
    size: int
    digest: str


@dataclass(frozen=True)
class Shard:
    artifact: Artifact
    layer: int
    expert_ids: tuple


class ExpertBank:
    """A normalized local index. A Silico export can be indexed without copying its weights."""

    def __init__(self, manifest):
        self.manifest_sha256 = sha(manifest)
        root = manifest.resolve().parent
        spec = read_json(manifest)
        if spec.get("format") != FORMAT or not SHA256.fullmatch(spec.get("source_model_sha256", "")):
            raise ValueError("expected an isolated expert manifest with original GGUF SHA-256")
        self.model_sha256 = spec["source_model_sha256"]
        self.backbone = self._artifact(root, spec["backbone"])
        self.shards = {}
        self.verified = {}
        for row in spec["shards"]:
            layer, ids = row["layer"], tuple(row["expert_ids"])
            if (type(layer) is not int or not 0 <= layer < LAYERS or not ids
                    or any(type(i) is not int or not 0 <= i < EXPERTS for i in ids)
                    or len(set(ids)) != len(ids)):
                raise ValueError("invalid shard layer or expert IDs")
            shard = Shard(self._artifact(root, row), layer, ids)
            for position, eid in enumerate(ids):
                key = layer, eid
                if key in self.shards:
                    raise ValueError(f"duplicate physical expert {key}")
                self.shards[key] = shard, position
        self.available = tuple(tuple(eid for eid in range(EXPERTS) if (layer, eid) in self.shards)
                               for layer in range(LAYERS))
        if any(len(ids) < 8 for ids in self.available):
            raise ValueError("every physical layer needs at least eight isolated experts")

    @staticmethod
    def _artifact(root, item):
        relative = Path(item["path"])
        path = (root / relative).resolve()
        size, digest = item["bytes"], item["sha256"]
        if (relative.is_absolute() or not path.is_relative_to(root) or type(size) is not int
                or size <= 0 or not SHA256.fullmatch(digest)):
            raise ValueError(f"invalid indexed file: {relative}")
        return Artifact(path, size, digest)

    def verify(self, artifact):
        stat = artifact.path.stat()
        signature = stat.st_size, stat.st_mtime_ns
        if signature != self.verified.get(artifact):
            if stat.st_size != artifact.size or sha(artifact.path) != artifact.digest:
                raise ValueError(f"isolated artifact integrity mismatch: {artifact.path.name}")
            self.verified[artifact] = signature

    def _project(self, gguf, reader, selection, guard):
        requests = defaultdict(list)
        for layer, ids in enumerate(selection):
            for eid in ids:
                if (layer, int(eid)) not in self.shards:
                    raise ValueError(f"selected expert ({layer}, {eid}) is absent from the bank")
                shard, position = self.shards[layer, int(eid)]
                requests[shard].append((int(eid), position))
        pieces = defaultdict(dict)
        layouts = {}
        for shard, wanted in requests.items():
            guard.check()
            self.verify(shard.artifact)
            source = gguf.GGUFReader(shard.artifact.path, "r")
            if source.endianess != reader.endianess:
                raise ValueError("incompatible shard endianness")
            required = {f"blk.{shard.layer}.{name}.weight" for name in PROJECTIONS}
            found = set()
            for tensor in source.tensors:
                match = BANK.fullmatch(tensor.name)
                if not match or int(match[1]) != shard.layer:
                    continue
                if "_exps" not in match[2]:
                    continue
                if match[2] not in PROJECTIONS:
                    raise ValueError(f"unsupported isolated projection {tensor.name}")
                count = len(shard.expert_ids)
                # A single-expert projection can retain the leading axis or omit it.
                squeezed = count == 1 and match[3] == "weight" and len(tensor.shape) == 2
                shape = tensor.data.shape if squeezed else tensor.data.shape[1:]
                stride, remainder = divmod(tensor.n_bytes, count)
                if remainder or (not squeezed and tensor.data.shape[0] != count):
                    raise ValueError(f"shard expert axis does not match its index: {tensor.name}")
                layout = shape, tensor.data.dtype, tensor.tensor_type, stride
                if layouts.setdefault(tensor.name, layout) != layout:
                    raise ValueError(f"isolated expert layout differs across shards: {tensor.name}")
                for eid, pos in wanted:
                    pieces[tensor.name][eid] = (shard.artifact.path, tensor.data_offset + pos * stride, stride)
                found.add(tensor.name)
            if not required <= found:
                raise ValueError(f"missing fused projections for physical layer {shard.layer}")
        result = []
        for name, (shape, dtype, quant, _) in sorted(layouts.items()):
            layer = int(name.split(".")[1])
            ids = selection[layer].tolist()
            if set(pieces[name]) != set(ids):
                raise ValueError(f"selected experts have inconsistent tensor coverage: {name}")
            result.append(TensorCopy(name, (len(ids), *shape), dtype, quant,
                                     tuple(pieces[name][eid] for eid in ids)))
        return result

    def pack(self, gguf, target, selection, max_bytes, guard):
        """Read the backbone and selected shard bytes; no original full GGUF is needed."""
        started = perf_counter()
        selected = plan_array(selection)
        self.verify(self.backbone)
        reader = gguf.GGUFReader(self.backbone.path, "r")
        validate_inkling_header(reader)
        parts = []
        router_layers = set()
        for tensor in reader.tensors:
            match = BANK.fullmatch(tensor.name)
            indices = None
            if match:
                layer, bank, suffix = int(match[1]), match[2], match[3]
                if not 0 <= layer < LAYERS:
                    raise ValueError(f"unexpected router layer: {tensor.name}")
                if "_exps" in bank:
                    raise ValueError(f"backbone includes an expert tensor: {tensor.name}")
                if tensor.data.shape[0] == EXPERTS:
                    indices = selected[layer]
                    if bank == "ffn_gate_inp" and suffix == "weight":
                        router_layers.add(layer)
                elif suffix in ("weight", "bias"):
                    raise ValueError(f"invalid native router axis: {tensor.name}")
            elif re.search(r"(?:_exps|exp_probs_b|ffn_gate_inp)", tensor.name):
                raise ValueError(f"unknown backbone expert tensor: {tensor.name}")
            parts.append(tensor_copy(self.backbone.path, tensor, indices))
        if len(router_layers) != LAYERS:
            raise ValueError("backbone needs the original router weight for every physical layer")
        packed = self._project(gguf, reader, selected, guard)
        size = write_compact(gguf, reader, target, selected.shape[1], parts + packed, max_bytes, guard)
        return {
            "expert_weight_bytes": sum(t.size for t in packed if t.name.endswith(".weight")),
            "compact_gguf_bytes": size,
            "pack_seconds": perf_counter() - started,
            "expert_manifest_sha256": self.manifest_sha256,
        }
