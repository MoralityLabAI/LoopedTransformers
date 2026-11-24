"""Pinned safetensors metadata inventory; never intentionally fetches weights."""
import argparse
import hashlib
import json
import re
import struct
from pathlib import Path
from urllib.request import Request, urlopen

MODEL = "thinkingmachines/Inkling-2-Mini"
REV = "ed40e7da625f594fc9c4990f5f66be24873c22d7"
BASE = f"https://huggingface.co/{MODEL}/resolve/{REV}/"
LIMIT = 16 * 1024 * 1024
KEY = re.compile(r"^model\.llm\.layers\.(\d+)\.mlp\.experts\.(w13|w2)_weight$")
NAMES = ("config.json", "hf_quant_config.json", "model.safetensors.index.json")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def remote_small(name):
    with urlopen(BASE + name, timeout=45) as f:
        data = f.read(LIMIT + 1)
    if len(data) > LIMIT:
        raise ValueError("metadata limit exceeded")
    return data


def remote_range(name, start, size):
    if not 0 < size <= LIMIT:
        raise ValueError("invalid bounded range")
    end = start + size - 1
    req = Request(BASE + name, headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"})
    with urlopen(req, timeout=90) as f:
        match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", f.headers.get("Content-Range", ""))
        if f.status != 206 or not match or tuple(map(int, match.groups()[:2])) != (start, end):
            raise ValueError("server refused exact range; abort before reading weight body")
        data = f.read(size + 1)
        if len(data) != size:
            raise ValueError("incomplete range")
        return data, int(match.group(3))


def local_range(root, name, start, size):
    path = root / name
    with path.open("rb") as f:
        f.seek(start)
        data = f.read(size)
    if len(data) != size:
        raise ValueError("incomplete local range")
    return data, path.stat().st_size


def header(name, read):
    prefix, total = read(name, 0, 8)
    n = struct.unpack("<Q", prefix)[0]
    if not 2 <= n <= LIMIT or n + 8 > total:
        raise ValueError("invalid safetensors header size")
    raw, total2 = read(name, 8, n)
    if total2 != total:
        raise ValueError("inconsistent source size")
    return json.loads(raw), total - n - 8, digest(prefix + raw)


def audit(config, index, headers):
    text = config["text_config"]
    expected = {"hidden_size":2304, "intermediate_size":1152, "num_hidden_layers":24,
                "n_routed_experts":128, "num_experts_per_tok":8}
    for key, value in expected.items():
        if text[key] != value:
            raise ValueError(f"unexpected config {key}")
    loop = text["loop_config"]
    if (loop["core_start"], loop["core_end"], loop["num_passes"]) != (6,18,3):
        raise ValueError("unexpected loop")
    weight_map = index["weight_map"]
    entries, ranges, experts = {}, {}, {}
    for shard, (h, payload, _) in headers.items():
        spans = []
        for name, item in h.items():
            if name == "__metadata__":
                continue
            if weight_map.get(name) != shard:
                raise ValueError("unindexed or wrongly mapped tensor: " + name)
            lo, hi = item["data_offsets"]
            if not 0 <= lo <= hi <= payload:
                raise ValueError("bad tensor offsets: " + name)
            shape = item["shape"]
            if any(type(d) is not int or d < 0 for d in shape):
                raise ValueError("bad shape: " + name)
            spans.append((lo, hi))
            entries[name] = {"shard":shard, "shape":shape, "dtype":item["dtype"], "bytes":hi-lo}
            m = KEY.fullmatch(name)
            if m:
                experts[(int(m[1]), m[2])] = name
        cursor = 0
        for lo, hi in sorted(spans):
            if lo != cursor:
                raise ValueError("overlap or gap in payload")
            cursor = hi
        if cursor != payload:
            raise ValueError("unindexed payload bytes")
        ranges[shard] = payload
    if set(entries) != set(weight_map):
        raise ValueError("index/header coverage mismatch")
    if set(experts) != {(l,p) for l in range(24) for p in ("w13","w2")}:
        raise ValueError("unexpected expert projection coverage")
    expert_groups = []
    for (l,p), name in sorted(experts.items()):
        associated = {k:v for k,v in entries.items() if k == name or k.startswith(name+"_")}
        if not entries[name]["shape"] or entries[name]["shape"][0] != 128:
            raise ValueError("unexpected packed expert axis: "+name)
        if name+"_scale" not in associated:
            raise ValueError("missing expert scale: "+name)
        expert_groups.append({"layer":l,"projection":p,"tensors":associated})
    buckets = {}
    for name, value in entries.items():
        group = "expert_bank" if ".mlp.experts." in name else "text_other" if name.startswith("model.llm.") else "other_review_required"
        buckets[group] = buckets.get(group,0) + value["bytes"]
    return {"tensors":entries,"tensor_count":len(entries),"expert_groups":expert_groups,
            "payload_bytes":ranges,"bucket_bytes":buckets,
            "status":"metadata_only_native_parity_pending"}


def main():
    p = argparse.ArgumentParser()
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--remote", action="store_true")
    mode.add_argument("--local", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    small = {n:remote_small(n) if a.remote else (a.local/n).read_bytes() for n in NAMES}
    data = {n:json.loads(v) for n,v in small.items()}
    index = data[NAMES[2]]
    shards = sorted(set(index["weight_map"].values()))
    if any(Path(s).name != s or not s.endswith(".safetensors") for s in shards):
        raise ValueError("unsafe shard name")
    read = remote_range if a.remote else lambda name,start,size:local_range(a.local,name,start,size)
    headers = {s:header(s,read) for s in shards}
    out = audit(data[NAMES[0]],index,headers)
    out.update({"model":MODEL,"revision":REV,"metadata_sha256":{k:digest(v) for k,v in small.items()},
                "header_sha256":{k:v[2] for k,v in headers.items()},
                "quant_config":data[NAMES[1]]})
    a.output.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n",encoding="utf8")
    print(json.dumps({k:out[k] for k in ("status","tensor_count","bucket_bytes","payload_bytes")},indent=2))


if __name__ == "__main__":
    main()
