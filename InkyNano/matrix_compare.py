"""Summarize matched matrix measurements and optional Goose storyworld receipts."""
import argparse
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

from matrix_cases import jsonl
from stage1_data import read_json, write_json


REPLAY_SETTINGS = (
    "case_file_sha256", "model_sha256", "runtime_revision", "expert_manifest_sha256",
    "context", "threads", "batch_tokens", "sampler", "max_new_override", "parse_special",
    "rss_limit_bytes", "capacity", "teacher_in_ram", "plans_sha256", "controller_checkpoint_sha256",
)


def index(rows, policy=None):
    result = {}
    for row in rows:
        key = row["case_id"], policy or row["policy"]
        if key in result:
            raise ValueError(f"duplicate benchmark result {key}")
        result[key] = row
    return result


def compare_backends(cpu_rows, cuda_rows, cpu_spec, cuda_spec):
    for specs, gpu in ((cpu_spec, False), (cuda_spec, True)):
        layers = specs.get("n_gpu_layers", -1)
        if (specs.get("command") != "benchmark" or layers < 0 or bool(layers) != gpu
                or not specs.get("runtime_binary_sha256")):
            raise ValueError("expected CPU and CUDA benchmark specifications in that order")
    different = [field for field in REPLAY_SETTINGS
                 if field not in cpu_spec or field not in cuda_spec or cpu_spec[field] != cuda_spec[field]]
    if cpu_spec.get("published_composition_sha256") != cuda_spec.get("published_composition_sha256"):
        different.append("published_composition_sha256")
    if different:
        raise ValueError(f"CPU/CUDA replay settings differ: {', '.join(different)}")

    cpu, cuda = index(cpu_rows), index(cuda_rows)
    paired = []
    for key in sorted(cpu.keys() & cuda.keys()):
        host, device = cpu[key], cuda[key]
        identity = ("root", "task", "mode", "prompt_sha256", "model_sha256", "runtime_revision",
                    "expert_manifest_sha256", "capacity", "storage_mode", "quality_evaluated")
        different = [field for field in identity if host[field] != device[field]]
        for field in ("model_sha256", "runtime_revision", "expert_manifest_sha256"):
            if host[field] != cpu_spec[field] and field not in different:
                different.append(field)
        if different:
            raise ValueError(f"CPU/CUDA receipt identity differs for {key}: {', '.join(different)}")
        if ("cuda" in host or device.get("cuda", {}).get("gpu_layers_requested") != cuda_spec["n_gpu_layers"]
                or device["cuda"].get("vram_cap_bytes") != cuda_spec.get("vram_limit_bytes")):
            raise ValueError(f"backend telemetry does not match the specifications for {key}")
        metrics = ("input_tokens", "output_tokens", "prefill_tps", "decode_tps", "end_to_end_seconds",
                   "peak_rss_bytes", "stop", "quality")
        same_counts = all(host[field] == device[field] for field in ("input_tokens", "output_tokens"))
        result = {field: host[field] for field in ("case_id", "policy", "root", "task", "mode",
                                                   "capacity", "prompt_sha256", "quality_evaluated")}
        result.update(cpu={field: host[field] for field in metrics}, cuda={field: device[field] for field in metrics},
                      cuda_offload=device["cuda"], same_token_counts=same_counts,
                      same_output=host["content"] == device["content"])
        if same_counts and device["end_to_end_seconds"] > 0:
            result["cpu_to_cuda_wall_time_ratio"] = host["end_to_end_seconds"] / device["end_to_end_seconds"]
        paired.append(result)
    return {
        "cpu_spec": cpu_spec, "cuda_spec": cuda_spec, "paired": paired,
        "cpu_only": [list(key) for key in sorted(cpu.keys() - cuda.keys())],
        "cuda_only": [list(key) for key in sorted(cuda.keys() - cpu.keys())],
    }


def historical_storyworld(receipt):
    if receipt["benchmark"] != "hilbert-storyworld-canvas-v1":
        raise ValueError("expected a Hilbert Storyworld Canvas aggregate receipt")
    totals = receipt["conveyor_16k_turn_totals"]
    prompt_tokens = totals["total_prompt_tokens_in"]
    cached_tokens = totals["total_cached_prompt_tokens"]
    if prompt_tokens <= 0 or not 0 <= cached_tokens <= prompt_tokens or (
        cached_tokens + totals["total_uncached_prompt_tokens"] != prompt_tokens
    ):
        raise ValueError("inconsistent historical Goose prompt-token totals")
    browser = receipt["canvas_edition"]["browser"]
    return {
        "benchmark": receipt["benchmark"], "storyworld": receipt["storyworld"],
        "run_id": receipt["run_id"], "runtime": receipt["runtime"],
        "conveyor_16k": {**totals, "cached_prompt_token_fraction": cached_tokens / prompt_tokens},
        "planning": [{"target_prompt_tokens": int(size), **metrics}
                     for size, metrics in sorted(receipt["sized_long_context_planning"].items(),
                                                 key=lambda item: int(item[0]))],
        "verified_canvas": {"status": browser["status"], "visited_scenes": browser["visited_scenes"],
                            "verified_secret_endings": browser["verified_secret_endings"]},
        "comparison_scope": "Historical authoring workflow with its original model, device and prompt cache",
    }


def compare(runtime, historical):
    paired = []
    same_prompt_goose = []
    differences = defaultdict(list)
    records = index(runtime)
    goose = index(historical, policy="goose")
    native_kind = "teacher_subset" if any(key[1] == "teacher_subset" for key in records) else "native"
    for (case, policy), row in records.items():
        if policy in ("native", "teacher_subset"):
            continue
        baselines = [(native_kind, records.get((case, native_kind))),
                     ("goose", goose.get((case, "goose")))]
        if policy in ("ldt", "trm"):
            baselines.append(("global", records.get((case, "global"))))
        for kind, other in baselines:
            if not other:
                differences[kind, "missing"].append(case)
                continue
            # A cross-runtime pairing requires the same underlying model bytes and exact request.
            identity = ("root", "task", "mode", "prompt_sha256", "model_sha256")
            if kind == "global":
                identity += ("capacity", "runtime_revision", "expert_manifest_sha256")
            mismatch = [field for field in identity if row.get(field) != other.get(field)]
            historical_context = kind == "goose" and mismatch == ["model_sha256"]
            if mismatch:
                differences[kind, "different " + ", ".join(mismatch)].append(case)
                if not historical_context:
                    continue
            entry = {"case_id": case, "root": row["root"], "task": row["task"],
                     "mode": row["mode"], "policy": policy, "baseline": kind,
                     "capacity": row["capacity"], "same_input_token_count":
                     row["input_tokens"] == other.get("input_tokens"),
                     "same_output_token_count": row["output_tokens"] == other.get("output_tokens")}
            for metric in ("prefill_tps", "decode_tps", "end_to_end_seconds", "peak_rss_bytes"):
                if isinstance(row.get(metric), (float, int)) and isinstance(other.get(metric), (float, int)):
                    entry[f"{metric}_delta"] = row[metric] - other[metric]
            for metric in ("correct", "exact_grid_match", "schema_keys_present", "storyworld_structure_ok"):
                score, control = row.get("quality", {}).get(metric), other.get("quality", {}).get(metric)
                if type(score) is type(control) is bool:
                    entry[f"{metric}_delta"] = int(score) - int(control)
            if historical_context:
                entry.update(matrix_model_sha256=row["model_sha256"], goose_model_sha256=other.get("model_sha256"))
                same_prompt_goose.append(entry)
            else:
                paired.append(entry)

    groups = defaultdict(list)
    for row in runtime:
        groups[row["task"], row["mode"], row["policy"], row["capacity"]].append(row)
    summaries = []
    for (task, mode, policy, capacity), rows in sorted(groups.items()):
        summary = {
            "task": task, "mode": mode, "policy": policy, "capacity": capacity,
            "requests": len(rows), "roots": len({r["root"] for r in rows}),
        }
        for key in ("prefill_tps", "decode_tps", "pack_seconds", "selection_seconds",
                    "load_seconds", "end_to_end_seconds", "peak_rss_bytes"):
            values = [r[key] for r in rows if key in r]
            if values:
                summary[f"median_{key}"] = median(values)
        cuda = [row["cuda"] for row in rows if "cuda" in row]
        if cuda:
            summary["median_planned_vram_bytes"] = median(row["planned_vram_bytes"] for row in cuda)
            summary["max_sampled_vram_delta_bytes"] = max(row["sampled_peak_device_free_delta_bytes"] for row in cuda)
        for field in ("correct", "exact_grid_match", "parse_ok", "schema_keys_present",
                      "storyworld_structure_ok", "storyworld_paragraph_target_ok"):
            per_root = defaultdict(list)
            for row in rows:
                value = row.get("quality", {}).get(field)
                if type(value) is bool:
                    per_root[row["root"]].append(value)
            if per_root:
                summary[f"root_mean_{field}"] = mean(mean(scores) for scores in per_root.values())
        summaries.append(summary)
    contrasts = defaultdict(list)
    for row in paired:
        key = tuple(row[field] for field in ("task", "mode", "policy", "capacity", "baseline"))
        contrasts[key].append(row)
    paired_groups = []
    for (task, mode, policy, capacity, baseline), rows in sorted(contrasts.items()):
        result = {
            "task": task, "mode": mode, "policy": policy, "capacity": capacity,
            "baseline": baseline, "requests": len(rows), "roots": len({r["root"] for r in rows}),
        }
        fields = sorted(set().union(*(r.keys() for r in rows)))
        for field in (f for f in fields if f.endswith("_delta")):
            per_root = defaultdict(list)
            for row in rows:
                if field in row:
                    per_root[row["root"]].append(row[field])
            result[f"root_mean_{field}"] = mean(mean(values) for values in per_root.values())
            result[f"roots_with_{field}"] = len(per_root)
        paired_groups.append(result)
    return {
        "groups": summaries, "paired": paired, "paired_groups": paired_groups,
        "same_prompt_goose": same_prompt_goose,
        "unmatched": [{"baseline": kind, "reason": why, "case_ids": sorted(set(ids))}
                      for (kind, why), ids in sorted(differences.items())],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--matrix", type=Path, required=True)
    p.add_argument("--cpu-baseline", type=Path, help="matching CPU metrics for a CUDA matrix, with sibling spec.json")
    p.add_argument("--goose-baseline", type=Path)
    p.add_argument("--goose-receipt", type=Path, help="historical MCP-05 Storyworld aggregate receipt")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    rows = list(jsonl(a.matrix))
    outcome = compare(rows, list(jsonl(a.goose_baseline)) if a.goose_baseline else [])
    if a.cpu_baseline:
        outcome["matched_cpu_cuda"] = compare_backends(list(jsonl(a.cpu_baseline)), rows,
                                                       read_json(a.cpu_baseline.with_name("spec.json")),
                                                       read_json(a.matrix.with_name("spec.json")))
    if a.goose_receipt:
        outcome["historical_goose"] = historical_storyworld(read_json(a.goose_receipt))
    write_json(a.out, outcome)
    backend_count = len(outcome.get("matched_cpu_cuda", {}).get("paired", []))
    print(f"{len(outcome['groups'])} metric groups, {len(outcome['paired'])} policy/baseline pairs, "
          f"{backend_count} CPU/CUDA pairs")


if __name__ == "__main__":
    main()
