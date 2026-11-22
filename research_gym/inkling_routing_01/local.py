"""INK-ROUTE-01 fixed-loop, local Transformers development pilot."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path
from threading import Event, Thread

import numpy as np

from .checkpoint import (
    FILE_HASHES, MODEL_ID, MODEL_REVISION, TRANSFORMERS_REVISION, TRANSFORMERS_VERSION,
    WEIGHTS, file_sha256, load_model, verify_metadata,
)
from .core import (
    ARMS, HF_READER_LAYER, TARGET_COORD, WIDTH, patch_vector, reconstruct_reader,
    rms_normalize, sham_coordinate, vector_metrics,
)
from .hooks import WRITER, ReaderTrace, as_numpy, reader_call

SCHEMA = "ink-route-01.native-dev.v1"
PROMPTS = {
    "capital": "The capital of France is",
    "geometry": "A triangle has exactly three",
    "story": "The sailor returned to the harbor after",
    "language": "The opposite of earlier is",
    "arithmetic": "Seven plus six equals",
    "agreement": "The musicians tuned their instruments before they",
    "instructions": "To unlock the door, she inserted the key into the",
    "geography": "The longest river in Africa is the",
}
RECONSTRUCTION_LIMIT = 1e-3
RESCUE_LOGPROB_LIMIT = 1e-4


def plan(pass_number: int | None, limit: int) -> dict:
    if not 1 <= limit <= len(PROMPTS):
        raise ValueError("limit must be 1..8")
    passes = [pass_number] if pass_number is not None else [2, 3]
    calls = {p: reader_call(p) for p in passes}
    config = {
        "schema_version": SCHEMA,
        "backend": "local_transformers_packed_expert_offload",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "transformers_version": TRANSFORMERS_VERSION,
        "transformers_revision": TRANSFORMERS_REVISION,
        "writer_layer_hf_zero_based": WRITER,
        "reader_layer_hf_zero_based": HF_READER_LAYER,
        "reader_layer_goodfire_one_based": HF_READER_LAYER + 1,
        "coordinate_zero_based": TARGET_COORD,
        "previous_loop_block3_bank_input": 1,
        "reader_pass": pass_number,
        "reader_passes": passes,
        "reader_call": calls.get(pass_number),
        "restore_call": calls[pass_number] + 1 if pass_number is not None else None,
        "reader_visits": [{"pass": p, "call": call, "restore_call": call + 1} for p, call in calls.items()],
        "root_prompts": dict(list(PROMPTS.items())[:limit]),
        "arms": ["clean", *ARMS],
        "status": "development_only",
        "primary_readout": "beta3 change at layer-13 Q/K/V/MoE reader",
        "secondary_readout": "final reference-token log probability and top1",
        "scientific_effect_floor": "pending independent sham pilot; 0.10 beta is exploratory",
        "independent_unit": "distinct prompt root, with all arms and passes within-root paired",
        "sealed_readout": False,
        "claims": "development diagnostics only; no mechanism confirmation",
    }
    config["manifest_sha256"] = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return config


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def gpu_state():
    try:
        command = ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,utilization.gpu,driver_version",
                   "--format=csv,noheader,nounits"]
        state = subprocess.run(command, check=True, capture_output=True, text=True, timeout=10)
        return [line.split(", ") for line in state.stdout.splitlines()]
    except (OSError, subprocess.SubprocessError):
        return []


class Resources:
    """Sample process RSS and machine headroom during load and inference."""

    def __init__(self, psutil, torch, device):
        self.process = psutil.Process()
        self.psutil, self.torch, self.device = psutil, torch, device
        self.peak_rss = 0
        self.minimum_available = psutil.virtual_memory().available
        self.stop = Event()
        self.thread = Thread(target=self.sample, daemon=True)

    def sample(self):
        while not self.stop.is_set():
            self.peak_rss = max(self.peak_rss, self.process.memory_info().rss)
            self.minimum_available = min(self.minimum_available, self.psutil.virtual_memory().available)
            self.stop.wait(0.25)

    def snapshot(self):
        cuda = self.device.startswith("cuda")
        return {
            "process_rss_bytes": self.process.memory_info().rss,
            "peak_sampled_process_rss_bytes": self.peak_rss,
            "system_ram_total_bytes": self.psutil.virtual_memory().total,
            "system_ram_available_bytes": self.psutil.virtual_memory().available,
            "minimum_sampled_system_ram_available_bytes": self.minimum_available,
            "process_cuda_peak_allocated_bytes": self.torch.cuda.max_memory_allocated(self.device) if cuda else 0,
            "process_cuda_peak_reserved_bytes": self.torch.cuda.max_memory_reserved(self.device) if cuda else 0,
            "nvidia_smi": gpu_state(),
        }


def score_reader(block2, block3, messages, queries):
    fits = reconstruct_reader(block2, block3, messages)
    errors = [fit.reconstruction_relative_error for fit in fits]
    if max(errors) >= RECONSTRUCTION_LIMIT:
        raise RuntimeError(f"reader reconstruction failed: {errors}")
    norm2, norm3 = rms_normalize(block2), rms_normalize(block3)
    weights = queries / np.sqrt(WIDTH)
    score2, score3 = weights @ norm2, weights @ norm3
    coordinate_weight = weights[:, TARGET_COORD]
    beta = np.asarray([fit.beta3 for fit in fits])
    score_beta = np.exp(-np.logaddexp(0, score2 - score3))
    if np.max(np.abs(score_beta - beta)) >= RECONSTRUCTION_LIMIT:
        raise RuntimeError(f"native query scores disagree with output-side reconstruction: {score_beta - beta}")
    return {
        "beta3_qkvm": beta.tolist(),
        "reconstruction_relative_error": errors,
        "block2_score_qkvm": score2.tolist(),
        "block3_score_qkvm": score3.tolist(),
        "block3_minus_block2_score_qkvm": (score3 - score2).tolist(),
        "coordinate_2027_block3_score_qkvm": (coordinate_weight * norm3[TARGET_COORD]).tolist(),
        "coordinate_2027_score_gap_qkvm": (coordinate_weight * (norm3[TARGET_COORD] - norm2[TARGET_COORD])).tolist(),
    }


def probe(model, tokenizer, prompt, root_id, passes, device, out):
    import torch

    sham = sham_coordinate(root_id)
    trace = ReaderTrace(model.model, passes, sham)
    inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
    inputs = {key: value.to(device).expand(len(trace.lanes), -1).contiguous() for key, value in inputs.items()}
    started = time.perf_counter()
    with torch.inference_mode(), trace:
        logits = model(**inputs, use_cache=False, logits_to_keep=1).logits[:, -1].float()
        logprobs = logits.log_softmax(-1)
        top_ids = logits.topk(2, dim=-1).indices
        top_logprobs = logprobs.gather(1, top_ids)
        reference = int(top_ids[0, 0].item())
        reference_values = as_numpy(logprobs[:, reference])
        top_ids = top_ids.cpu().tolist()
        top_logprobs = as_numpy(top_logprobs)
        full_logit_delta = as_numpy((logits - logits[0]).abs().amax(-1))
    seconds = time.perf_counter() - started

    # Preserve tensors even when post-forward validation fails.
    raw = {"writer_last_token": np.asarray(trace.writers),
           "input_ids": inputs["input_ids"][0].cpu().numpy(),
           "reference_logprobs": reference_values, "top2_ids": np.asarray(top_ids),
           "top2_logprobs": top_logprobs}
    raw.update({f"pass{p}_{key}": value for p, reader in trace.readers.items() for key, value in reader.items()})
    np.savez_compressed(out / f"vectors_{root_id}.npz", **raw)
    trace.verify()

    writer = [vector_metrics(value, value) for value in trace.writers]
    rows = []
    for reader_pass, reader in trace.readers.items():
        clean_bank, clean_message = reader["bank3"][0], reader["message"][0]
        clean = score_reader(reader["bank2"][0], clean_bank, clean_message, reader["query"])
        for arm in ("clean", *ARMS):
            lane = 0 if arm == "clean" else trace.lanes[reader_pass, arm]
            after = reader["bank3"][lane]
            delivered = reader["delivered"][lane]
            if not np.array_equal(reader["restored"][lane], clean_bank):
                raise RuntimeError(f"shared bank restoration failed for {root_id}/{reader_pass}/{arm}")
            if arm != "clean" and not np.allclose(patch_vector(clean_bank, arm, sham), after, rtol=0.02, atol=0.05):
                raise RuntimeError(f"realized patch does not match {root_id}/{reader_pass}/{arm}")
            fits = score_reader(reader["bank2"][lane], after, reader["message"][lane], reader["query"])
            dose = vector_metrics(clean_bank, after)
            if arm == "l2_sham" and not np.isclose(dose["raw_l2_delta"], abs(clean_bank[TARGET_COORD]), rtol=0.01, atol=1e-6):
                raise RuntimeError(f"realized sham dose is not matched for {root_id}/{reader_pass}")
            if arm in ("swap", "norm_preserving_zero") and not np.isclose(
                dose["raw_l2_before"], dose["raw_l2_after"], rtol=0.01, atol=1e-6,
            ):
                raise RuntimeError(f"realized bank norm is not preserved for {root_id}/{reader_pass}/{arm}")
            delta = float(reference_values[lane] - reference_values[0])
            if arm == "reader_rescue":
                zero_lane = trace.lanes[reader_pass, "zero_2027"]
                zero_delta = float(reference_values[zero_lane] - reference_values[0])
                if not np.array_equal(reader["message"][lane], reader["message"][zero_lane]):
                    raise RuntimeError(f"rescue lane differs from ablation before output replacement: {root_id}/{reader_pass}")
                if not np.array_equal(delivered, clean_message) or abs(delta) > RESCUE_LOGPROB_LIMIT or top_ids[lane][0] != reference:
                    raise RuntimeError(f"downstream rescue failed for {root_id}/{reader_pass}: {delta}")
                rescue_status = "restored" if abs(zero_delta) > RESCUE_LOGPROB_LIMIT else "ablation_logprob_null"
            else:
                rescue_status = None
            rows.append({
                "root_id": root_id, "arm": arm, "reader_pass": reader_pass,
                "reader_call": reader_call(reader_pass), "lane": lane, "sham_coordinate": sham,
                "reference_token_id": reference, "clean_top1_token_id": reference,
                "native_top1_token_id": top_ids[lane][0], "native_top1_token": tokenizer.decode(top_ids[lane][0]),
                "top1_flip": top_ids[lane][0] != reference, "top2_ids": top_ids[lane],
                "top2_logprobs": top_logprobs[lane].tolist(),
                "clean_reference_logprob": float(reference_values[0]),
                "patched_reference_logprob": float(reference_values[lane]), "reference_logprob_delta": delta,
                "max_abs_final_logit_delta": float(full_logit_delta[lane]),
                "layer17_raw_coordinate_by_pass": [item["raw_coordinate"] for item in writer],
                "layer17_last_token_metrics_by_pass": writer,
                "clean_beta3_qkvm": clean["beta3_qkvm"], "patched_beta3_qkvm": fits["beta3_qkvm"],
                "beta3_delta_qkvm": (np.array(fits["beta3_qkvm"]) - clean["beta3_qkvm"]).tolist(),
                "reader_scores": fits, "clean_reader_scores": clean,
                "reconstruction_relative_error": fits["reconstruction_relative_error"],
                "delivered_reader_l2_from_clean_qkvm": np.linalg.norm(delivered - clean_message, axis=-1).tolist(),
                "raw_target_after": float(after[TARGET_COORD]), "restored_bank_exact": True,
                "rescue_status": rescue_status, "paired_forward_seconds": seconds,
                **dose, "status": "development_only",
            })
    return rows


def execute(out: Path, config: dict, args) -> None:
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "manifest.json", config)
    (out / "records.jsonl").touch()
    (out / "failures.jsonl").touch()
    stage, resources, current_root = "checkpoint metadata", None, None
    timing = {}
    try:
        started = time.perf_counter()
        identity = verify_metadata(args.snapshot)
        write_json(out / "checkpoint.json", identity)
        stage = "checkpoint weights SHA-256"
        weight_path = args.snapshot / WEIGHTS
        identity["weight_sha256"] = file_sha256(weight_path)
        if identity["weight_sha256"] != FILE_HASHES[WEIGHTS]:
            raise RuntimeError("official checkpoint weight SHA-256 mismatch")
        timing["checkpoint_verification_seconds"] = time.perf_counter() - started
        write_json(out / "checkpoint.json", identity)

        stage = "software and resource preflight"
        import psutil
        import torch
        from safetensors import safe_open
        from transformers import AutoTokenizer

        packages = {name: importlib.metadata.version(name) for name in
                    ("torch", "transformers", "accelerate", "tokenizers", "safetensors", "numpy", "psutil")}
        software = {
            "python": sys.version, "platform": platform.platform(), "packages": packages,
            "pytorch_cuda": torch.version.cuda, "device": args.device, "cpu_threads": args.threads,
            "cuda_budget_gib": args.cuda_budget_gib,
        }
        write_json(out / "software.json", software)
        if packages["transformers"] != TRANSFORMERS_VERSION:
            raise RuntimeError(f"install transformers=={TRANSFORMERS_VERSION} for this hook and converter")
        torch.set_num_threads(args.threads)
        resources = Resources(psutil, torch, args.device)
        resources.thread.start()
        write_json(out / "resources_before.json", resources.snapshot())
        gib = 1024**3
        if psutil.virtual_memory().available < (4 if args.device == "cpu" else 2) * gib:
            raise RuntimeError("insufficient available host RAM for bounded packed checkpoint load")
        if args.device.startswith("cuda"):
            free, total = torch.cuda.mem_get_info(args.device)
            budget = args.cuda_budget_gib * gib
            if free < budget + 2 * gib:
                raise RuntimeError("insufficient CUDA headroom for pilot budget plus 2 GiB reserve")
            torch.cuda.set_per_process_memory_fraction(budget / total, args.device)
            software["cuda_device"] = torch.cuda.get_device_name(args.device)
            write_json(out / "software.json", software)

        stage = "tokenizer verification"
        tokenizer = AutoTokenizer.from_pretrained(args.snapshot, local_files_only=True, trust_remote_code=False)
        if not tokenizer.is_fast or tokenizer.eos_token_id != identity["tokenizer_eos_id"] or len(tokenizer) != identity["unpadded_vocab_size"]:
            raise RuntimeError(f"unexpected tokenizer contract: {type(tokenizer).__name__}, {tokenizer.eos_token_id}, {len(tokenizer)}")
        identity["loaded_tokenizer"] = {"class": type(tokenizer).__name__, "size": len(tokenizer), "fast": tokenizer.is_fast}
        write_json(out / "checkpoint.json", identity)

        stage = "native model load"
        started = time.perf_counter()
        with safe_open(weight_path, framework="pt", device="cpu") as weights:
            model, tensor_count = load_model(args.snapshot, weights, args.device)
            timing["load_seconds"] = time.perf_counter() - started
            identity["streamed_expert_tensor_count"] = tensor_count
            identity["execution_architecture"] = "Inkling2LoopForCausalLM"
            identity["resident_parameter_bytes"] = sum(p.numel() * p.element_size() for p in model.parameters() if not p.is_meta)
            write_json(out / "checkpoint.json", identity)
            write_json(out / "resources_loaded.json", resources.snapshot())
            stage = "native forward and intervention verification"
            with (out / "records.jsonl").open("a", encoding="utf-8") as records:
                for current_root, prompt in config["root_prompts"].items():
                    available = psutil.virtual_memory().available
                    if available < gib:
                        raise RuntimeError("less than 1 GiB available system RAM before next root")
                    rows = probe(model, tokenizer, prompt, current_root, config["reader_passes"], args.device, out)
                    for row in rows:
                        records.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                    records.flush()
                    timing[current_root] = rows[0]["paired_forward_seconds"]
                    print(f"{current_root}: {len(rows)} rows, {timing[current_root]:.2f}s", flush=True)
    except BaseException:
        with (out / "failures.jsonl").open("a", encoding="utf-8") as failures:
            failures.write(json.dumps({"stage": stage, "root_id": current_root, "traceback": traceback.format_exc()}) + "\n")
        raise
    finally:
        if resources is not None:
            resources.stop.set()
            resources.thread.join()
            write_json(out / "resources_after.json", resources.snapshot())
        write_json(out / "timing.json", timing)
        hashes = {p.name: file_sha256(p) for p in out.iterdir() if p.is_file() and p.name != "sha256.json"}
        write_json(out / "sha256.json", hashes)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="run the verified official local checkpoint")
    parser.add_argument("--snapshot", type=Path, help="local snapshot of the pinned checkpoint")
    parser.add_argument("--pass-number", type=int, choices=(2, 3), help="default: measure both reader passes")
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--out", type=Path, default=Path("runs/ink_route_01"))
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    parser.add_argument("--cuda-budget-gib", type=float, default=5)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    config = plan(args.pass_number, args.limit)
    if args.run:
        if args.snapshot is None or args.threads < 1 or args.cuda_budget_gib <= 0:
            parser.error("--run requires --snapshot and positive resource limits")
        execute(args.out, config, args)
    else:
        print(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
