"""Capture native labels and replay prompt-resident InkyNano controllers."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

from gym_manifest import sha
from matrix_cases import cases_and_labels, digest, jsonl, render, score
from stage1_data import read_json, write_json
from stage1_memory import GIB, MemoryGuard, available_memory


def runtime_path(path, distro):
    path = path.resolve()
    if not distro or os.name != "nt":
        return str(path)
    drive = path.drive
    if len(drive) == 2 and drive[1] == ":":
        return f"/mnt/{drive[0].lower()}/" + "/".join(path.parts[1:])
    if drive.casefold() in (f"\\\\wsl.localhost\\{distro}", f"\\\\wsl$\\{distro}"):
        return "/" + "/".join(path.parts[1:])
    raise ValueError(f"cannot translate runtime path {path}")


def decode_native(args, model, text, limit, mode, directory, log):
    prompt, receipt = directory / "prompt.txt", directory / "native.json"
    prompt.write_text(text, encoding="utf8", newline="\n")
    command = ["wsl.exe", "-d", args.wsl, "--exec"] if args.wsl and os.name == "nt" else []
    if args.wsl_run_lock:
        command.extend(("flock", args.wsl_run_lock))
    command.extend(runtime_path(path, args.wsl) for path in (args.native, model, prompt, receipt))
    command.extend((str(limit), str(args.context), str(args.threads), mode,
                    str(args.rss_limit_bytes), "1" if args.parse_special else "0", str(args.batch_tokens)))
    if args.gpu_layers:
        command.extend((str(args.gpu_layers), str(args.vram_limit_bytes)))
    start = perf_counter()
    with log.open("wb") as stderr:
        subprocess.run(command, check=True, stdout=stderr, stderr=stderr, timeout=args.case_timeout)
    result = read_json(receipt)
    result["subprocess_seconds"] = perf_counter() - start
    return result


def validate_plan(plan, case, source_hash, teacher_runtime, bank_digest):
    if (plan["prompt_sha256"] != digest(render(case)) or plan["model_sha256"] != source_hash
            or plan["runtime_revision"] != teacher_runtime
            or plan.get("expert_manifest_sha256") != bank_digest):
        raise ValueError(f"stale plan {case['id']}")


def run(args):
    from matrix_expert_bank import ExpertBank
    from matrix_gguf import PublishedComposition, pack

    guard = MemoryGuard()
    guard.check()
    if args.rss_limit_bytes > available_memory() - GIB:
        raise MemoryError("insufficient host RAM for the configured native process limit plus headroom")
    cases = cases_and_labels(args.cases)
    if args.command == "benchmark":
        cases = [case for case in cases if case["split"] == "dev"]
    if args.limit is not None:
        cases = cases[:args.limit]
    if not cases:
        raise ValueError("empty run selection")
    bank = ExpertBank(args.expert_bundle) if getattr(args, "expert_bundle", None) else None
    composition = PublishedComposition.read(args.published_composition) if args.published_composition else None
    source_hash = bank.model_sha256 if bank else sha(args.model)
    bank_digest = bank.manifest_sha256 if bank else composition.manifest_sha256 if composition else None
    if composition and composition.model_sha256 != source_hash:
        raise ValueError("published composition receipt identifies a different teacher GGUF")
    if args.model_sha256 and args.model_sha256.casefold() != source_hash:
        raise ValueError("source GGUF SHA-256 mismatch")
    args.out.mkdir(parents=True, exist_ok=True)
    log_folder = args.out / "logs"
    log_folder.mkdir(exist_ok=True)
    requested_capacity = getattr(args, "capacity", None)
    if args.command == "benchmark":
        checkpoint_dir = args.controllers or args.plans.parent
        checkpoint_paths = {name: checkpoint_dir / f"{name}.pt" for name in ("ldt", "trm")}
    else:
        checkpoint_paths = {}
    spec = {
        "command": args.command, "case_file_sha256": sha(args.cases), "model_sha256": source_hash,
        "runtime_revision": args.runtime_revision, "runtime_binary_sha256": sha(args.native),
        "expert_manifest_sha256": bank_digest,
        "context": args.context, "threads": args.threads, "batch_tokens": args.batch_tokens, "sampler": "greedy",
        "n_gpu_layers": args.gpu_layers, "max_new_override": args.max_new, "parse_special": args.parse_special,
        "rss_limit_bytes": args.rss_limit_bytes, "capacity": requested_capacity,
        "teacher_in_ram": args.teacher_in_ram,
        "plans_sha256": sha(args.plans) if args.command == "benchmark" else None,
        "controller_checkpoint_sha256": {name: sha(path) for name, path in checkpoint_paths.items()},
    }
    if composition:
        spec["published_composition_sha256"] = sha(args.published_composition)
    if args.gpu_layers:
        spec["vram_limit_bytes"] = args.vram_limit_bytes
    if args.wsl_run_lock:
        spec["wsl_run_lock"] = args.wsl_run_lock
    spec_path = args.out / "spec.json"
    if spec_path.is_file() and read_json(spec_path) != spec:
        raise ValueError("run settings changed; use a new --out directory")
    write_json(spec_path, spec)

    plans_by_case = {}
    teacher_runtime = None
    if args.command == "benchmark":
        sys.path.insert(0, str(args.gguf_python))
        import gguf
        import torch
        from prompt_matrix import PromptController, features, select

        torch.set_num_threads(1)
        controllers = {}
        for architecture, path in checkpoint_paths.items():
            state = torch.load(path, map_location="cpu", weights_only=True)
            if (state["architecture"] != architecture or state["model_sha256"] != source_hash
                    or state.get("expert_manifest_sha256") != bank_digest
                    or teacher_runtime not in (None, state["runtime_revision"])):
                raise ValueError(f"incompatible controller checkpoint {path}")
            teacher_runtime = state["runtime_revision"]
            controllers[architecture] = PromptController.from_checkpoint(state)
        chosen = [plan for plan in read_json(args.plans) if plan["capacity"] == args.capacity
                  and plan["policy"] in ("global", "ldt", "trm")]
        for plan in chosen:
            policies = plans_by_case.setdefault(plan["case_id"], {})
            if plan["policy"] in policies:
                raise ValueError(f"duplicate plan {plan['case_id']} {plan['policy']}")
            policies[plan["policy"]] = plan
        if any(set(plans_by_case.get(case["id"], {})) != {"global", "ldt", "trm"} for case in cases):
            raise ValueError("missing matched global/LDT/TRM plans")

    receipts = args.out / ("routing.jsonl" if args.command == "capture" else "metrics.jsonl")
    seen = set()
    if receipts.is_file():
        for row in jsonl(receipts):
            pair = row["case_id"], row.get("policy")
            if pair in seen:
                raise ValueError(f"duplicate receipt {pair}")
            seen.add(pair)
    n_written = 0
    with receipts.open("a", encoding="utf8", newline="\n") as sink:
        for case in cases:
            prompt = render(case)
            case_plans = plans_by_case.get(case["id"], {})
            variants = [(None, None)] if args.command == "capture" else (
                ([] if bank else [("teacher_subset" if composition else "native", None)]) + list(case_plans.items())
            )
            for policy, plan in variants:
                if (case["id"], policy) in seen:
                    continue
                guard.check()
                tag = f"{digest(case['id'])[:12]}-{policy or 'trace'}"
                with TemporaryDirectory(prefix="case-", dir=args.out) as scratch:
                    directory = Path(scratch)
                    model, layout = args.model, {}
                    selection_seconds = 0.0
                    if plan:
                        validate_plan(plan, case, source_hash, teacher_runtime, bank_digest)
                        prepare = perf_counter()
                        selection = plan["resident_by_layer"]
                        if policy in controllers:
                            with torch.inference_mode():
                                evidence = torch.from_numpy(features(case)).unsqueeze(0)
                                demand = controllers[policy](evidence).softmax(-1)[0].mean(0).numpy()
                            support = bank or composition
                            selection = select(demand, args.capacity, support.available if support else None)
                            if selection != plan["resident_by_layer"]:
                                raise ValueError(f"live and recorded high-scoring sets differ for {case['id']}")
                        selection_seconds = perf_counter() - prepare
                        model = directory / "resident.gguf"
                        max_bytes = args.rss_limit_bytes - (768 << 20)
                        layout = (bank.pack(gguf, model, selection, max_bytes, guard) if bank else
                                  pack(gguf, args.model, model, selection, max_bytes, guard, composition))
                    native_mode = (("trace-ram" if args.teacher_in_ram else "trace") if policy is None
                                   else ("ram" if plan or args.teacher_in_ram else "mmap"))
                    output = decode_native(args, model, prompt, args.max_new or case["max_tokens"],
                                           native_mode, directory, log_folder / f"{tag}.log")
                row = {
                    "case_id": case["id"], "root": case["root"], "task": case["task"],
                    "mode": case["mode"], "prompt_sha256": digest(prompt), "model_sha256": source_hash,
                    "runtime_revision": args.runtime_revision,
                }
                if args.command == "capture":
                    if output["embedding_length"] != 2304:
                        raise ValueError("native teacher capture requires the real Inkling Mini embedding width")
                    if output["decoded_tokens"] < 1:
                        raise ValueError(f"no observed decode routing for {case['id']}")
                    row.update(trace_kind="llama_cpp_inkling2_native", phases=output.pop("phases"),
                               traced_runtime=output)
                    if composition:
                        composition.expand_mass(row["phases"])
                        row["published_composition_sha256"] = spec["published_composition_sha256"]
                else:
                    teacher_capacity = len(composition.available[0]) if composition else 128
                    quality_evaluated = args.max_new is None or args.max_new >= case["max_tokens"]
                    row.update(policy=policy, capacity=args.capacity if plan else teacher_capacity,
                               expert_manifest_sha256=bank_digest,
                               storage_mode=native_mode, quality_evaluated=quality_evaluated,
                               quality=score(case, output["content"]) if quality_evaluated else {},
                               teacher_runtime_revision=teacher_runtime,
                               offline_mass=plan["offline_mass"] if plan else None,
                               selection_seconds=selection_seconds,
                               end_to_end_seconds=selection_seconds + layout.get("pack_seconds", 0.0)
                               + output["subprocess_seconds"], **layout, **output)
                sink.write(json.dumps(row, allow_nan=False, ensure_ascii=False) + "\n")
                sink.flush()
                os.fsync(sink.fileno())
                n_written += 1
                print(f"{case['id']} {policy or 'trace'}", flush=True)
    print(f"committed {n_written} new receipts to {receipts}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("capture", "benchmark"):
        p = sub.add_parser(name)
        p.add_argument("--cases", type=Path, required=True)
        if name == "capture":
            p.add_argument("--model", type=Path, required=True)
        else:
            inputs = p.add_mutually_exclusive_group(required=True)
            inputs.add_argument("--model", type=Path, help="run an unrestricted control from a supplied GGUF")
            inputs.add_argument("--expert-bundle", type=Path, help="assemble only chosen isolated experts")
        p.add_argument("--model-sha256")
        p.add_argument("--published-composition", type=Path, help="receipt for a scoped published-bank teacher GGUF")
        p.add_argument("--teacher-in-ram", action="store_true", help="eagerly load a bounded teacher without mapping a second file view")
        p.add_argument("--native", type=Path, required=True)
        p.add_argument("--runtime-revision", required=True)
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--wsl")
        p.add_argument("--wsl-run-lock", help="Linux lock path shared by local native model processes")
        p.add_argument("--context", type=int, default=2048)
        p.add_argument("--threads", type=int, default=4)
        p.add_argument("--batch-tokens", type=int, default=128)
        p.add_argument("--gpu-layers", type=int, default=0, help="partially offload to CUDA0")
        p.add_argument("--vram-limit-bytes", type=int, default=2 * GIB)
        p.add_argument("--rss-limit-bytes", type=int, default=4 * GIB)
        p.add_argument("--case-timeout", type=int, default=900)
        p.add_argument("--max-new", type=int)
        p.add_argument("--limit", type=int)
        p.add_argument("--parse-special", action="store_true",
                       help="parse native tokenizer markers in already formatted prompts")
        if name == "benchmark":
            p.add_argument("--plans", type=Path, required=True)
            p.add_argument("--controllers", type=Path, help="defaults to the plans directory")
            p.add_argument("--gguf-python", type=Path, required=True)
            p.add_argument("--capacity", type=int, choices=(8, 16, 32), default=8)
    a = parser.parse_args()
    if (a.max_new is not None and a.max_new < 2 or a.limit is not None and a.limit <= 0
            or a.context < 128 or a.threads < 1 or not 1 <= a.batch_tokens <= a.context or a.rss_limit_bytes <= (768 << 20)
            or a.gpu_layers < 0 or a.vram_limit_bytes <= 0
            or a.wsl_run_lock and (not a.wsl or not a.wsl_run_lock.startswith("/"))
            or a.published_composition and not a.model or a.teacher_in_ram and not a.model):
        parser.error("invalid runtime limits")
    run(a)


if __name__ == "__main__":
    main()
