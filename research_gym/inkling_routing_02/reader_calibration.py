"""Development-only native reader calibration from batched clean captures."""

import argparse
from contextlib import ExitStack
from functools import partial
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np

from ..inkling_routing_01.checkpoint import MODEL_ID, MODEL_REVISION, TRANSFORMERS_VERSION, file_sha256, verify_metadata
from ..inkling_routing_01.core import WIDTH
from ..inkling_routing_01.hooks import CORE_END, CORE_START, as_numpy, reader_call
from ..inkling_routing_01.local import RECONSTRUCTION_LIMIT, execute, write_json
from .native import (NATIVE_PACKAGES, PASSES, REPLAYS, CalibrationTrace,
                     development_roots, digest, headroom, score_edit)


def capture(model, inputs):
    """Observe the actual banks and messages without modifying the model call."""
    import torch

    visits, reader_data, writers = 0, {}, []
    reader = model.model.message_input
    target = {reader_call(number): number for number in PASSES}

    def save_input(module, args):
        nonlocal visits
        if visits % (CORE_END - CORE_START) != args[2]:
            raise RuntimeError("unexpected reader visit in clean capture")
        if visits in target:
            bank2, bank3, local_idx = args
            reader_data[target[visits]] = {
                "bank2": bank2[:, -1:].clone(), "bank3": bank3[:, -1:].clone(),
                "local_idx": local_idx,
            }
        visits += 1

    def save_output(module, args, output):
        if visits - 1 in target:
            reader_data[target[visits - 1]]["message"] = output[:, -1:].clone()

    def save_writer(module, args, output):
        writers.append(output[:, -1].clone())

    with ExitStack() as stack:
        stack.callback(reader.register_forward_pre_hook(save_input).remove)
        stack.callback(reader.register_forward_hook(save_output).remove)
        stack.callback(model.model.layers[CORE_END - 1].register_forward_hook(save_writer).remove)
        model(**inputs, use_cache=False, logits_to_keep=1)
    if visits != 2 * (CORE_END - CORE_START) or len(writers) != 3 or set(reader_data) != set(PASSES):
        raise RuntimeError("incomplete native clean capture")
    for number, recorded in reader_data.items():
        if not torch.equal(recorded["bank3"][:, 0], writers[number - 2]):
            raise RuntimeError("clean bank does not match previous writer")
    return reader_data, writers


def replay_reader(reader, trace, pass_number, bank2, bank3, local_idx):
    """Apply the native solo reader to copies of the observed last-token banks."""
    call = reader_call(pass_number)
    trace.visits = call
    trace.previous_banks = (bank2, bank3, bank2.clone(), bank3.clone())
    patched_args = trace.before_reader(reader, (bank2, bank3, local_idx))
    raw = reader(*patched_args)
    trace.after_reader(reader, patched_args, raw)
    traced = trace.readers[pass_number]
    zero = trace.lanes[pass_number, "zero_2027"]
    rescue = trace.lanes[pass_number, "reader_rescue"]
    if (not np.array_equal(traced["message"][zero], traced["message"][rescue])
            or not np.array_equal(traced["delivered"][rescue], traced["message"][0])):
        raise RuntimeError("isolated reader-output rescue failed")
    return traced


def probe(groups, model, tokenizer, _, group_id, passes, device, out):
    import torch

    group = groups[group_id]
    headroom(device)
    inputs = tokenizer([root["prompt"] for root in group], add_special_tokens=False, return_tensors="pt")
    if inputs["input_ids"].tolist() != [root["token_ids"] for root in group]:
        raise RuntimeError("clean capture tokenization drift")
    inputs = {name: value.to(device) for name, value in inputs.items()}
    reader = model.model.message_input
    rows, started = [], time.perf_counter()
    with torch.inference_mode():
        captured, writers = capture(model, inputs)
        prefill_time = time.perf_counter() - started
        for i, root in enumerate(group):
            for pass_number in passes:
                real = captured[pass_number]
                for replay in range(REPLAYS):
                    trace = CalibrationTrace(model.model, pass_number, root)
                    n = len(trace.lanes)
                    bank2 = real["bank2"][i:i + 1].expand(n, -1, -1, -1).contiguous()
                    bank3 = real["bank3"][i:i + 1].expand(n, -1, -1).contiguous()
                    traced = replay_reader(reader, trace, pass_number, bank2, bank3, real["local_idx"])
                    trace.writers = [as_numpy(writer[i]) for writer in writers]
                    error = np.linalg.norm(traced["message"][0] - as_numpy(real["message"][i, 0]))
                    error /= np.linalg.norm(traced["message"][0])
                    if error >= RECONSTRUCTION_LIMIT:
                        raise RuntimeError("isolated reader does not reproduce the full-forward clean message")
                    if replay == 0:
                        np.savez_compressed(out / f"reader_{root['root_id']}_pass{pass_number}.npz", **traced,
                                            writer_last_token=np.asarray(trace.writers),
                                            input_ids=np.asarray(root["token_ids"]))
                    rows.extend(score_replay(root, trace, pass_number, replay, lane, label, error, prefill_time)
                                for (_, label), lane in trace.lanes.items())
    if any(row["validation_issues"] for row in rows):
        with (out / "invalid-rows.jsonl").open("a", encoding="utf-8") as invalid:
            for row in rows:
                invalid.write(json.dumps(row, allow_nan=False) + "\n")
        raise RuntimeError("invalid native reader control dose or edit")
    return rows


def score_replay(root, trace, pass_number, replay, lane, label, error, elapsed):
    result = score_edit(trace, pass_number, lane, label)
    return {
        **result, "root_id": root["root_id"], "reader_pass": pass_number, "replay": replay,
        "label": label, "lane": lane, "clean_capture_relative_error": float(error),
        "paired_forward_seconds": elapsed, "execution": "native_reader_only",
    }


def replay_archived_pass3(args, roots):
    """Run native reader controls on verified earlier pretrained CUDA banks."""
    import importlib.metadata
    import torch
    from transformers.models.inkling2_loop.modeling_inkling2_loop import Inkling2LoopMessageAttentionResidual

    from .receipts import read_json, verified_run

    manifest, _, failures = verified_run(args.archived_pass3)
    if (manifest["schema"] != "ink-route-01.pass3-k.calibration.v1" or failures
            or set(manifest["root_prompts"]) != {root["root_id"] for root in roots}):
        raise ValueError("expected the complete earlier native pass-3 development run")
    if importlib.metadata.version("transformers") != TRANSFORMERS_VERSION:
        raise ValueError("archived reader replay requires pinned Transformers")
    identity = verify_metadata(args.snapshot)
    identity["weight_sha256"] = read_json(args.archived_pass3 / "checkpoint.json")["weight_sha256"]
    identity["weight_identity_source"] = "verified original native prefill receipts"
    torch.set_num_threads(args.threads)
    reader = Inkling2LoopMessageAttentionResidual(SimpleNamespace(
        loop_layer_index_start=CORE_START, loop_layer_index_end=CORE_END,
        hidden_size=WIDTH, _attn_implementation="eager",
    )).eval()
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    write_json(args.out / "checkpoint.json", identity)
    with torch.inference_mode(), (args.out / "records.jsonl").open("w", encoding="utf-8") as records:
        for root in roots:
            headroom("cpu")
            for replay in range(REPLAYS):
                trace = CalibrationTrace(None, 3, root)
                n = len(trace.lanes)
                source = args.archived_pass3 / f"vectors_{root['root_id']}_replay{replay}.npz"
                with np.load(source, allow_pickle=False) as archives:
                    bank2 = torch.tensor(archives["bank2"][0], dtype=torch.float32).expand(n, 1, 1, WIDTH).contiguous()
                    original = archives["original"][0]
                    bank3 = torch.tensor(original, dtype=torch.bfloat16).expand(n, 1, WIDTH).contiguous()
                    if not np.array_equal(as_numpy(bank3[0, 0]), original):
                        raise ValueError("archived block-3 bank is not losslessly BF16")
                    index = reader_call(2)
                    reader.query[index].copy_(torch.tensor(archives["query"], dtype=torch.float32))
                    captured = archives["message"][0]
                    traced = replay_reader(reader, trace, 3, bank2, bank3, index)
                    error = np.linalg.norm(traced["message"][0] - captured) / np.linalg.norm(captured)
                    if error >= RECONSTRUCTION_LIMIT:
                        raise RuntimeError(f"archived clean reader mismatch on {root['root_id']}")
                    for (_, label), lane in trace.lanes.items():
                        row = score_replay(root, trace, 3, replay, lane, label, error, 0)
                        records.write(json.dumps(row, allow_nan=False) + "\n")
    write_json(args.out / "timing.json", {"elapsed_seconds": time.perf_counter() - started})
    config = {
        "schema": "ink-route-02.archived-pass3-native-reader.v1", "reader_passes": [3],
        "development_only": True, "sealed_evaluations_accessed": False,
        "roster_sha256": file_sha256(args.public_roster),
        "source_manifest_sha256": file_sha256(args.archived_pass3 / "sha256.json"),
        "source_sha256": {p.name: file_sha256(p) for p in (Path(__file__), Path(__file__).with_name("native.py"))},
        "transformers": TRANSFORMERS_VERSION, "torch": torch.__version__,
    }
    config["manifest_sha256"] = digest(config)
    write_json(args.out / "manifest.json", config)
    (args.out / "failures.jsonl").touch()
    write_json(args.out / "sha256.json", {p.name: file_sha256(p) for p in args.out.iterdir()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--public-roster", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--archived-pass3", type=Path, help="replay verified earlier pass-3 banks directly on the CPU")
    parser.add_argument("--batch-size", type=int, default=15)
    parser.add_argument("--count", type=int, default=60)
    parser.add_argument("--device", choices=("cuda:0",), default="cuda:0")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--cuda-budget-gib", type=float, default=5)
    args = parser.parse_args()
    roster = development_roots(args.public_roster)
    if not 1 <= args.batch_size <= 20 or not 1 <= args.count <= len(roster):
        parser.error("expected up to 20 roots per batch from the public roster")
    if args.threads < 1 or not 0 < args.cuda_budget_gib <= 5:
        parser.error("invalid resource budget")
    if args.archived_pass3 is not None:
        replay_archived_pass3(args, roster)
        return
    groups = {"smoke": roster[:min(2, args.count)]}
    groups.update({f"batch-{start:02d}": roster[start:min(start + args.batch_size, args.count)]
                   for start in range(2, args.count, args.batch_size)})
    config = {
        "schema": "ink-route-02.native-reader-development.v1",
        "development_only": True, "sealed_evaluations_accessed": False,
        "root_prompts": {name: "public roster batch" for name in groups},
        "groups": {name: [root["root_id"] for root in group] for name, group in groups.items()},
        "roster_sha256": file_sha256(args.public_roster), "reader_passes": PASSES, "replays": REPLAYS,
        "model_id": MODEL_ID, "checkpoint_revision": MODEL_REVISION,
        "source_sha256": {path.name: file_sha256(path) for path in (
            Path(__file__), Path(__file__).with_name("native.py"),
            *(Path(__file__).parent.parent / "inkling_routing_01" / name
              for name in ("checkpoint.py", "core.py", "hooks.py", "local.py")))},
    }
    config["manifest_sha256"] = digest(config)

    def preflight(software):
        if software["packages"] != NATIVE_PACKAGES or software["pytorch_cuda"] != "12.8":
            raise RuntimeError("native CUDA calibration requires the pinned package set and CUDA 12.8")
        headroom(args.device, args.cuda_budget_gib)

    execute(args.out, config, args, probe_root=partial(probe, groups), check_preflight=preflight)


if __name__ == "__main__":
    main()
