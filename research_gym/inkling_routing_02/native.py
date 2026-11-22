"""Calibrate coordinate-2027 controls on the pinned pretrained checkpoint."""

import argparse
from functools import partial
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np

from ..inkling_routing_01.checkpoint import MODEL_ID, MODEL_REVISION, file_sha256
from ..inkling_routing_01.core import HF_READER_LAYER, TARGET_COORD, WIDTH, vector_metrics
from ..inkling_routing_01.hooks import CORE_END, CORE_START, ReaderTrace, apply_native_patch, as_numpy, reader_call
from ..inkling_routing_01.local import PROMPTS, execute, gpu_state, score_reader

REFERENCES, PLACEBOS = 16, 4
PASSES = (2, 3)
REPLAYS = 2
DOSE_TOLERANCE = 0.01
NULL_LOGP = 1e-4
NATIVE_PACKAGES = {
    "accelerate": "1.16.0", "numpy": "2.5.5", "psutil": "7.2.2", "safetensors": "0.8.0",
    "tokenizers": "0.23.3", "torch": "2.11.0+cu128", "transformers": "5.24.0",
}
CONTROLS = ("zero_2027", "norm_preserving_zero", "reader_rescue", "random_vector", "norm_preserving_vector")
SMOKE = {
    "smoke-garden": "The greenhouse roof collected rainwater for seedlings.",
    "smoke-music": "Each string vibrated softly under the bow.",
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def seeded(*parts):
    return int(digest(parts)[:16], 16)


def development_roots(path):
    if path is None:
        coordinates = [i for i in range(WIDTH) if i != TARGET_COORD]
        return [
            {"root_id": key, "prompt": prompt,
             "reference_coordinates": draw[:REFERENCES], "placebo_coordinates": draw[REFERENCES:]}
            for key, prompt in SMOKE.items()
            for draw in [random.Random(seeded(key)).sample(coordinates, REFERENCES + PLACEBOS)]
        ]
    roster = json.loads(path.read_text(encoding="utf-8"))
    hash_record = roster.pop("roster_sha256")
    if (digest(roster) != hash_record or roster["schema"] != "ink-route-01.pass3-k.roots.v1"
            or roster["development_only"] is not True or roster["checkpoint_revision"] != MODEL_REVISION):
        raise ValueError("expected the frozen, public development roster")
    roots = roster["roots"]
    for key in ("root_id", "domain", "row_id", "text_sha256", "prompt"):
        if len({root[key] for root in roots}) != len(roots):
            raise ValueError(f"development roots share {key}")
    for root in roots:
        reference, placebo = root["reference_coordinates"], root["placebo_coordinates"]
        draw = reference + placebo
        if (root["prompt"] in (*PROMPTS.values(), *SMOKE.values()) or len(reference) != REFERENCES
                or len(placebo) != PLACEBOS or len(set(draw)) != len(draw)
                or any(type(i) is not int or not 0 <= i < WIDTH or i == TARGET_COORD for i in draw)):
            raise ValueError(f"invalid development root {root['root_id']}")
    return roots


def edit_vector(original, seed, preserve_norm):
    """Return an equal-chord random edit with coordinate 2027 fixed."""
    x = original.astype(np.float64, copy=True)
    target = x[TARGET_COORD]
    x[TARGET_COORD] = 0
    direction = np.random.default_rng(seed).standard_normal(WIDTH)
    direction[TARGET_COORD] = 0
    dose = abs(target)
    if preserve_norm:
        norm = np.linalg.norm(x)
        if norm == 0 or dose > 2 * norm:
            raise ValueError("equal-dose norm-preserving rotation is infeasible")
        direction -= x * (np.dot(x, direction) / norm**2)
        radial = dose**2 / (2 * norm)
        tangent = np.sqrt(max(0, dose**2 - radial**2))
        x = x * (1 - radial / norm) + direction * (tangent / np.linalg.norm(direction))
    else:
        x += direction * (dose / np.linalg.norm(direction))
    x[TARGET_COORD] = target
    return x


class CalibrationTrace(ReaderTrace):
    """Use the original native hooks with one independent intervention per lane."""

    def __init__(self, decoder, pass_number, root):
        super().__init__(decoder, (), None)
        self.calls = {reader_call(pass_number): pass_number}
        self.labels = dict.fromkeys(CONTROLS)
        for group in ("reference", "placebo"):
            self.labels.update({f"{group}_{i:02d}": c for i, c in enumerate(root[f"{group}_coordinates"])})
        self.lanes.update({(pass_number, label): i for i, label in enumerate(self.labels, 1)})
        self.seed = seeded("INK-ROUTE-02", root["root_id"])
        self.previous_banks = None
        self.restored_calls = []

    def before_reader(self, module, args):
        import torch

        bank2, bank3, index = args
        call = self.visits
        sweep_size = CORE_END - CORE_START
        target_call = next(iter(self.calls))
        if index != call % sweep_size:
            raise RuntimeError(f"unexpected reader order at visit {call}")
        self.visits += 1
        if index == 0:
            self.previous_banks = (bank2, bank3, bank2.clone(), bank3.clone())
        native2, native3, expected2, expected3 = self.previous_banks
        if (bank2 is not native2 or bank3 is not native3
                or not torch.equal(bank2, expected2) or not torch.equal(bank3, expected3)):
            raise RuntimeError(f"original banks did not reach reader {call}")
        if call not in self.calls:
            if call > target_call and call // sweep_size == target_call // sweep_size:
                self.restored_calls.append(call)
            return None

        pass_number = self.calls[call]
        if (bank3.ndim != 3 or bank3.shape[0] != len(self.lanes) or bank3.shape[-1] != WIDTH
                or bank2.shape != (*bank3.shape[:2], 1, WIDTH)):
            raise RuntimeError("unexpected native bank layout")
        patched = bank3.clone()
        original = as_numpy(bank3[0, -1])
        for label, coordinate in self.labels.items():
            lane = self.lanes[pass_number, label]
            if label in ("random_vector", "norm_preserving_vector"):
                value = edit_vector(original, self.seed, label == "norm_preserving_vector")
                patched[lane, -1] = torch.as_tensor(value, device=patched.device, dtype=patched.dtype)
            else:
                arm = label if coordinate is None else "l2_sham"
                apply_native_patch(patched[lane:lane + 1], arm, coordinate)
        if not torch.equal(patched[0], bank3[0]) or not torch.equal(patched[:, :-1], bank3[:, :-1]):
            raise RuntimeError("edit escaped its assigned final-token lanes")
        self.readers[pass_number] = {
            "bank2": as_numpy(bank2[:, -1, 0]), "original": as_numpy(bank3[:, -1]),
            "bank3": as_numpy(patched[:, -1]), "query": as_numpy(module.query[index]),
        }
        return bank2, patched, index

    def verify(self):
        import torch

        super().verify()
        call = next(iter(self.calls))
        stop = (1 + call // (CORE_END - CORE_START)) * (CORE_END - CORE_START)
        if self.restored_calls != list(range(call + 1, stop)):
            raise RuntimeError("missing downstream restoration check")
        bank2, bank3, expected2, expected3 = self.previous_banks
        if not torch.equal(bank2, expected2) or not torch.equal(bank3, expected3):
            raise RuntimeError("native banks changed after the final read")


def headroom(device, budget=None):
    import psutil

    if psutil.virtual_memory().available < (10 if budget is not None else 2) * 1024**3:
        raise RuntimeError("insufficient host RAM for a bounded native calibration")
    if device.startswith("cuda"):
        gpu = next((row for row in gpu_state() if row[0] == "0"), None)
        min_mib = (budget * 1024 if budget is not None else 0) + 2048
        if gpu is None or int(gpu[2]) - int(gpu[3]) < min_mib:
            raise RuntimeError(f"insufficient NVIDIA headroom ({min_mib:g} MiB required)")


def score_lane(root_id, pass_number, replay, trace, lane, label, logits, elapsed):
    reader = trace.readers[pass_number]
    original, after = reader["original"][lane], reader["bank3"][lane]
    delta = after - original
    dose = float(np.linalg.norm(delta))
    target = original[TARGET_COORD]
    coordinate = trace.labels.get(label)
    fit = score_reader(reader["bank2"][lane], after, reader["message"][lane], reader["query"])
    issues = []
    dose_error = None
    if coordinate is not None or label in ("random_vector", "norm_preserving_vector"):
        dose_error = abs(dose - abs(target)) / abs(target) if target else None
        if dose_error is None or dose_error > DOSE_TOLERANCE or delta[TARGET_COORD] != 0:
            issues.append("dose_or_target")
    if coordinate is not None and (np.count_nonzero(delta) != 1 or delta[coordinate] * target >= 0):
        issues.append("coordinate_or_direction")
    if label in ("zero_2027", "reader_rescue") and (after[TARGET_COORD] != 0 or dose != abs(target)):
        issues.append("zero_ablation")
    norm_error = abs(np.linalg.norm(after) / np.linalg.norm(original) - 1)
    if label.startswith("norm_preserving_") and norm_error > DOSE_TOLERANCE:
        issues.append("norm_preservation")
    tokens, logp, max_logit = logits
    logp_delta = float(logp[lane] - logp[0])
    if label == "reader_rescue":
        zero = trace.lanes[pass_number, "zero_2027"]
        if (not np.array_equal(reader["message"][lane], reader["message"][zero])
                or not np.array_equal(reader["delivered"][lane], reader["message"][0])
                or abs(logp_delta) > NULL_LOGP or tokens[lane] != tokens[0]):
            issues.append("rescue")
    return {
        "root_id": root_id, "reader_pass": pass_number, "replay": replay, "lane": lane, "label": label,
        "sham_coordinate": coordinate, "vector_seed": trace.seed, "raw_l2_dose": dose,
        "relative_sham_dose_error": dose_error, "relative_norm_error": float(norm_error),
        "bank_target_before": float(target), "bank_target_after": float(after[TARGET_COORD]),
        "changed_coordinates": int(np.count_nonzero(delta)), "restored_calls": trace.restored_calls,
        "writer": [vector_metrics(x, x) for x in trace.writers],
        "reader_scores": fit, "reference_token_id": tokens[0], "native_top1_token_id": tokens[lane],
        "reference_logprob": float(logp[lane]), "reference_logprob_delta": logp_delta,
        "logprob_null": abs(logp_delta) <= NULL_LOGP, "top_token_flip": tokens[lane] != tokens[0],
        "max_abs_final_logit_delta": float(max_logit[lane]), "validation_issues": issues,
        "paired_forward_seconds": elapsed,
    }


def probe(roots, model, tokenizer, prompt, root_id, passes, device, out):
    import torch

    root = roots[root_id]
    inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
    ids = inputs["input_ids"][0].tolist()
    if ids != root.get("token_ids", ids):
        raise RuntimeError(f"tokenization drift on {root_id}")
    rows = []
    for pass_number in passes:
        for replay in range(REPLAYS):
            headroom(device)
            trace = CalibrationTrace(model.model, pass_number, root)
            batch = {name: value.to(device).expand(len(trace.lanes), -1).contiguous() for name, value in inputs.items()}
            started = time.perf_counter()
            with torch.inference_mode(), trace:
                logits = model(**batch, use_cache=False, logits_to_keep=1).logits[:, -1].float()
                tokens = logits.argmax(-1).cpu().tolist()
                logp = as_numpy(logits.log_softmax(-1)[:, tokens[0]])
                max_logit = as_numpy((logits - logits[0]).abs().amax(-1))
            elapsed = time.perf_counter() - started
            np.savez_compressed(out / f"vectors_{root_id}_pass{pass_number}_replay{replay}.npz",
                                **trace.readers[pass_number], writer_last_token=np.asarray(trace.writers),
                                input_ids=np.asarray(ids))
            trace.verify()
            rows.extend(score_lane(root_id, pass_number, replay, trace, lane, label,
                                   (tokens, logp, max_logit), elapsed)
                        for (_, label), lane in trace.lanes.items())
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--public-roster", type=Path, help="the existing FineWeb development roster; omit for two fresh smoke roots")
    parser.add_argument("--start", type=int, default=0, help="zero-based development roster offset")
    parser.add_argument("--count", type=int)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--cuda-budget-gib", type=float, default=5)
    args = parser.parse_args()
    roster = development_roots(args.public_roster)
    count = len(roster) - args.start if args.count is None else args.count
    if args.start < 0 or count < 1 or args.start + count > len(roster) or args.threads < 1 or not 0 < args.cuda_budget_gib <= 5:
        parser.error("invalid roster slice or resource budget")
    roots = {root["root_id"]: root for root in roster[args.start:args.start + count]}
    config = {
        "schema": "ink-route-02.native-development.v1", "development_only": True,
        "sealed_evaluations_accessed": False, "root_prompts": {k: v["prompt"] for k, v in roots.items()},
        "roster_sha256": file_sha256(args.public_roster) if args.public_roster else None,
        "reader_passes": PASSES, "reader_layer": HF_READER_LAYER, "target_coordinate": TARGET_COORD,
        "model_id": MODEL_ID, "checkpoint_revision": MODEL_REVISION, "replays": REPLAYS,
        "source_sha256": {p.name: file_sha256(p) for p in (
            Path(__file__), *(Path(__file__).parent.parent / "inkling_routing_01" / name
                              for name in ("core.py", "hooks.py", "local.py", "checkpoint.py")))},
        "launch": {"device": args.device, "cuda_budget_gib": args.cuda_budget_gib, "threads": args.threads},
    }
    config["manifest_sha256"] = digest(config)
    def preflight(software):
        if args.device.startswith("cuda") and (
            software["packages"] != NATIVE_PACKAGES or software["pytorch_cuda"] != "12.8"
        ):
            raise RuntimeError("native CUDA calibration requires the pinned package set and CUDA 12.8")
        headroom(args.device, args.cuda_budget_gib if args.device.startswith("cuda") else None)

    execute(args.out, config, args, probe_root=partial(probe, roots), check_preflight=preflight)


if __name__ == "__main__":
    main()
