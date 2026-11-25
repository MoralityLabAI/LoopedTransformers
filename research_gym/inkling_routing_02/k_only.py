"""Native end-to-end development test of pass-3 K messages at matched beta."""

import argparse
from functools import partial
from itertools import product
from pathlib import Path
import time

import numpy as np

from ..inkling_routing_01.checkpoint import MODEL_ID, MODEL_REVISION, file_sha256
from ..inkling_routing_01.core import HF_READER_LAYER, TARGET_COORD, reconstruct_reader
from ..inkling_routing_01.hooks import as_numpy
from ..inkling_routing_01.local import RECONSTRUCTION_LIMIT, execute, score_reader, write_json
from .matched_controls import K, matched_vector
from .native import DOSE_TOLERANCE, NATIVE_PACKAGES, REPLAYS, CalibrationTrace, digest, headroom
from .receipts import RESOLUTION, verified_run

PASS = 3
ARMS = ("clean_replay", "zero_k", "vector_k", "zero_rescue", "vector_rescue")
TREATMENTS = ("zero_k", "vector_k")
UNCLAMPED = ("zero_k_unclamped", "vector_k_unclamped")
RESCUES = ("zero_rescue", "vector_rescue")
ALTERNATE_CHANNELS = (0, 2, 3)
SCIENTIFIC_MINIMUM_TV = 0.01
RESCUE_TOLERANCE_TV = 1e-4
REPLAY_SAFETY_FACTOR = 10
CPU_MIN_AVAILABLE_GIB = 12
CLAMP_RECOVERY_TOLERANCE = 2e-7
SMOKE = {
    "k-only-balloon": "On the ridge, the weather balloon burst above",
    "k-only-bakery": "The bakery labels each sourdough starter with a",
}
EXPANSION = {
    "k-only-tile": "The apprentice replaced the cracked ceramic tile beside",
    "k-only-eclipse": "During the eclipse, the school telescope pointed toward",
    "k-only-estuary": "Low tide exposed small burrows along the estuary",
    "k-only-press": "The printing press stopped when the paper roll",
    "k-only-museum": "At the museum, the curator dimmed the lights to",
    "k-only-reservoir": "The engineers measured the reservoir level every morning before",
    "k-only-pottery": "Before firing the porcelain bowl, the potter brushed",
    "k-only-archive": "The town archive stored the original ferry timetable in",
    "k-only-beehive": "A beekeeper placed a shallow dish of water near",
    "k-only-mosaic": "Under the mosaic, archaeologists discovered a narrow passage leading",
}


def clamp_k_messages(module, args, lanes):
    """Use one FP32 coefficient for both K mixtures of native normalized banks."""
    import torch

    bank2, bank3, index = args
    banks = module.norm(torch.cat((bank2[lanes, -1], bank3[lanes, -1, None]), dim=-2).float())
    query = module.query[index, K].float()
    scores = (banks @ query) * (query.numel() ** -0.5)
    beta = scores.softmax(dim=-1)[:, 1].mean()
    weights = torch.stack((1 - beta, beta))
    return weights @ banks, beta


class KOnlyTrace(CalibrationTrace):
    def __init__(self, decoder, root_id, *, clamp_beta=False):
        root = {"root_id": root_id, "reference_coordinates": [], "placebo_coordinates": []}
        super().__init__(decoder, PASS, root, controls=(*ARMS, *UNCLAMPED) if clamp_beta else ARMS)
        self.clamp_beta = clamp_beta
        self.minimum_dose_ratio = None

    def patch_bank(self, module, bank3, patched, index, pass_number):
        import torch

        if not torch.equal(bank3, bank3[:1].expand_as(bank3)):
            raise RuntimeError("full block-3 bank diverged before the K-only edit")
        x = as_numpy(bank3[0, -1])
        query = as_numpy(module.query[index, K])
        delta, self.minimum_dose_ratio = matched_vector(x, query, self.seed)
        if delta is None:
            raise RuntimeError(f"matched vector infeasible (minimum dose ratio {self.minimum_dose_ratio})")
        edited = torch.as_tensor(x + delta, device=bank3.device, dtype=bank3.dtype)
        for (_, arm), lane in self.lanes.items():
            if arm.startswith("zero_"):
                patched[lane, -1, TARGET_COORD] = 0
            elif arm.startswith("vector_"):
                patched[lane, -1] = edited

    def after_reader(self, module, args, output):
        import torch

        if self.visits - 1 not in self.calls:
            return None
        if output.shape != (*args[1].shape[:2], 4, args[1].shape[-1]):
            raise RuntimeError("unexpected native reader output shape")
        if not torch.equal(output[:, :-1], output[:1, :-1].expand_as(output[:, :-1])):
            raise RuntimeError("intervention changed an earlier token's reader output")
        clean = output[0, -1]
        delivered = output.clone()
        delivered[:, -1] = clean
        for treatment, rescue in zip(TREATMENTS, RESCUES):
            treated = self.lanes[PASS, treatment]
            recovered = self.lanes[PASS, rescue]
            if not torch.equal(output[treated, -1], output[recovered, -1]):
                raise RuntimeError(f"rescue pre-treatment drift for {treatment}")
            delivered[treated, -1, K] = output[treated, -1, K]
        result = self.readers[PASS]
        if self.clamp_beta:
            lanes = [self.lanes[PASS, arm] for arm in TREATMENTS]
            clamped, beta = clamp_k_messages(module, args, lanes)
            for lane, control, message in zip(lanes, UNCLAMPED, clamped):
                reference = self.lanes[PASS, control]
                if not torch.equal(output[lane, -1], output[reference, -1]):
                    raise RuntimeError("unclamped comparison lane differs before delivery")
                delivered[reference, -1, K] = output[reference, -1, K]
                delivered[lane, -1, K] = message
            result["clamped_k"] = as_numpy(clamped)
            result["shared_beta3"] = as_numpy(beta)
        result["message"] = as_numpy(output[:, -1])
        result["delivered"] = as_numpy(delivered[:, -1])
        return delivered


def total_variation(logp1, logp2):
    return float(0.5 * np.sum(abs(np.exp(logp1.astype(np.float64)) - np.exp(logp2.astype(np.float64)))))


def score_forward(trace, logp, root_id, replay, elapsed):
    trace.verify()
    reader = trace.readers[PASS]
    original = reader["original"][0]
    target = original[TARGET_COORD]
    query = reader["query"][K]
    zero, vector = (trace.lanes[PASS, label] for label in TREATMENTS)
    fits = [score_reader(b2, b3, msg, reader["query"])
            for b2, b3, msg in zip(reader["bank2"], reader["bank3"], reader["message"])]
    beta_gap = abs(fits[zero]["beta3_qkvm"][K] - fits[vector]["beta3_qkvm"][K])
    if beta_gap > RESOLUTION:
        raise RuntimeError(f"matched K beta gap {beta_gap} exceeds resolution")
    recovered_betas = {}
    extras = {}
    if trace.clamp_beta:
        shared_beta = float(reader["shared_beta3"])
        for arm, lane in zip(TREATMENTS, (zero, vector)):
            recovered = reconstruct_reader(reader["bank2"][lane], reader["bank3"][lane], reader["delivered"][lane])[K]
            if (abs(recovered.beta3 - shared_beta) > CLAMP_RECOVERY_TOLERANCE
                    or recovered.reconstruction_relative_error > CLAMP_RECOVERY_TOLERANCE):
                raise RuntimeError(f"delivered K message failed exact-clamp reconstruction for {arm}")
            recovered_betas[arm] = recovered.beta3
        unclamped_lanes = [trace.lanes[PASS, arm] for arm in UNCLAMPED]
        extras = {
            "shared_beta3": shared_beta, "delivered_beta3_gap": abs(recovered_betas[TREATMENTS[0]] - recovered_betas[TREATMENTS[1]]),
            "tv_unclamped_zero_vs_vector": total_variation(logp[unclamped_lanes[0]], logp[unclamped_lanes[1]]),
        }
    top_ids = logp.argmax(axis=-1)
    rows = []
    for (_, arm), lane in trace.lanes.items():
        delta = reader["bank3"][lane] - original
        dose = np.linalg.norm(delta)
        dose_error = abs(dose / abs(target) - 1)
        tv_from_clean = total_variation(logp[lane], logp[0])
        if arm in ("clean", "clean_replay"):
            if np.any(delta):
                raise RuntimeError("clean lane received an edit")
        elif dose_error > DOSE_TOLERANCE:
            raise RuntimeError(f"{arm} failed realized dose matching")
        if arm.startswith("zero_") and (delta[TARGET_COORD] != -target or np.count_nonzero(delta) != 1):
            raise RuntimeError("coordinate ablation escaped its target")
        if arm.startswith("vector_"):
            x_err = abs(np.dot(original, delta) / (-target**2) - 1)
            q_err = abs(np.dot(query, delta) / (-target * query[TARGET_COORD]) - 1)
            if delta[TARGET_COORD] or max(x_err, q_err) > DOSE_TOLERANCE:
                raise RuntimeError(f"{arm} failed realized projection matching")
        else:
            x_err = q_err = None
        delivered = reader["delivered"][lane]
        clean = reader["message"][0]
        channels = list(ALTERNATE_CHANNELS)
        if not np.array_equal(delivered[channels], clean[channels]):
            raise RuntimeError("a non-K channel received a treated message")
        if trace.clamp_beta and arm in TREATMENTS:
            expected_k = reader["clamped_k"][TREATMENTS.index(arm)]
        else:
            expected_k = reader["message"][lane, K] if arm in (*TREATMENTS, *UNCLAMPED) else clean[K]
        if not np.array_equal(delivered[K], expected_k):
            raise RuntimeError("K message delivery mismatch")
        if arm in RESCUES and tv_from_clean > RESCUE_TOLERANCE_TV:
            raise RuntimeError(f"{arm} failed downstream rescue: TV {tv_from_clean}")
        rows.append({
            **extras, **({"delivered_beta3_k": recovered_betas.get(arm)} if trace.clamp_beta else {}),
            "root_id": root_id, "replay": replay, "arm": arm, "lane": lane, "reader_pass": PASS,
            "reader_scores": fits[lane], "matched_beta_gap": beta_gap, "restored_calls": trace.restored_calls,
            "raw_l2_dose": float(dose), "relative_dose_error": float(dose_error) if arm not in ("clean", "clean_replay") else None,
            "relative_bank_projection_error": float(x_err) if x_err is not None else None,
            "relative_query_projection_error": float(q_err) if q_err is not None else None,
            "vector_minimum_dose_ratio": trace.minimum_dose_ratio,
            "k_message_l2_from_clean": float(np.linalg.norm(delivered[K] - clean[K])),
            "k_message_l2_zero_vs_vector": float(np.linalg.norm(reader["delivered"][zero, K] - reader["delivered"][vector, K])),
            "tv_from_clean": tv_from_clean, "tv_zero_vs_vector": total_variation(logp[zero], logp[vector]),
            "reference_token_id": int(top_ids[0]), "top_token_id": int(top_ids[lane]),
            "top_token_flip": bool(top_ids[lane] != top_ids[0]),
            "reference_logprob_delta": float(logp[lane, top_ids[0]] - logp[0, top_ids[0]]),
            "paired_forward_seconds": elapsed,
        })
    return rows


def probe(model, tokenizer, prompt, root_id, passes, device, out, *, reserve_gib, clamp_beta=False):
    import torch

    inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
    ids = inputs["input_ids"][0].tolist()
    rows, outcomes = [], []
    for replay in range(REPLAYS):
        headroom(device, reserve_gib=reserve_gib)
        trace = KOnlyTrace(model.model, root_id, clamp_beta=clamp_beta)
        batch = {name: value.to(device).expand(len(trace.lanes), -1).contiguous() for name, value in inputs.items()}
        started = time.perf_counter()
        with torch.inference_mode(), trace:
            logp = model(**batch, use_cache=False, logits_to_keep=1).logits[:, -1].float().log_softmax(-1).cpu().numpy()
        elapsed = time.perf_counter() - started
        np.savez_compressed(out / f"k_only_{root_id}_replay{replay}.npz", **trace.readers[PASS],
                            writer_last_token=np.asarray(trace.writers), input_ids=np.asarray(ids), logp=logp)
        rows.extend(score_forward(trace, logp, root_id, replay, elapsed))
        outcomes.append(logp)
    for row in rows:
        lane = row["lane"]
        row["tv_same_arm_cross_replay"] = total_variation(outcomes[0][lane], outcomes[1][lane])
    return rows


def summarize(rows, *, frozen_floor=None, clamp_beta=False):
    keys = {(r["root_id"], r["replay"], r["arm"]): r for r in rows}
    root_ids = sorted({r["root_id"] for r in rows})
    arms = ("clean", *ARMS, *UNCLAMPED) if clamp_beta else ("clean", *ARMS)
    expected = set(product(root_ids, range(REPLAYS), arms))
    if not root_ids or len(rows) != len(expected) or set(keys) != expected:
        raise ValueError("incomplete K-only paired development grid")
    noise = max(max(r["tv_same_arm_cross_replay"], r["tv_from_clean"] if r["arm"] in ("clean_replay", *RESCUES) else 0)
                for r in rows)
    observed_floor = max(SCIENTIFIC_MINIMUM_TV, REPLAY_SAFETY_FACTOR * noise)
    floor = observed_floor if frozen_floor is None else frozen_floor
    noise_resolves_floor = observed_floor <= floor
    roots = {}
    for root_id in root_ids:
        forwards = [keys[root_id, replay, "zero_k"] for replay in range(REPLAYS)]
        between = [r["tv_zero_vs_vector"] for r in forwards]
        roots[root_id] = {
            "tv_zero_vs_vector_by_replay": between, "clears_floor_both_replays": min(between) >= floor,
            "max_beta_gap": max(r["matched_beta_gap"] for r in forwards),
            "k_message_l2_zero_vs_vector": forwards[0]["k_message_l2_zero_vs_vector"],
            "first_replay_treatment_outcomes": {
                arm: {name: keys[root_id, 0, arm][name] for name in
                      ("tv_from_clean", "top_token_id", "top_token_flip", "reference_logprob_delta")}
                for arm in TREATMENTS
            },
        }
        if clamp_beta:
            roots[root_id].update({
                "shared_beta3_by_replay": [r["shared_beta3"] for r in forwards],
                "max_delivered_beta_gap": max(r["delivered_beta3_gap"] for r in forwards),
                "tv_unclamped_zero_vs_vector_by_replay": [r["tv_unclamped_zero_vs_vector"] for r in forwards],
            })
    root_means = [np.mean(root["tv_zero_vs_vector_by_replay"]) for root in roots.values()]
    return {
        "development_only": True, "sealed_evaluations_accessed": False, "rows": len(rows),
        "roots": roots, "max_observed_replay_or_rescue_tv": noise, "minimum_meaningful_tv": floor,
        "noise_resolves_floor": noise_resolves_floor,
        "mean_paired_root_tv": float(np.mean(root_means)),
        "max_rescue_tv": max(r["tv_from_clean"] for r in rows if r["arm"] in RESCUES),
        "max_reconstruction_error": max(max(r["reader_scores"]["reconstruction_relative_error"]) for r in rows),
        "expand": noise_resolves_floor and all(root["clears_floor_both_replays"] for root in roots.values()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--smoke-receipts", type=Path, help="verified two-root gate required for expansion")
    mode.add_argument("--clamp-beta", action="store_true", help="replay both development roots at shared K beta")
    parser.add_argument("--device", choices=("cuda:0", "cpu"), default="cuda:0")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--cuda-budget-gib", type=float)
    parser.add_argument("--offload-vocabulary", action="store_true")
    args = parser.parse_args()
    if args.cuda_budget_gib is None:
        args.cuda_budget_gib = 1.85 if args.offload_vocabulary else 4.2
    if args.threads < 1 or not 0 < args.cuda_budget_gib <= 5:
        parser.error("invalid resource budget")
    if args.offload_vocabulary and args.device != "cuda:0":
        parser.error("vocabulary offload requires a CUDA decoder")
    reserve_gib = 1 if args.offload_vocabulary else 2
    prompts, gate_sha, gate_floor = SMOKE, None, SCIENTIFIC_MINIMUM_TV if args.clamp_beta else None
    if args.smoke_receipts:
        manifest, rows, _ = verified_run(args.smoke_receipts)
        gate = summarize(rows)
        if (manifest.get("schema") != "ink-route-02.k-only.v1" or manifest.get("root_prompts") != SMOKE
                or set(gate["roots"]) != set(SMOKE) or manifest.get("sealed_evaluations_accessed") is not False
                or not gate["expand"]):
            parser.error("the development smoke evidence does not support expansion")
        prompts, gate_sha = EXPANSION, file_sha256(args.smoke_receipts / "sha256.json")
        gate_floor = gate["minimum_meaningful_tv"]
    config = {
        "schema": "ink-route-02.k-only.v1", "development_only": True, "sealed_evaluations_accessed": False,
        "root_prompts": prompts, "reader_passes": [PASS], "reader_layer": HF_READER_LAYER,
        "arms": ("clean", *ARMS, *UNCLAMPED) if args.clamp_beta else ("clean", *ARMS),
        "replays": REPLAYS, "model_id": MODEL_ID, "checkpoint_revision": MODEL_REVISION,
        "beta_tolerance": RESOLUTION, "dose_tolerance": DOSE_TOLERANCE,
        "launch": {"device": args.device, "offload_vocabulary": args.offload_vocabulary,
                   "budget_gib": args.cuda_budget_gib, "reserve_gib": reserve_gib},
        "rescue_tv_tolerance": RESCUE_TOLERANCE_TV, "scientific_minimum_tv": SCIENTIFIC_MINIMUM_TV,
        "replay_safety_factor": REPLAY_SAFETY_FACTOR, "smoke_gate_sha256": gate_sha, "frozen_tv_floor": gate_floor,
        "source_sha256": {p.name: file_sha256(p) for p in (
            Path(__file__), Path(__file__).with_name("matched_controls.py"), Path(__file__).with_name("native.py"),
            *(Path(__file__).parent.parent / "inkling_routing_01" / name for name in ("local.py", "checkpoint.py", "hooks.py", "core.py"))
        )},
    }
    if args.clamp_beta:
        config["clamp"] = {
            "shared_beta_rule": "FP32 midpoint of the two native K score softmax weights",
            "recovery_tolerance": CLAMP_RECOVERY_TOLERANCE,
        }
    config["manifest_sha256"] = digest(config)

    def preflight(software):
        import psutil

        if software["packages"] != NATIVE_PACKAGES or software["pytorch_cuda"] != "12.8":
            raise RuntimeError("native calibration requires the pinned software set")
        if args.device.startswith("cuda"):
            headroom(args.device, args.cuda_budget_gib, reserve_gib=reserve_gib,
                     load_ram_gib=8 if args.offload_vocabulary else 10)
        elif psutil.virtual_memory().available < CPU_MIN_AVAILABLE_GIB * 1024**3:
            raise RuntimeError("insufficient host RAM for the native CPU fallback")

    execute(args.out, config, args,
            probe_root=partial(probe, reserve_gib=reserve_gib, clamp_beta=args.clamp_beta), check_preflight=preflight)
    _, rows, _ = verified_run(args.out)
    summary_path = args.out.with_name(args.out.name + "-summary.json")
    write_json(summary_path, summarize(rows, frozen_floor=gate_floor, clamp_beta=args.clamp_beta))


if __name__ == "__main__":
    main()
