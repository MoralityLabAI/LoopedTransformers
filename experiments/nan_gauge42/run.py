"""Audit native twin contrasts and gauge sensitivity using pinned COMP14 archives."""
import argparse
from collections import defaultdict
from importlib.metadata import version
from io import BytesIO
import json
from pathlib import Path
import sys

from experiments.nan_delay39.graph_host import OCCUPANCIES
from experiments.nan_phase35.run import lease
from experiments.nan_readpath37.run import FixedHead, SHARD, bound, case_states, digest, joined, write_new
from .transport import fit, gauge_family, head_gradient, in_gauge, orthonormal_bank


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
MANIFEST = HERE / "SOURCE_MANIFEST.json"
READPATH = REPO / "experiments/nan_readpath37"
ARCHIVE = REPO / "runs/nan_comp14"
BANKS = REPO / "runs/nan_fit16_io18"


def seal():
    paths = [HERE / name for name in (".gitattributes", "__init__.py", "PROTOCOL.md", "registration.json",
                                     "run.py", "transport.py")]
    paths += [REPO / name for name in ("experiments/nan_readpath37/run.py",
                                       "experiments/nan_readpath37/registration.json",
                                       "experiments/nan_readpath37/SOURCE_MANIFEST.json",
                                       "experiments/nan_delay39/graph_host.py", "experiments/nan_delay39/source.py",
                                       "experiments/nan_phase35/run.py", "experiments/nan_ram28/resources.py",
                                       "experiments/nan_ranksite13/resources.py")]
    return {"experiment": "NAN-GAUGE-42",
            "files_sha256": {path.relative_to(REPO).as_posix(): digest(path) for path in paths}}


def read_archive(backend, cfg, roster, head, check, np):
    pins = cfg["backends"][backend]
    folder = ARCHIVE / backend
    audit = json.loads(bound(ARCHIVE / "audits" / f"{backend}.json", pins["audit"]))
    attest = json.loads(bound(folder / "manifest.json", pins["manifest"]))
    bank_dir = BANKS / backend / "t2_layer21_down_proj_rank8"
    receipt = json.loads(bound(bank_dir / "receipt.json", pins["factor_receipt"]))
    if ((audit["status"], audit["worker_status"], audit["missing_contexts"], audit["contexts_checked"]) !=
            ("passed", "completed", 0, 240) or len(audit["records"]) != 240
            or attest["checkpoint_revision"] != cfg["revision"]
            or attest["target_parameter_sha256"] != pins["target"]
            or receipt["target_parameter_sha256_unchanged"] != pins["target"]
            or receipt["optimizer_steps_completed"] != 16):
        raise ValueError("Archive or fitted factors differ from their native custody")
    with np.load(BytesIO(bound(bank_dir / "final.npz", pins["factor_bank"])), allow_pickle=False) as bank:
        reads, writes = (bank[key].astype(np.float64) for key in ("V", "U"))
    if (reads.shape != (cfg["input_width"], cfg["rank"]) or writes.shape != (cfg["rank"], cfg["width"])
            or not np.isfinite(reads).all() or not np.isfinite(writes).all()):
        raise ValueError("Unexpected bank shape or nonfinite weights")

    twins = defaultdict(dict)
    closure = 0.0
    for slot, row in roster.items():
        check()
        verified = audit["records"][slot]
        case = json.loads(bound(folder / f"case_{slot:04d}.json", verified["case_sha256"]))
        if any(row[key] != case[key] or row[key] != verified[key]
               for key in ("id", "root", "world", "split", "arm")):
            raise ValueError("Case identity differs from frozen roster")
        path = f"case_{slot:04d}_states.npz"
        if case["state_file"] != path or case["state_file_sha256"] != verified["state_sha256"]:
            raise ValueError("State identity differs from the archive audit")
        states = case_states(case, bound(folder / path, verified["state_sha256"]), np)
        for phase, recorded in enumerate(case["initial_phases"], 1):
            if recorded["loop"] != phase:
                raise ValueError("Native phase order changed")
            scores = recorded["score"]
            saved = {int(i): value for i, value in scores["selected_logits"].items()}
            saved[scores["free_top_token_id"]] = scores["free_top_token_logit"]
            replay = head.logits(states[phase, "residual_after_layer"], list(saved))
            closure = max(closure, float(np.max(abs(replay - list(saved.values())))))
        key = (row["split"], row["arm"], row["root"])
        if row["world"] in twins[key]:
            raise ValueError("Repeated world in source pair")
        twins[key][row["world"]] = (row["answer"], states, case["initial_prompt_ids_sha256"])
    if closure > cfg["native_logit_abs_tol"] or len(twins) * 2 != len(roster):
        raise ValueError("Native head closure or twin coverage failed")
    return twins, reads, writes, closure


def extract_pair(key, worlds, label_ids, head, reads, root, writes, np):
    if set(worlds) != {"A", "B"} or worlds["A"][0] == worlds["B"][0]:
        raise ValueError("Expected distinct A/B oracle labels")
    at, a, ahash = worlds["A"]
    bt, b, bhash = worlds["B"]
    token_ids = [label_ids[at], label_ids[bt]]
    direction = head.gamma.astype(np.float64) * (
        head.row(token_ids[0]).astype(np.float64) - head.row(token_ids[1]).astype(np.float64))
    records = []
    for phase in (1, 2):
        h = np.asarray([states[phase, "residual_after_layer"] for states in (a, b)], dtype=np.float64)
        x_a, x_b = (states[phase, "down_proj_input"].astype(np.float64) for states in (a, b))
        delta = (x_a - x_b) @ reads
        white_coeff = delta @ root
        projected = white_coeff @ writes
        midpoint = h.mean(axis=0)
        grad = head_gradient(midpoint, direction, head.eps, np)
        finite_difference_error = verify_gradient(midpoint, direction, projected, head.eps, np)
        logits = head.logits(h, token_ids).astype(np.float64)
        margin = logits @ np.array([1.0, -1.0])
        records.append({"coefficients": white_coeff, "gradient": grad,
                        "write_tangent": float(projected @ grad), "native_margin_swing": float(margin[0] - margin[1]),
                        "finite_difference_error": finite_difference_error})
    if key[1] == "focused_erased":
        if ahash != bhash or any(record["coefficients"].any() or record["native_margin_swing"] != 0
                                 for record in records):
            raise ValueError("Erased twin output no longer closes exactly")
    return records


def audit_gauges(left, target, gradient, transfer, whitened, gauges, close, np):
    canonical_left = left @ whitened
    canonical_target = target @ whitened
    canonical_prediction = (left @ transfer) @ whitened
    full = float(canonical_prediction @ gradient)
    results = {}
    for name, change, inverse in gauges:
        try:
            incoming, actual, predicted, graph = in_gauge(
                left, target, whitened, transfer, gradient, change, inverse, np)
            close(incoming, canonical_left, "input physical write")
            close(actual, canonical_target, "target physical write")
            close(predicted, canonical_prediction, "predicted physical write")
            close(graph[-1]["signed_path_product_sum"], full, "full signed path sum")
        except AssertionError as error:
            raise ValueError(f"Coordinate closure failed in {name}") from error
        results[name] = graph
    for native, reciprocal in zip(results["native"], results["reciprocal"]):
        close([native[k] for k in native], [reciprocal[k] for k in native], "reciprocal graph")
    return results


def max_error(np, cfg):
    def close(observed, expected, context):
        close.maximum = max(close.maximum, float(np.max(abs(np.asarray(observed) - np.asarray(expected)))))
        np.testing.assert_allclose(observed, expected, atol=cfg["numerical_atol"],
                                   rtol=cfg["numerical_rtol"], err_msg=context)
    close.maximum = 0.0
    return close


def summarize(rows, cutoffs, np):
    names = tuple(rows[0]["physical_squared_error"])
    denominator = sum(row["target_energy"] for row in rows)
    tangent_denominator = sum(row["target_tangent"]**2 for row in rows)
    conditions = []
    eps = 1e-10
    for i, cutoff in enumerate(cutoffs):
        native = [row["graphs"]["native"][i] for row in rows]
        rotations = [[graph[i]["signed_path_product_sum"] for name, graph in row["graphs"].items()
                      if name.startswith("rotation_")] for row in rows]
        amplitudes = [max(values) - min(values) for values in rotations]
        conditions.append({"rank_cutoff": cutoff, "native_path_degree_ties": sum(
            cell["exact_membership_tie"] for cell in native),
            "native_mean_paths": float(np.mean([cell["through_paths"] for cell in native])),
            "rotation_sensitive_roots": sum(delta > eps for delta in amplitudes),
            "rotation_sign_reversal_roots": sum(min(values) < -eps and max(values) > eps for values in rotations),
            "mean_rotation_range_logit": float(np.mean(amplitudes)),
            "max_rotation_range_logit": max(amplitudes)})
    return {
        "roots": len(rows),
        "phase1_native_mean_absolute_margin_swing": float(np.mean([abs(row["native_swing"][0]) for row in rows])),
        "phase2_native_mean_absolute_margin_swing": float(np.mean([abs(row["native_swing"][1]) for row in rows])),
        "phase1_mean_absolute_BPD_write_tangent": float(np.mean([abs(row["source_tangent"]) for row in rows])),
        "phase2_mean_absolute_BPD_write_tangent": float(np.mean([abs(row["target_tangent"]) for row in rows])),
        "relative_physical_squared_error": {
            name: sum(row["physical_squared_error"][name] for row in rows) / denominator if denominator else None
            for name in names},
        "relative_direct_tangent_squared_error": {
            name: sum(row["direct_tangent_squared_error"][name] for row in rows) / tangent_denominator
            if tangent_denominator else None for name in names},
        "cutoffs": conditions,
    }


def evaluate(backend, base, cfg, roster, label_ids, head, output, check, np):
    twins, reads, native, logit_error = read_archive(backend, base, roster, head, check, np)
    root, inverse, whitened = orthonormal_bank(native, np)
    close = max_error(np, cfg)
    close(whitened @ whitened.T, np.eye(base["rank"]), "shared orthonormal span")
    pairs = {key: extract_pair(key, values, label_ids, head, reads, root, whitened, np)
             for key, values in sorted(twins.items())}
    sizes = {(split, arm): sum(key[:2] == (split, arm) for key in pairs)
             for split in base["roots"] for arm in base["arms"]}
    if sizes != {(split, arm): size for split, size in base["roots"].items() for arm in base["arms"]}:
        raise ValueError("Missing independent source pairs")
    cal = [v for (split, arm, _), v in pairs.items() if (split, arm) == ("calibration", "focused")]
    c1, c2 = (np.stack([row[i]["coefficients"] for row in cal]) for i in (0, 1))
    transfer = fit(c1, c2, cfg["ridge_factor"], np)
    shifted = fit(c1, np.roll(c2, 1, axis=0), cfg["ridge_factor"], np)
    gauges = tuple(gauge_family(root, inverse, base["rank"], cfg["orthogonal_rotations"], cfg["rotation_seed"], np))
    for name, a, b in gauges:
        close(a @ b, np.eye(base["rank"]), "inverse coordinate map")
        close(((reads @ root) @ a) @ b, reads @ root, "whole factor operator coordinates")
        if name.startswith("rotation_"):
            close(fit(c1 @ a, c2 @ a, cfg["ridge_factor"], np), b @ transfer @ a, "orthogonal ridge equivariance")

    result = defaultdict(list)
    private = []
    for (split, arm, puzzle), (i, f) in pairs.items():
        check()
        left, target, gradient = i["coefficients"], f["coefficients"], f["gradient"]
        candidates = {"zero": np.zeros_like(target), "carry_pass1": left,
                      "cal_transfer": left @ transfer, "cal_root_shift": left @ shifted}
        tangent_coordinates = whitened @ gradient
        record = {
            "native_swing": [i["native_margin_swing"], f["native_margin_swing"]],
            "source_tangent": i["write_tangent"], "target_tangent": f["write_tangent"],
            "target_energy": float(target @ target),
            "physical_squared_error": {name: float(np.linalg.norm(value - target)**2)
                                       for name, value in candidates.items()},
            "direct_tangent_squared_error": {name: float(((value - target) @ tangent_coordinates)**2)
                                              for name, value in candidates.items()},
            "graphs": audit_gauges(left, target, gradient, transfer, whitened, gauges, close, np),
        }
        result[f"{split}/{arm}"].append(record)
        private.append({"split": split, "arm": arm, "root": puzzle, **record})
    write_new(output / f"{backend}_private.json", private)
    return {"native_logit_replay_max_abs_error": logit_error, "numerical_closure_max_abs_error": close.maximum,
            "native_state_gradient_max_abs_error": max(row["finite_difference_error"]
                                                        for pair in pairs.values() for row in pair),
            "fitted_CAL_roots": len(cal), "coordinate_systems": len(gauges),
            "groups": {key: summarize(rows, cfg["cutoffs"], np) for key, rows in result.items()}}


def verify_gradient(h, w, v, eps, np):
    norm = np.linalg.norm(v)
    if not norm:
        return 0.0
    v = v / norm
    step = 1e-4 * max(1, np.sqrt(np.mean(h*h)))

    def obs(x):
        return (x @ w) / np.sqrt(np.mean(x*x) + eps)

    actual = (obs(h + step*v) - obs(h - step*v)) / (2*step)
    predicted = head_gradient(h, w, eps, np) @ v
    np.testing.assert_allclose(actual, predicted, atol=1e-8, rtol=1e-7)
    return float(abs(actual - predicted))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--freeze", action="store_true")
    cli.add_argument("--snapshot", type=Path)
    cli.add_argument("--output", type=Path)
    args = cli.parse_args()
    if not __debug__:
        raise RuntimeError("Use assertions for the pinned head and archive")
    if args.freeze:
        write_new(MANIFEST, seal())
        print(digest(MANIFEST))
        return
    if not (args.snapshot and args.output):
        cli.error("--snapshot and --output are required")
    if json.loads(MANIFEST.read_bytes()) != seal():
        raise ValueError("Frozen dependencies changed")
    base = json.loads((READPATH / "registration.json").read_bytes())
    cfg = json.loads((HERE / "registration.json").read_bytes())
    if tuple(cfg["cutoffs"]) != OCCUPANCIES:
        raise ValueError("Registered rank fractions changed")
    if args.snapshot.name != base["revision"]:
        raise ValueError("Wrong pretrained snapshot")
    with lease(cfg) as (check, samples):
        if (version("numpy"), version("tokenizers")) != ("2.5.5", "0.20.3"):
            raise ValueError("Unexpected numerical environment")
        import numpy as np
        from tokenizers import Tokenizer

        h, w, v = np.random.default_rng(42022).normal(size=(3, 12))
        error = verify_gradient(h, w, v, 1e-6, np)
        bound(args.snapshot / "tokenizer.json", base["snapshot_pins"]["tokenizer.json"])
        tok = Tokenizer.from_file(str(args.snapshot / "tokenizer.json"))
        ids = {key: tok.encode(key, add_special_tokens=False).ids for key in "ABCDEFGHIJKLMNOP"}
        if any(len(value) != 1 for value in ids.values()) or len(set(v[0] for v in ids.values())) != 16:
            raise ValueError("Bare-label tokenization changed")
        roster, _ = joined(base)
        args.output.mkdir(parents=True, exist_ok=False)
        with (args.snapshot / SHARD).open("rb") as stream:
            head = FixedHead(stream, args.snapshot, base, np, check)
            label_ids = {key: values[0] for key, values in ids.items()}
            outcomes = {backend: evaluate(backend, base, cfg, roster, label_ids, head, args.output, check, np)
                        for backend in base["backends"]}
        check()
        if any(name == "torch" or name.startswith("torch.") for name in sys.modules):
            raise RuntimeError("CPU archive audit imported Torch")
    write_new(args.output / "aggregate.json", {
        "experiment": cfg["experiment"], "status": "completed", "source_manifest_sha256": digest(MANIFEST),
        "roots": base["roots"], "per_backend": outcomes, "finite_difference_gradient_error": error,
        "new_pretrained_forwards": 0, "formal_comparisons": 0, "sealed_test_accessed": False,
        "resource": {"max_sampled_private_bytes": max(v["private_bytes"] for v in samples),
                     "max_sampled_working_set_bytes": max(v["working_set_bytes"] for v in samples),
                     "min_sampled_available_RAM_bytes": min(v["available_physical_bytes"] for v in samples),
                     "sample_count": len(samples)},
    })
    print("Verified native head, shared factor action, fitted transport and signed graph closure", flush=True)


if __name__ == "__main__":
    main()
