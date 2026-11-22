"""Audit whether the IO18 controller improves clean-state writes at equal soft budget."""
import argparse
from collections import defaultdict
from importlib.metadata import version
from io import BytesIO
import json
from pathlib import Path
from statistics import fmean
import sys

from experiments.nan_phase35.run import lease
from experiments.nan_readpath37.run import bound, digest, joined, write_new
from experiments.nan_refit49.run import FIT_REPORT, FIT_ARCHIVE, load_fits

from .masks import best_cal_anchor, mass_matched, relative_write_error


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
MANIFEST = HERE / "SOURCE_MANIFEST.json"
READPATH = REPO / "experiments/nan_readpath37"
SOURCE = REPO / "runs/nan_comp14"


def registration():
    paths = [HERE / name for name in (".gitattributes", "__init__.py", "PROTOCOL.md", "registration.json",
                                     "run.py", "masks.py")]
    paths += [READPATH / "registration.json", FIT_REPORT] + [REPO / name for name in (
        "experiments/nan_refit49/SOURCE_MANIFEST.json", "experiments/nan_refit49/run.py",
        "experiments/nan_fit07/pointwise.py", "experiments/nan_fit07/worker.py",
        "mechinterp/nanbeige_bpd_gym/native_components.py", "experiments/nan_readpath37/run.py",
        "experiments/nan_phase35/run.py")]
    return {"experiment": "NAN-ROUTE-50", "files_sha256": {p.relative_to(REPO).as_posix(): digest(p) for p in paths}}


def npz_fields(path, expected, keys, np):
    with np.load(BytesIO(bound(path, expected)), allow_pickle=False) as archive:
        return {key: archive[key] for key in keys}


def fit_packet(backend, base, np):
    report = json.loads(FIT_REPORT.read_bytes())
    audit = json.loads(bound(FIT_ARCHIVE / "audits" / f"{backend}.json", report["backends"][backend]["audit_sha256"]))
    measured = [cell for cell in audit["cells"] if cell["passes"] == 2]
    if len(measured) != 1 or measured[0]["status"] != "passed":
        raise ValueError("T2 routing fit lacks the independent audit")
    data = FIT_ARCHIVE / backend / "t2_layer21_down_proj_rank8"
    receipt = json.loads(bound(data / "receipt.json", measured[0]["receipt_sha256"]))
    if receipt["target_parameter_sha256_unchanged"] != base["backends"][backend]["target"]:
        raise ValueError("Saved controller has a different physical target")
    controller = ("ci__proj_in.weight", "ci__proj_in.bias", "ci__out_proj.weight", "ci__out_proj.bias")
    physics = ("V", "U", "mean_comp_acts", "mean_delta_out", "mean_write_rms")
    initial = receipt["initial_checkpoint"]
    final = receipt["final_checkpoint"]
    initial_bank = npz_fields(data / initial["file"], initial["sha256"], controller, np)
    final_bank = npz_fields(data / final["file"], final["sha256"], controller + physics, np)
    native_mask = npz_fields(data / "CI_replay_local.npz", receipt["CI_replay_sha256"], ("X", "masks"), np)
    diagnostics = json.loads(bound(data / "diagnostic_local.json", receipt["diagnostic_artifact_sha256"]["diagnostic_local.json"]))
    ablation = receipt["training_contract"]["ablation"]
    if (ablation["mode"] != "pass_mean" or ablation["sphere_rescale"] is not True
            or final["CI_config"] != initial["CI_config"]):
        raise ValueError("The T2 routing contract changed")
    return initial_bank, final_bank, final["CI_config"], ablation["eps"], native_mask, diagnostics


def state_packet(backend, base, roster, check, np):
    pin = base["backends"][backend]
    location = SOURCE / backend
    audit = json.loads(bound(SOURCE / "audits" / f"{backend}.json", pin["audit"]))
    native = json.loads(bound(location / "manifest.json", pin["manifest"]))
    if ((audit["status"], audit["worker_status"], audit["missing_contexts"], audit["contexts_checked"]) !=
            ("passed", "completed", 0, 240) or len(audit["records"]) != 240
            or native["target_parameter_sha256"] != pin["target"] or native["checkpoint_revision"] != base["revision"]):
        raise ValueError("COMP14 archive is outside its completed native audit")
    rows, x, target = [], [], []
    for slot, source in roster.items():
        check()
        lock = audit["records"][slot]
        case = json.loads(bound(location / f"case_{slot:04d}.json", lock["case_sha256"]))
        if (any(case[k] != source[k] or lock[k] != source[k] for k in ("id", "root", "split", "arm", "world"))
                or case["state_file_sha256"] != lock["state_sha256"] or case["state_file"] != f"case_{slot:04d}_states.npz"):
            raise ValueError("Native COMP14 cache identity changed")
        arrays = {(f["loop"], f["site"]): f["array_key"] for f in case["state_index"]
                  if f["layer_zero_based"] == 21 and f["site"] in {"down_proj_input", "down_proj_output"}}
        ordered = [(pass_id, site) for site in ("down_proj_input", "down_proj_output") for pass_id in (1, 2)]
        if set(arrays) != set(ordered):
            raise ValueError("Missing native final-token input or output vector")
        pack = npz_fields(location / case["state_file"], lock["state_sha256"], [arrays[k] for k in ordered], np)
        followed = [pack[arrays[key]].astype(np.float32) for key in ordered]
        rows.append({k: source[k] for k in ("id", "root", "split", "arm", "world")})
        x.append(followed[:2])
        target.append(followed[2:])
    x, target = np.asarray(x), np.asarray(target)
    if (x.shape != (144, 2, base["input_width"]) or target.shape != (144, 2, base["width"])
            or not np.isfinite(x).all() or not np.isfinite(target).all()):
        raise ValueError("Wrong cached COMP14 projection bundle")
    return rows, x, target


def native_write_closure(rows, learned, target, reads, bank, eps, old, gate, np):
    lookup = {(e["id"], e["mode"]): e["phases"][0]["actual_last_token_module_write_relative_l2"] for e in old["records"]}
    index = [i for i, row in enumerate(rows) if row["arm"] == "focused"]
    assert len(index) == 48
    errors = {}
    for mode, mask in (("clean_informed", learned[index, 0]), ("static_CAL_mean", old["static_CAL_mask"][0])):
        estimate = relative_write_error(mask, target[index, 0], reads[index, 0], bank, 0, eps, np)
        baseline = [lookup[rows[i]["id"], mode] for i in index]
        errors[mode] = float(np.max(abs(estimate - baseline)))
        if errors[mode] > gate:
            raise ValueError(f"Phase 1 {mode} write replay differs from its audited native receipt: {errors[mode]:.6g}")
    return errors


def evaluate_phase(rows, inputs, learned, initial, target, bank, diagnostics, eps, phase, cfg, np):
    reads = inputs @ bank["V"]
    cal = [i for i, row in enumerate(rows) if (row["split"], row["arm"]) == ("calibration", "focused")]
    if len(cal) != 16 or len({rows[i]["root"] for i in cal}) != 8:
        raise ValueError("Static anchor calibration roster changed")
    frozen_mean = np.asarray(diagnostics["static_CAL_mask"][phase])
    anchor, receipt = best_cal_anchor(learned[cal], target[cal], reads[cal], bank, phase, eps, frozen_mean,
                                     cfg["maximum_static_anchor_batch"], np)
    lookup = {(row["arm"], row["root"], row["world"]): i for i, row in enumerate(rows)}
    twin = [lookup[row["arm"], row["root"], "B" if row["world"] == "A" else "A"] for row in rows]
    budget = learned.sum(axis=1)
    candidates = {
        "learned": learned,
        "cal_best_static": mass_matched(anchor, budget, np),
        "cal_last_token_mean": mass_matched(learned[cal].mean(axis=0), budget, np),
        "archived_CAL_all_token_mean": mass_matched(frozen_mean, budget, np),
        "twin_swap": mass_matched(learned[twin], budget, np),
        "initial_controller": mass_matched(initial, budget, np),
    }
    errors = {key: relative_write_error(value, target, reads, bank, phase, eps, np) for key, value in candidates.items()}
    mass_error = max(float(np.max(abs(mask.sum(axis=1) - budget))) for mask in candidates.values())
    private = []
    for i, row in enumerate(rows):
        private.append(row | {"soft_budget": float(budget[i]),
            "near_zero_mask_slots": int(sum(learned[i] < .01)), "near_one_mask_slots": int(sum(learned[i] > .99)),
            "max_raw_twin_mask_delta": float(abs(learned[i] - learned[twin[i]]).max()),
            "relative_write_error": {key: float(value[i]) for key, value in errors.items()}})
    return private, receipt | {"max_abs_budget_error_all_rows": mass_error}


def summarize(rows, np):
    groups = defaultdict(dict)
    for row in rows:
        groups[row["root"]][row["world"]] = row
    if any(set(pair) != {"A", "B"} for pair in groups.values()):
        raise ValueError("Calibration or evaluation root is missing its twin")
    keys = list(rows[0]["relative_write_error"])
    root_scores = [{key: fmean(row["relative_write_error"][key] for row in pair.values()) for key in keys}
                   for pair in groups.values()]
    a, b = keys[0], keys[1]
    difference = [item[a] - item[b] for item in root_scores]
    return {"roots": len(groups), "worlds": len(rows),
            "relative_write_error": {key: fmean(item[key] for item in root_scores) for key in keys},
            "learned_minus_CAL_best_static": fmean(difference),
            "learned_wins_vs_CAL_best_static_roots": sum(v < -1e-8 for v in difference),
            "static_wins_vs_learned_roots": sum(v > 1e-8 for v in difference),
            "twin_swap_exact_mask_worlds": sum(row["max_raw_twin_mask_delta"] == 0 for row in rows),
            "mean_twin_max_mask_delta": fmean(row["max_raw_twin_mask_delta"] for row in rows),
            "mean_soft_budget": fmean(row["soft_budget"] for row in rows),
            "mean_near_zero_slots": fmean(row["near_zero_mask_slots"] for row in rows),
            "mean_near_one_slots": fmean(row["near_one_mask_slots"] for row in rows)}


def run_backend(backend, base, source, cfg, masks, check, np):
    initial, bank, ci, eps, replay, diagnostics = fit_packet(backend, base, np)
    ci_error = float(np.max(abs(masks(replay["X"], bank, ci) - replay["masks"])))
    if ci_error > cfg["native_controller_gate_abs_tolerance"]:
        raise ValueError("Pointwise CI does not reproduce the archived native mask fixture")
    rows, x, target = state_packet(backend, base, source, check, np)
    learned, init = (masks(x, tagged, ci) for tagged in (bank, initial))
    native = native_write_closure(rows, learned, target, x @ bank["V"], bank, eps,
                                  diagnostics, cfg["native_phase1_write_gate_abs_tolerance"], np)
    result, private = {}, {}
    for phase in (0, 1):
        check()
        records, control = evaluate_phase(rows, x[:, phase], learned[:, phase], init[:, phase], target[:, phase],
                                          bank, diagnostics, eps, phase, cfg, np)
        private[str(phase + 1)] = records
        arms = {(r["split"], r["arm"]) for r in records}
        summary = {f"{split}/{arm}": summarize([r for r in records if (r["split"], r["arm"]) == (split, arm)], np)
                   for split, arm in sorted(arms)}
        result[str(phase + 1)] = {"static_prefit": control, "arms": summary}
    return {"native_CI_max_abs_error": ci_error, "phase1_native_write_max_abs_errors": native, "passes": result}, private


def verification(np):
    masks = np.array([[.2, .4, .9], [.5, .1, .7]])
    targets = np.array([0., 2.1])
    result = mass_matched(masks, targets, np)
    np.testing.assert_allclose(result.sum(axis=1), targets, rtol=0, atol=1e-12)
    np.testing.assert_allclose(result[0], 0., rtol=0, atol=1e-12)
    np.testing.assert_allclose(mass_matched(np.zeros(3), 3., np), np.ones(3), rtol=0, atol=1e-12)
    assert np.all(result >= 0) and np.all(result <= 1)
    return {"capped_simplex_mass_and_box_verified": True}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--freeze", action="store_true")
    cli.add_argument("--output", type=Path)
    args = cli.parse_args()
    if not __debug__:
        raise RuntimeError("Use assertions for native archive custody")
    if args.freeze:
        write_new(MANIFEST, registration())
        print(digest(MANIFEST))
        return
    if args.output is None:
        cli.error("--output is required")
    if json.loads(MANIFEST.read_bytes()) != registration():
        raise ValueError("The ROUTE50 replay source has changed")
    cfg = json.loads((HERE / "registration.json").read_bytes())
    base = json.loads((READPATH / "registration.json").read_bytes())
    with lease(cfg) as (check, samples):
        if version("numpy") != "2.5.5":
            raise ValueError("Pinned numerical runtime changed")
        import numpy as np
        from experiments.nan_fit07.pointwise import masks

        proof = verification(np)
        load_fits(base, np)
        source, _ = joined(base)
        outcomes, private = {}, {}
        for backend in base["backends"]:
            outcomes[backend], private[backend] = run_backend(backend, base, source, cfg, masks, check, np)
            print(f"{backend}: matched soft-budget controls scored both physical passes", flush=True)
        check()
        if "torch" in sys.modules:
            raise RuntimeError("The CPU-matched routing replay imported Torch")
    args.output.mkdir(parents=True, exist_ok=False)
    out = args.output / "private.json"
    write_new(out, private)
    write_new(args.output / "aggregate.json", {
        "experiment": cfg["experiment"], "status": "complete", "source_manifest_sha256": digest(MANIFEST),
        "private_receipt_sha256": digest(out), "verification": proof, "per_backend": outcomes,
        "new_pretrained_forwards": 0, "formal_comparisons": 0, "sealed_test_accessed": False,
        "resource": {"max_sampled_private_bytes": max(v["private_bytes"] for v in samples),
                     "max_sampled_working_set_bytes": max(v["working_set_bytes"] for v in samples),
                     "min_sampled_available_RAM_bytes": min(v["available_physical_bytes"] for v in samples)},
    })


if __name__ == "__main__":
    main()
