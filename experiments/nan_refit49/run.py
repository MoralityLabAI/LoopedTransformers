"""Audit initializer inheritance and transport agreement in the four IO18 banks."""
import argparse
from importlib.metadata import version
from io import BytesIO
from itertools import permutations
import json
from pathlib import Path
from statistics import fmean, median
import sys

from experiments.nan_phase35.run import lease
from experiments.nan_readpath37.run import bound, case_states, digest, joined, write_new
from .geometry import assignment, atom_geometry, cos, difference, geometry, joint_coordinates, projection_fractions


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
MANIFEST = HERE / "SOURCE_MANIFEST.json"
FIT = REPO / "experiments/nan_fit16_io18"
FIT_ARCHIVE = REPO / "runs/nan_fit16_io18"
FIT_REPORT = REPO / "reports/nanbeige_bpd_gym/nan_fit16_io18_progress.json"
READPATH = REPO / "experiments/nan_readpath37"
SOURCE = REPO / "runs/nan_comp14"


def registration():
    parents = (
        FIT / "SOURCE_MANIFEST.json", FIT_REPORT,
        READPATH / "registration.json", READPATH / "SOURCE_MANIFEST.json", READPATH / "run.py",
        REPO / "experiments/nan_phase35/run.py",
    )
    paths = [HERE / name for name in (".gitattributes", "__init__.py", "PROTOCOL.md", "registration.json",
                                     "run.py", "geometry.py")] + list(parents)
    return {"experiment": "NAN-REFIT-49", "files_sha256": {p.relative_to(REPO).as_posix(): digest(p) for p in paths}}


def factors(path, expected, keys, shape, np):
    with np.load(BytesIO(bound(path, expected)), allow_pickle=False) as z:
        values = tuple(z[name].astype(np.float64) for name in keys)
    if (tuple(x.shape for x in values) != shape or any(not np.isfinite(x).all() for x in values)
            or np.linalg.matrix_rank(values[0]) != shape[0][1] or np.linalg.matrix_rank(values[1]) != shape[0][1]):
        raise ValueError("Physical factor widths, rank or finite values changed")
    return values


def load_fits(base, np):
    published = json.loads(FIT_REPORT.read_bytes())
    manifest = FIT / "SOURCE_MANIFEST.json"
    source_sha = digest(manifest)
    if published["source_manifest_sha256"] != source_sha:
        raise ValueError("IO18 result summary refers to a different source")
    spec = json.loads(manifest.read_bytes())
    dimensions = ((base["input_width"], base["rank"]), (base["rank"], base["width"]))
    banks = {}
    for backend, old in published["backends"].items():
        audited = json.loads(bound(FIT_ARCHIVE / "audits" / f"{backend}.json", old["audit_sha256"]))
        if (old["status"] != "completed_audited" or audited["status"] != "passed"
                or audited["receipt_sha256"] != old["receipt_sha256"] or audited["source_manifest_sha256"] != source_sha):
            raise ValueError("Fit does not have a closed matching independent audit")
        initial = factors(REPO / "runs/nan_gym05" / backend / "linear_baselines/layer21_down_proj.npz",
                          spec["initialization"][backend]["file_sha256"], ("V8", "U8"), dimensions, np)
        for cell in audited["cells"]:
            phase = cell["passes"]
            parent = FIT_ARCHIVE / backend / f"t{phase}_layer21_down_proj_rank8"
            receipt = json.loads(bound(parent / "receipt.json", cell["receipt_sha256"]))
            if (cell["status"] != "passed" or receipt["status"] != "completed"
                    or receipt["source_manifest_sha256"] != source_sha
                    or receipt["target_parameter_sha256_unchanged"] != base["backends"][backend]["target"]
                    or receipt["backend"] != backend or receipt["passes"] != phase
                    or not receipt["native_all_one_full_head_bit_exact"] or receipt["optimizer_steps_completed"] != 16):
                raise ValueError("Fit terminal did not preserve the native target or optimizer gate")
            values = []
            for checkpoint, counter in (("initial_checkpoint", 0), ("final_checkpoint", 16)):
                item = receipt[checkpoint]
                if (item["optimizer_steps_completed"] != counter or (checkpoint == "final_checkpoint"
                        and cell["checkpoint_sha256"] != item["sha256"])):
                    raise ValueError("Independent checkpoint audit differs")
                values.append(factors(parent / item["file"], item["sha256"], ("V", "U"), dimensions, np))
            if any(not np.array_equal(a, b) for a, b in zip(initial, values[0], strict=True)):
                raise ValueError("A fit did not start from its registered SVD")
            banks[f"{backend}/T{phase}"] = tuple(values)
    if set(banks) != {f"{backend}/T{i}" for backend in base["backends"] for i in (1, 2)}:
        raise ValueError("All four audited fits are required")
    return banks


def captured_pairs(backend, base, roster, check, np):
    pin = base["backends"][backend]
    folder = SOURCE / backend
    audit = json.loads(bound(SOURCE / "audits" / f"{backend}.json", pin["audit"]))
    manifest = json.loads(bound(folder / "manifest.json", pin["manifest"]))
    if ((audit["status"], audit["worker_status"], audit["missing_contexts"], audit["contexts_checked"])
            != ("passed", "completed", 0, 240) or len(audit["records"]) != 240
            or manifest["target_parameter_sha256"] != pin["target"] or manifest["checkpoint_revision"] != base["revision"]):
        raise ValueError("Captured COMP14 inputs do not have the native audit")
    grouped = {}
    for i, source in roster.items():
        check()
        verified = audit["records"][i]
        path = f"case_{i:04d}"
        case = json.loads(bound(folder / (path + ".json"), verified["case_sha256"]))
        if any(verified[k] != source[k] or case[k] != source[k] for k in ("id", "root", "split", "arm", "world")):
            raise ValueError("Cached input is outside the frozen development pair")
        if case["state_file"] != path + "_states.npz" or case["state_file_sha256"] != verified["state_sha256"]:
            raise ValueError("State filename or custody mismatch")
        state = case_states(case, bound(folder / case["state_file"], verified["state_sha256"]), np)
        group = grouped.setdefault(source["root"], {})
        if source["world"] in group:
            raise ValueError("Duplicate world in a fixed source contrast")
        group[source["world"]] = [state[phase, "down_proj_input"].astype(np.float64) for phase in (1, 2)]
    if any(set(group) != {"A", "B"} for group in grouped.values()):
        raise ValueError("Missing member of a cached native A/B pair")
    return {root: [b - a for a, b in zip(worlds["A"], worlds["B"], strict=True)] for root, worlds in grouped.items()}


def transport(a0, a, b0, b, x, y, mapping, separate_carrier, np):
    def apply(inputs, bank, keep=None):
        V, U = bank
        return (inputs @ V) @ U if keep is None else (inputs @ V[:, keep]) @ U[keep]

    init_a, init_b = apply(x, a0), apply(y, b0)
    final_a, final_b = apply(x, a), apply(y, b)
    da, db = final_a - init_a, final_b - init_b

    def relative(change, before):
        den = float(np.linalg.norm(before))
        return float(np.linalg.norm(change) / den) if den > 1e-15 else None

    result = {"initial_donor_cosine": cos(init_a, init_b, np), "final_donor_cosine": cos(final_a, final_b, np),
            "fitted_seven_write_cosine": cos(apply(x, a, list(range(1, 8))), apply(y, b, mapping[1:]), np),
            "update_only_donor_cosine": cos(da, db, np),
            "reference_learned_update_over_initial": relative(da, init_a),
            "comparison_learned_update_over_initial": relative(db, init_b)}
    if separate_carrier:
        a_on_y, b_on_x = apply(y, a), apply(x, b)
        state = .5 * ((final_a - a_on_y) + (b_on_x - final_b))
        operator = .5 * ((final_a - b_on_x) + (a_on_y - final_b))
        total = final_a - final_b
        np.testing.assert_allclose(state + operator, total, atol=1e-10, rtol=1e-10)
        sq = float(total @ total)
        result.update({
            "native_input_difference_cosine": cos(x, y, np),
            "carrier_change_norm_over_total_disagreement": relative(state, total),
            "operator_change_norm_over_total_disagreement": relative(operator, total),
            "carrier_signed_projection_share": float(state @ total) / sq if sq > 1e-30 else None,
            "operator_signed_projection_share": float(operator @ total) / sq if sq > 1e-30 else None,
        })
    return result


def summarize(records):
    out = {"roots": len(records)}
    for key in records[0]:
        values = [x[key] for x in records if x[key] is not None]
        out[key] = {"defined_roots": len(values), "median": median(values) if values else None,
                    "mean": fmean(values) if values else None, "minimum": min(values) if values else None,
                    "negative_roots": sum(v < 0 for v in values)}
    return out


def compare(a, b, bank, inputs, phases, check, np):
    a0, fa = bank[a]
    b0, fb = bank[b]
    shape = geometry(a0, fa, b0, fb, np)
    u, v = (key.split("/")[0] for key in (a, b))
    cases = {f"fixed_{u}_inputs": (u, u)}
    if u != v:
        cases |= {f"fixed_{v}_inputs": (v, v), "paired_native_inputs": (u, v)}
    rows, aggregated = {}, {}
    for carrier, (x, y) in cases.items():
        if set(inputs[x]) != set(inputs[y]):
            raise ValueError("Precision input carriers do not have the same DEV roots")
        per_phase = {}
        for phase in phases:
            check()
            cases_by_root = {r: transport(a0, fa, b0, fb, inputs[x][r][phase - 1], inputs[y][r][phase - 1],
                                          shape["final_assignment"], x != y, np) for r in sorted(inputs[x])}
            rows[carrier, phase] = cases_by_root
            per_phase[str(phase)] = summarize(list(cases_by_root.values()))
        aggregated[carrier] = per_phase
    return {"geometry": shape, "captured_transport": aggregated}, {
        f"{carrier}/pass{phase}": value for (carrier, phase), value in rows.items()}


def verification(np):
    rng = np.random.default_rng(49)
    v = np.linalg.qr(rng.normal(size=(9, 3)), mode="reduced")[0]
    u = np.linalg.qr(rng.normal(size=(8, 3)), mode="reduced")[0].T
    initial = v, u
    final = v.copy(), u.copy()
    final[0][:, 0] *= 1.5
    delta = difference(final, initial, np)
    output = projection_fractions(initial, delta, np)
    np.testing.assert_allclose(list(output.values()), [1, 1], atol=1e-12, rtol=0)
    order = [2, 0, 1]
    scale = np.array([-2.0, 4.0, .5])
    mix = v[:, order] * scale, u[order] / scale[:, None]
    _, _, atom = atom_geometry(initial, mix, np)
    best = assignment(atom)
    exhaustive = min(permutations(range(3)), key=lambda order: (-sum(abs(atom[i, j]) for i, j in enumerate(order)), order))
    assert best == tuple(exhaustive) == (1, 2, 0)
    compact, peer = joint_coordinates(delta, initial, np)
    dense = final[0] @ final[1] - v @ u
    np.testing.assert_allclose(cos(compact, peer, np), cos(dense, v @ u, np), atol=1e-12, rtol=0)
    np.testing.assert_allclose(np.linalg.norm(compact), np.linalg.norm(dense), atol=1e-12, rtol=0)
    return {"exact_assignment_brute_force": True, "reciprocal_gauge_recovered": True,
            "dense_update_norm_and_cosine_verified": True, "initial_atom_projection_verified": True}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--freeze", action="store_true")
    cli.add_argument("--output", type=Path)
    args = cli.parse_args()
    if not __debug__:
        raise RuntimeError("Use assertions to verify frozen array contents")
    if args.freeze:
        write_new(MANIFEST, registration())
        print(digest(MANIFEST))
        return
    if args.output is None:
        cli.error("--output is required")
    if json.loads(MANIFEST.read_bytes()) != registration():
        raise ValueError("REFIT49 source freeze changed")
    base = json.loads((READPATH / "registration.json").read_bytes())
    cfg = json.loads((HERE / "registration.json").read_bytes())
    with lease(cfg) as (check, samples):
        if version("numpy") != "2.5.5":
            raise ValueError("NumPy runtime pin changed")
        import numpy as np

        tests = verification(np)
        data, _ = joined(base)
        focused = {i: row for i, row in data.items() if (row["split"], row["arm"]) == ("development", "focused")}
        if len(focused) != 32 or len({row["root"] for row in focused.values()}) != 16:
            raise ValueError("Physical writing audit requires exactly sixteen exposed DEV roots")
        bank = load_fits(base, np)
        captured = {b: captured_pairs(b, base, focused, check, np) for b in base["backends"]}
        summary, details = {}, {}
        for a, b in cfg["comparisons"]:
            name = a + "_" + b
            summary[name], details[name] = compare(a, b, bank, captured, cfg["phase_observations"], check, np)
        check()
        if any(name == "torch" or name.startswith("torch.") for name in sys.modules):
            raise RuntimeError("The archive audit imported Torch")
    args.output.mkdir(parents=True, exist_ok=False)
    private = args.output / "private.json"
    write_new(private, details)
    write_new(args.output / "aggregate.json", {
        "experiment": cfg["experiment"], "status": "complete", "source_manifest_sha256": digest(MANIFEST),
        "private_pairing_sha256": digest(private), "geometry_verification": tests, "fit_recipe_runs": len(bank),
        "exposed_DEV_roots": 16, "new_pretrained_forwards": 0, "formal_comparisons": 0, "sealed_test_accessed": False,
        "comparisons": summary, "resource": {"max_sampled_private_bytes": max(v["private_bytes"] for v in samples),
                                           "max_sampled_working_set_bytes": max(v["working_set_bytes"] for v in samples),
                                           "min_sampled_available_RAM_bytes": min(v["available_physical_bytes"] for v in samples)},
    })
    print(f"Verified {len(bank)} fits and {len(focused)} native DEV worlds per precision", flush=True)


if __name__ == "__main__":
    main()
