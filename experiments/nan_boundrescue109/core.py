"""Connect the archived control rule to fresh native B2 counterfactual states."""

from hashlib import sha256
import json
from pathlib import Path

from experiments.nan_attain101 import run as attain
from experiments.nan_attain101.geometry import attained, measure, sphere_pair, unit_directions
from experiments.nan_bankdose99.geometry import bind_banks, materialize
from experiments.nan_centersham108.run import EXPERIMENT as SHAM_SEED
from experiments.nan_centersham108.search import select
from experiments.nan_commonhead102.run import centered
from experiments.nan_headswap107.search import design_ids
from experiments.nan_radial76.native import WRITER, check_writer
from experiments.nan_readpath37.run import digest
from experiments.nan_refit49.run import load_fits
from experiments.nan_writer77.pipeline import matched_duplicate

from .native import execute_arm, grade_first_head


REPO = Path(__file__).resolve().parents[2]
PARENT = REPO / "reports/nanbeige_bpd_gym/nan_attain101_results.json"
ARMS = ("active_t1", "active_t2", "sham_t1", "sham_t2")
WORLDS = ("A", "B")
SIGNS = (1, -1)


def certified_planner(backend, design_head, np):
    report = json.loads(PARENT.read_bytes())
    frozen = attain.registered()
    if (frozen != json.loads(attain.MANIFEST.read_bytes())
            or report["source_manifest_sha256"] != digest(attain.MANIFEST)):
        raise ValueError("The frozen ATTAIN101 source differs")
    chosen = report["calibration"][backend]["locked"]
    if chosen is None:
        raise ValueError("This backend has no calibrated active arm")
    base = json.loads(attain.dose.BASE.read_bytes())
    fits = load_fits(base, np)
    banks = bind_banks(fits[f"{backend}/T1"], fits[f"{backend}/T2"], np)
    cfg = frozen["dose"]
    ids = design_ids(design_head.vocab, np)
    weighted = np.array([design_head.row(i) for i in ids], dtype=np.float64) * design_head.gamma

    def plan(anchor, root, check):
        before = np.stack([anchor["writer_raw"][1, i].float().numpy() for i in (0, 1)]).astype(np.float64)
        inputs = np.stack([anchor["writer_input"][1, i].float().numpy() for i in (0, 1)]).astype(np.float64)
        proposed, receipts = {}, {}
        for i, world in enumerate(WORLDS):
            check()
            dirs = unit_directions(materialize(inputs[1-i]-inputs[i], banks, root, world, np), np)
            carrier = attained(before[i], dirs["svd"], chosen["seed_fraction"], cfg, np) if dirs else None
            reached = measure(before[i], dirs, carrier, chosen["gain"], cfg, np)
            report = {"parent": reached, "signed": {}}
            receipts[world] = report
            if reached["status"] != "measured" or not attain.world_pass(
                    [{"methods": m} for m in reached["signs"].values()], "joint", cfg)[1]:
                continue
            _, nominal, _ = carrier
            (a, b), _ = sphere_pair(dirs["t1_fit"], dirs["t2_fit"], chosen["gain"], np)
            for sign in SIGNS:
                check()
                active, geom = centered(before[i], (sign*nominal*b) - (sign*nominal*a), cfg, np)
                if not geom["differential"]["qualified"]:
                    report["signed"][str(sign)] = {"centered": geom, "status": "centered_chord_unqualified"}
                    continue
                sham, detail = select(before[i], *active, f"{SHAM_SEED}/{backend}/{root}/{world}/{sign}",
                                      design_head, ids, weighted, np)
                report["signed"][str(sign)] = {"centered": geom, "sham": detail}
                if sham is not None:
                    proposed[world, sign] = tuple(active) + tuple(sham)
        return proposed, receipts

    return plan, ids


def effect(arms, first):
    """Use the half-chord convention in all three margin contrasts."""
    def difference(field):
        t1, t2, s1, s2 = [arms[k][field]["answer_margin_change"] for k in ARMS]
        return ((t2-t1) - (s2-s1)) / 2

    current = first["paired_label_contrast"]
    future, rescue = difference("edited"), difference("k_rescue")
    return {"first_pass": current, "second_pass": future, "second_pass_with_clean_k": rescue,
            "delayed": future-current, "delayed_with_clean_k": rescue-current,
            "k_rescue_removed": future-rescue}


def run_root(capture, root, prompts, labels, plan, design, commit_plan, check, np):
    """Commit all preflight decisions before allowing any edited second pass."""
    import torch

    if (len(prompts) != 2 or prompts[0] == prompts[1] or len(labels) != 2
            or len(set(labels)) != 2 or any(type(t) is not int for t in labels)):
        raise ValueError("A root needs two distinct counterfactual prompts and answer token IDs")
    check()
    anchor = capture.capture(prompts)
    vocab = anchor["logits"].shape[-1]
    if any(not 0 <= i < vocab for i in labels):
        raise ValueError("The paired labels must bind the native vocabulary")
    pairs, geometry = plan(anchor, root, check)
    expected = {(world, sign) for world in WORLDS for sign in SIGNS}
    if pairs.keys() - expected:
        raise ValueError("The planner returned an unexpected root world or sign")
    receipt = {"root": root, "geometry": geometry, "signed": {}}
    if pairs.keys() != expected:
        commit_plan(receipt | {"status": "geometry_unqualified"})
        return receipt | {"status": "geometry_unqualified", "edited_pass2_runs": 0}

    pending, duplicates = {}, {}
    for i, world in enumerate(WORLDS):
        check()
        clean = capture.capture([prompts[i], prompts[i]])
        if not matched_duplicate(anchor, clean, i):
            commit_plan(receipt | {"status": "mixed_duplicate_unqualified"})
            return receipt | {"status": "mixed_duplicate_unqualified", "edited_pass2_runs": 0}
        duplicates[world] = clean
        label_pair = (labels[i], labels[1-i])
        receipt["signed"][world] = {}
        for sign in SIGNS:
            check()
            endpoints = np.asarray(pairs[world, sign], dtype=np.float64)
            rows = torch.as_tensor(endpoints, dtype=torch.bfloat16)
            if (len(rows) != len(ARMS) or not np.array_equal(rows.float().numpy(), endpoints)
                    or any(torch.equal(row, clean["writer_raw"][WRITER]) for row in rows)
                    or not np.array_equal(endpoints[0]+endpoints[1], endpoints[2]+endpoints[3])):
                raise ValueError("The plan lacks four attained BF16 endpoints at the common midpoint")
            heads = [capture.predict(clean, row) for row in rows]
            stats = grade_first_head(clean["logits"][0], heads, label_pair, np, design=design)
            receipt["signed"][world][str(sign)] = stats | {
                "endpoints_sha256": sha256(endpoints.astype("<f4").tobytes()).hexdigest(),
            }
            pending[world, sign] = rows, heads

    accepted = all(v["passed"] for world in receipt["signed"].values() for v in world.values())
    state = "qualified" if accepted else "head_unqualified"
    commit_plan(receipt | {"status": state})
    if not accepted:
        return receipt | {"status": state, "edited_pass2_runs": 0}
    worlds = {}
    for i, world in enumerate(WORLDS):
        clean = duplicates[world]
        base = clean["writer_raw"][WRITER]
        prompt = prompts[i]
        label_pair = (labels[i], labels[1-i])
        check()
        identity = capture.predict(clean, base)
        check_writer(clean, capture.capture([prompt, prompt], edit=(base, base), planned=identity), identity=True)
        worlds[world] = {}
        for sign in SIGNS:
            rows, heads = pending[world, sign]
            measurements = {key: execute_arm(capture, prompt, clean, row, head, label_pair, check)
                            for key, row, head in zip(ARMS, rows, heads, strict=True)}
            worlds[world][str(sign)] = {"arms": measurements,
                                         "contrast": effect(measurements, receipt["signed"][world][str(sign)])}
        check()
        check_writer(clean, capture.capture([prompt, prompt]), identity=True)
    root_effect = {
        key: sum(world["1"]["contrast"][key] - world["-1"]["contrast"][key] for world in worlds.values()) / 4
        for key in ("first_pass", "second_pass", "delayed", "delayed_with_clean_k", "k_rescue_removed")
    }
    return receipt | {"status": "closed", "worlds": worlds, "root_signed_contrast": root_effect,
                      "edited_pass2_runs": len(WORLDS)*len(SIGNS)*len(ARMS)*2}
