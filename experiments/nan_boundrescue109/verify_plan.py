"""Compare the native planner adapter with sealed, archived NF4 and BF16 endpoints."""

import argparse
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import sys

from experiments.nan_bankdose99 import run as dose
from experiments.nan_phase35.run import lease
from experiments.nan_readpath37.run import FixedHead, SHARD, bound, digest, joined, write_new

from .core import REPO, SIGNS, WORLDS, certified_planner
from .verify import HERE, MANIFEST as NATIVE_MANIFEST, registration as native_registration


MANIFEST = HERE / "PLAN_MANIFEST.json"
BUDGET = dose.BUDGET | {"maximum_seconds": 180}
AGGREGATE = REPO / "reports/nanbeige_bpd_gym/nan_centersham108_results.json"


def registration():
    if json.loads(NATIVE_MANIFEST.read_bytes()) != native_registration():
        raise ValueError("The native planner source differs")
    return {"stage": "archive_planner_identity", "budget": BUDGET,
            "native_manifest_sha256": digest(NATIVE_MANIFEST), "archive_aggregate_sha256": digest(AGGREGATE),
            "verifier_sha256": digest(Path(__file__))}


def audit(snapshot, selected, base, check, np):
    cases = {(r["precision"], r["root"], r["world"], r["sign"]): r["geometry"] for r in selected["cases"]}
    roots = {
        b: {r for precision, r, _, _ in cases if precision == b}
        for b in ("nf4", "bf16")
    }
    common = sorted(roots["nf4"] & roots["bf16"])
    root = common[0]
    public, _ = joined(base)
    roster = {i: row for i, row in public.items() if row["arm"] == "focused" and row["root"] == root}
    assert len(roster) == 2 and {r["split"] for r in roster.values()} == {"development"}

    class ArchiveVector:
        def __init__(self, values):
            self.values = values

        def float(self):
            return self

        def numpy(self):
            return self.values.astype(np.float32)

    counts = {}
    with (snapshot / SHARD).open("rb") as weights:
        head = FixedHead(weights, snapshot, base, np, check)
        for precision in roots:
            actual = dose.captured(precision, base, roster, check, np)["development", root]
            anchor = {
                name: {(1, i): ArchiveVector(actual[w][k]) for i, w in enumerate(WORLDS)}
                for k, name in enumerate(("writer_input", "writer_raw"))
            }
            planner, ids = certified_planner(precision, head, np)
            assert ids.tolist() == selected["design_token_ids"]
            plans, receipt = planner(anchor, root, check)
            assert plans.keys() == {(w, s) for w in WORLDS for s in SIGNS}
            counts[precision] = 0
            for (world, sign), (_, _, a, b) in plans.items():
                reached = receipt[world]["signed"][str(sign)]["sham"]
                prior = cases[precision, root, world, sign]
                assert reached == prior
                assert sha256(np.asarray([a, b], dtype="<f4").tobytes()).hexdigest() == prior["selected_pair_sha256"]
                counts[precision] += 1
    assert "torch" not in sys.modules
    return {"archived_common_roots": 1, "exact_candidate_pairs": counts,
            "design_ids_exact": True, "formal_comparisons": 0, "pretrained_decoder_forwards": 0,
            "sealed_test_accessed": False}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--freeze", action="store_true")
    for name in ("snapshot", "selection", "work"):
        cli.add_argument("--"+name, type=Path)
    args = cli.parse_args()
    registered = registration()
    if args.freeze:
        write_new(MANIFEST, registered)
        return
    if not all((args.snapshot, args.selection, args.work)) or registered != json.loads(MANIFEST.read_bytes()):
        cli.error("Require the frozen pinned-head replay and a fresh work directory")
    base = json.loads(dose.BASE.read_bytes())
    if args.snapshot.name != base["revision"]:
        raise ValueError("The fixed head snapshot changed")
    proof = json.loads(AGGREGATE.read_bytes())
    lock = json.loads(bound(args.selection, proof["selection_lock_sha256"]))
    with lease(BUDGET) as (check, samples):
        assert version("numpy") == "2.5.5" and "torch" not in sys.modules
        import numpy as np

        result = audit(args.snapshot, lock, base, check, np)
    args.work.mkdir(parents=True, exist_ok=False)
    write_new(args.work / "verification.json", result | {
        "status": "passed", "manifest_sha256": digest(MANIFEST), "selection_sha256": digest(args.selection),
        "max_sampled_private_bytes": max(s["private_bytes"] for s in samples),
        "max_sampled_working_set_bytes": max(s["working_set_bytes"] for s in samples),
        "min_sampled_available_RAM_bytes": min(s["available_physical_bytes"] for s in samples),
    })


if __name__ == "__main__":
    main()
