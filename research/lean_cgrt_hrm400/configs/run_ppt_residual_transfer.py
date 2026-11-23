"""Score the pre-registered residual transfer on the existing frozen checkpoints."""

import argparse
from collections import Counter, defaultdict
from math import fsum
from pathlib import Path
import random

import torch

from configs.freeze_ppt_residual_transfer import COUNTS, FIT_SHA, NATIVE, REGISTRY, SEED, STOP_SHA, graph_hash
from configs.freeze_ppt_scope_recovery import prefix_text
from configs.run_ppt_mathcan_progress import read_receipt
from configs.run_ppt_premise_budget import MixedSourcePrior
from configs.run_ppt_residual_search import FLOOR, SCHEMA as FIT_SCHEMA
from configs.run_ppt_residual_search import audit, loaded_scores, models, pack, summarize, trajectories
from configs.run_ppt_rollback_diagnostics import write_receipt
from lcgrt.config import LEAN_EXE, REPL_EXE
from ppt_gym.data import digest, sha256_file
from ppt_gym.ram_campaign import configure
from ppt_gym.ram_learning import tensor_hash
from ppt_gym.ram_limits import TreeBudget


SCHEMA = "ppt.residual-transfer-results.v1"
PRIMARY = "fresh_bpd_directed"
CANDIDATE = "two_slot_reward_source_stop"
CONTROLS = ("trm_reward_source_stop", "two_slot_imitation_source_stop",
            "raw_calibration_source_stop", "frozen_slot_calibration_source_stop")


def inputs(args):
    pins = {name: sha256_file(getattr(args, name))
            for name in ("bank", "fit", "stopping", "registry", "native", "warm", "prior")}
    registry, native = (read_receipt(args.registry), read_receipt(args.native))
    prelude = prefix_text(args.mathcan)
    if (pins["fit"] != FIT_SHA or pins["stopping"] != STOP_SHA or registry["schema"] != REGISTRY
            or registry["counts"] != COUNTS or native["schema"] != NATIVE or native["registry"] != pins["registry"]
            or native["training_bank"] != pins["bank"] or registry["training_bank"] != pins["bank"]
            or registry["fit"] != pins["fit"] or registry["stopping"] != pins["stopping"]
            or native["prelude"] != digest(prelude) or native["lean"] != sha256_file(LEAN_EXE)
            or native["repl"] != sha256_file(REPL_EXE)):
        raise ValueError("the sealed prospective cohort or fit changed")
    expected = {item["payload"]["root"]: item for item in registry["rows"]}
    if (len(expected) != sum(COUNTS.values()) or len(native["rows"]) != len(expected)
            or set(expected) != {row["module"] for row in native["rows"]}
            or Counter(row["source"] for row in native["rows"]) != COUNTS
            or native["independent"]["certificates"] != len(expected)):
        raise ValueError("the native bank lost a registered proof root")
    cases = []
    for row in native["rows"]:
        sealed = expected[row["module"]]
        payload = sealed["payload"]
        if (row["statement"] != f"∀ {payload['binders']}, {payload['target']}"
                or graph_hash(row["statement"]) != sealed["signature"] or row["reference"] != payload["proof"]
                or row["source"] != payload["source"] or not row["tasks"]):
            raise ValueError("a prospectively registered proof or structure changed")
        cases.extend((row, task) for task in row["tasks"])
    parent = read_receipt(args.bank)
    prior_states = {task[role]["key"] for row in parent["rows"] for task in row["tasks"] for role in ("target", "sibling")}
    if any(task[role]["key"] in prior_states for _, task in cases for role in ("target", "sibling")):
        raise ValueError("a prospective public state duplicates an earlier residual")
    return cases, prelude, pins


def checkpoints(args, pins):
    checkpoint = torch.load(args.fit, map_location="cpu", weights_only=True)
    if (checkpoint["schema"] != FIT_SCHEMA
            or checkpoint["pins"] != {name: pins[name] for name in ("bank", "warm", "prior")}
            or checkpoint["calibration_sha256"] != tensor_hash(checkpoint["calibration"])):
        raise ValueError("frozen model or readout pins changed")
    readers = models()
    for name, model in readers.items():
        weights = checkpoint["weights"][name]
        if tensor_hash(weights) != checkpoint["weight_sha256"][name]:
            raise ValueError("a prospective policy weight changed")
        model.load_state_dict(weights)
        model.eval()
    stopping = read_receipt(args.stopping)
    if stopping["fit_sha256"] != pins["fit"] or stopping["pins"] != checkpoint["pins"]:
        raise ValueError("the stopping margins do not come from the archived source control")
    return checkpoint, readers, {name: float(value["threshold"]) for name, value in stopping["source_stop"].items()}


def bootstrap(values):
    sampler = random.Random(SEED)
    draws = sorted(fsum(sampler.choices(values, k=len(values))) / len(values) for _ in range(10000))
    return [draws[249], draws[9749]]


def compare(rows, baseline):
    groups = defaultdict(list)
    for candidate, control in zip(rows, baseline, strict=True):
        keys = ("module", "case", "budget")
        if tuple(candidate[k] for k in keys) != tuple(control[k] for k in keys):
            raise ValueError("the prospective comparison lost paired scenarios")
        groups[candidate["module"]].append((candidate["utility"] - control["utility"],
                                            int(candidate["proof"]) - int(control["proof"])))
    utility = {key: fsum(pair[0] for pair in values) / len(values) for key, values in groups.items()}
    proof = {key: fsum(pair[1] for pair in values) / len(values) for key, values in groups.items()}
    effects = list(utility.values())
    return {"mean": fsum(effects) / len(effects), "root_effects": utility, "descriptive_95": bootstrap(effects),
            "wins": sum(v > 1e-8 for v in effects), "losses": sum(v < -1e-8 for v in effects),
            "proof_rate_difference": fsum(proof.values()) / len(proof)}


@torch.no_grad()
def run(args, tree):
    cases, prelude, pins = inputs(args)
    checkpoint, readers, thresholds = checkpoints(args, pins)
    packed, index = pack(cases, MixedSourcePrior(args))
    mask = packed[3]
    readouts = loaded_scores(packed, readers, checkpoint)
    records = {}
    for name, readout in readouts.items():
        records[name] = trajectories(cases, index, mask, readout, 0.)
        if name in thresholds:
            records[name + "_source_stop"] = trajectories(cases, index, mask, readout, thresholds[name])
    panels, contrasts = {}, {}
    for panel in ("all", *COUNTS):
        rows = {name: [row for row in results if panel == "all" or row["source"] == panel]
                for name, results in records.items()}
        panels[panel] = {name: summarize(result) for name, result in rows.items()}
        contrasts[panel] = {name: compare(rows[CANDIDATE], rows[name]) for name in CONTROLS}
        print(panel, {name: panels[panel][name] for name in (CANDIDATE, *CONTROLS, "native_exact")}, flush=True)
    replays, independent = audit(cases, records, prelude)
    write_receipt(args.results, {"schema": SCHEMA, "pins": pins, "floor": FLOOR, "primary": PRIMARY,
                                 "stopping": thresholds, "summary": panels, "contrasts": contrasts,
                                 "results": records, "replays": replays, "independent": independent,
                                 "memory": tree.stats()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bank", "mathcan", "warm", "prior", "fit", "stopping", "registry", "native", "results"):
        parser.add_argument(name, type=Path)
    tree = TreeBudget(2, 1)
    try:
        configure(1, SEED)
        run(parser.parse_args(), tree)
    finally:
        tree.close()


if __name__ == "__main__":
    main()
