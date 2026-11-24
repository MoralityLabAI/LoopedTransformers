"""Compile sender-binding controls from immutable native DELAY39 evidence."""

import argparse
from collections import Counter, defaultdict
from importlib.metadata import version
import json
from math import isclose, isfinite
from pathlib import Path
import re
from statistics import fmean
import sys

from experiments.nan_delay39 import pilot
from experiments.nan_delay39.contrasts import qualified_pair
from experiments.nan_delay39.graph_host import graph_cells
from experiments.nan_delay39.source import digest
from experiments.nan_phase35.run import lease
from experiments.nan_readout34.runtime_plan import write_new
from experiments.nan_readout34.tokenizer_check import sha
from experiments.nan_readpath37.run import bound
from experiments.nan_select47 import run as selectors

from .exact import nominations, summarize


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
EXPERIMENT = "NAN-MEASUREDWIRE-93"
MANIFEST = HERE / "SOURCE_MANIFEST.json"
PROOF = HERE / "CPU_VERIFICATION.json"
BUDGET = {"minimum_start_available_RAM_bytes": 5 * 2**30, "minimum_available_RAM_bytes": 4 * 2**30,
          "maximum_private_bytes": 384 * 2**20, "maximum_seconds": 180}


def registered():
    selectors.verify()
    files = [HERE / p for p in (".gitattributes", "__init__.py", "PROTOCOL.md", "exact.py", "run.py", "verify.py")]
    files += [selectors.MANIFEST, pilot.SEAL, selectors.DELAY / "registration.json",
              selectors.DELAY / "contrasts.py", selectors.DELAY / "sink_edges.py",
              REPO / "experiments/nan_sourcewire92/permutations.py"]
    inherited = selectors.registration()["files_sha256"]
    return {"experiment": EXPERIMENT, "resource_budget": BUDGET,
            "files_sha256": inherited | {p.relative_to(REPO).as_posix(): sha(p) for p in files}}


def source_payload(folder):
    pins = json.loads((selectors.DELAY / "SOURCE_CHECK.json").read_bytes())
    gold = json.loads((folder / "gold.json").read_bytes())
    tokens = json.loads((folder / "tokens.json").read_bytes())["row_bindings"]
    if digest(gold) != pins["gold_digest"] or digest(tokens) != pins["token_digest"]:
        raise ValueError("Frozen DELAY39 gold or token bindings differ")
    return gold, tokens


def audit_dose(case, cfg):
    if type(case["eligible"]) is not bool:
        raise ValueError("Finite signed dose requires a Boolean qualification")
    if not case["eligible"]:
        return
    target, actual, angle = (case[k] for k in ("requested_l2", "realized_l2", "direction_cosine"))
    if (not all(isfinite(v) for v in (target, actual, angle)) or target <= 0 or actual <= 0
            or abs(actual / target - 1) > cfg["dose_relative_tolerance"]
            or not cfg["minimum_direction_cosine"] <= angle <= 1 + 1e-12):
        raise ValueError("Recorded reached edit violates the physical dose or cosine")


def audit_slopes(episode, writes, scales, cfg, cols, np):
    phase2 = episode["phase2_terminal"]
    if (phase2["registered_phase2_dose_fraction"] != cfg["phase2_readout_dose_fraction"]
            or set(phase2["phase2_write_responses"]) != set(map(str, cols))):
        raise ValueError("Phase-two terminal dose or receiver roster changed")
    for j in cols:
        rec = phase2["phase2_write_responses"][str(j)]
        pos, neg = rec["positive"], rec["negative"]
        for p in (pos, neg):
            audit_dose(p, cfg)
        slope = rec["signed_label_slopes_per_read_coefficient"]
        if qualified_pair(pos, neg, cfg):
            if slope is None or len(slope) != 16 or not all(isfinite(v) for v in slope):
                raise ValueError("A qualified direct sink slope is missing")
        elif slope is not None:
            raise ValueError("An unqualified direct sink provided a measured slope")
    closed = 0
    for i in cfg["phase1_components"]:
        pos, neg = (episode["cases"][f"component_{i}_{k}"] for k in ("positive", "negative"))
        for factor, case in ((1., pos), (-1., neg)):
            audit_dose(case, cfg)
            if case["eligible"]:
                delta = np.asarray(case["response"]["realized_residual_delta"], dtype=np.float64)
                if delta.shape != writes[i].shape or not np.isfinite(delta).all():
                    raise ValueError("Malformed realized signed residual displacement")
                actual = float(np.linalg.norm(delta))
                direction = float(delta @ writes[i] / (actual * scales[i])) * factor if actual else 0.
                if (not isclose(actual, case["realized_l2"], rel_tol=1e-12, abs_tol=1e-12)
                        or not isclose(direction, case["direction_cosine"], rel_tol=1e-12, abs_tol=1e-12)):
                    raise ValueError("Signed dose receipt disagrees with the stored physical edit")
        observed = episode["signed_phase2_edges_per_l2"][str(i)]
        if qualified_pair(pos, neg, cfg):
            a, b = (np.asarray(p["response"]["phase2_read"], dtype=np.float64) for p in (pos, neg))
            if (a.shape != (len(cols),) or b.shape != a.shape or not np.isfinite([a, b]).all()
                    or observed is None):
                raise ValueError("A qualified signed delayed read is missing")
            replay = (a - b) / (pos["realized_l2"] + neg["realized_l2"])
            if not np.array_equal(replay, observed):
                raise ValueError("Signed delayed slopes disagree with paired response bytes")
            closed += 1
        elif observed is not None:
            raise ValueError("An unqualified delayed read provided a measured slope")
    return closed


def publication_kind(path, *, fixture):
    terminal = json.loads(path.read_bytes())
    stamp = terminal["published_commit"]
    if fixture:
        if stamp != "synthetic":
            raise ValueError("Fixture has a pretrained publication identity")
    elif re.fullmatch(r"[0-9a-f]{40}", stamp) is None:
        raise ValueError("A synthetic or malformed publication cannot enter the measured audit")


def profiles(work, gold, tokens, writes, cfg, check, np, *, fixture=False):
    path = work / "terminal.json"
    if not path.is_file():
        raise RuntimeError("Awaiting the closed DELAY39 device terminal")
    publication_kind(path, fixture=fixture)
    seal = sha(path)
    scales = np.linalg.norm(writes, axis=1).tolist()
    cols = cfg["receiver_components"]
    if (cols != list(range(len(writes))) or any(not isfinite(v) or v <= 0 for v in scales)
            or np.linalg.matrix_rank(writes) != len(writes)):
        raise ValueError("The pinned physical writer span is invalid")
    custody = selectors.collect(work, gold, tokens, scales, cfg, check)
    if seal != custody["terminal_sha256"] or sha(path) != seal:
        raise ValueError("Terminal changed during native custody evaluation")
    focus = {r["id"]: r for r in gold if r["arm"] == "focused"}
    twin = {(r["root"], r["world"]): r for r in focus.values()}
    raw = {root: {} for root, _ in sorted(twin)}
    complete = []
    slope_count = 0
    if custody["status"] != "native_gate_failed":
        old = json.loads((selectors.DELAY / "registration.json").read_bytes())
        journal = bound(work / "closed.jsonl", custody["native_journal_sha256"])
        seen = set()
        statuses = Counter()
        for index, entry in enumerate(map(json.loads, journal.splitlines())):
            check()
            episode = work / f"episode_{index:04d}.json"
            if episode.name != entry["file"]:
                raise ValueError("Native graph journal order changed")
            item = json.loads(bound(episode, entry["sha256"]))
            row = focus[item["source_id"]]
            if item["source_id"] in seen or item["prompt_sha256"] != tokens[row["id"]]["sha256"]:
                raise ValueError("The native episode binding changed")
            seen.add(item["source_id"])
            root, world = row["root"], row["world"]
            peer = twin[root, "B" if world == "A" else "A"]
            slope_count += audit_slopes(item, writes, scales, old, cols, np)
            graph, reason = selectors.extract(item, row, peer, scales, cfg, old)
            raw[root][world] = {"status": reason, "graph": graph}
            statuses[reason] += 1
        if seen != set(focus) or statuses != custody["per_world_status"]:
            raise ValueError("Native world status differs from the selector seal")
        complete = [[worlds[w]["graph"] for w in ("A", "B")] for worlds in raw.values()
                    if set(worlds) == {"A", "B"} and all(v["status"] == "complete" for v in worlds.values())]
    if len(complete) != custody["complete_CAL_roots"]:
        raise ValueError("Complete root count differs from native selector custody")
    coverage = {"backend": custody["backend"], "native_terminal_sha256": seal, "status": custody["status"],
                "mode": "synthetic_fixture" if fixture else "measured_native", "registered_worlds": len(focus),
                "native_journal_sha256": custody.get("native_journal_sha256"),
                "complete_CAL_roots": len(complete), "qualified_signed_sender_slopes_checked": slope_count,
                "per_world_status": dict(custody["per_world_status"])}
    if custody["status"] != "frozen":
        return coverage, raw
    panel = summarize(complete, cfg, check, np)
    cells = [graph_cells(w["source"], w["middle"], w["sink"])[1] for pair in complete for w in pair]
    for field, values in (("mean_signed_path_sum", [g["signed_path_product_sum"] for g in cells]),
                          ("mean_absolute_net", [abs(g["signed_path_product_sum"]) for g in cells]),
                          ("mean_unsigned_path_mass", [g["absolute_path_product_sum"] for g in cells])):
        if not isclose(fmean(values), panel["primary_signed_paths"][field]["identity"], rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("Identity signed path products differ from the native graph reducer")
    masks = custody["masks"]
    picked = nominations(panel, masks)
    groups = defaultdict(list)
    for name, data in (masks | picked).items():
        groups[tuple(data["source"])].append(name)
    plan = [{"source": list(key), "selectors": sorted(names)} for key, names in sorted(groups.items())]
    return coverage | {"observed_masks": masks, "sender_assignment": panel, "nominated_controls": picked,
                       "physical_source_intervention_groups": plan,
                       "distinct_physical_source_masks": len(plan)}, raw


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--freeze", action="store_true")
    cli.add_argument("--source-dir", type=Path)
    cli.add_argument("--backend", choices=("nf4", "bf16"))
    cli.add_argument("--output", type=Path)
    args = cli.parse_args()
    frozen = registered()
    if args.freeze:
        write_new(MANIFEST, frozen)
        return
    if (not __debug__ or not args.source_dir or not args.backend or not args.output
            or json.loads(MANIFEST.read_bytes()) != frozen):
        cli.error("Require the frozen audit, native backend and a fresh output directory")
    receipt = json.loads(PROOF.read_bytes())
    if (receipt["source_manifest_sha256"] != sha(MANIFEST) or receipt["status"] != "passed"
            or receipt["native_forwards"] != 0 or not receipt["tampered_signed_slopes_refused"]):
        raise ValueError("Measured audit is missing its frozen CPU commissioning receipt")
    work = pilot.ARTIFACT / args.backend
    if not (work / "terminal.json").is_file():
        raise RuntimeError("Awaiting the closed DELAY39 device terminal")
    gold, tokens = source_payload(args.source_dir)
    cfg = json.loads(selectors.CONFIG.read_bytes())
    with lease(BUDGET) as (check, samples):
        if version("numpy") != "2.5.5":
            raise ValueError("Pinned numerical runtime changed")
        import numpy as np

        _, writes = pilot.get_bank(args.backend, np)
        outcome, raw = profiles(work, gold, tokens, writes, cfg, check, np)
    assert frozen == registered() and "torch" not in sys.modules
    args.output.mkdir(parents=True, exist_ok=False)
    evidence = args.output / "private_graphs.json"
    write_new(evidence, raw)
    write_new(args.output / "aggregate.json", outcome | {"experiment": EXPERIMENT,
              "source_manifest_sha256": sha(MANIFEST), "private_graphs_sha256": sha(evidence),
              "formal_comparisons": 0, "native_forwards": 0, "sealed_test_accessed": False,
              "max_sampled_private_bytes": max(v["private_bytes"] for v in samples),
              "max_sampled_working_set_bytes": max(v["working_set_bytes"] for v in samples)})


if __name__ == "__main__":
    main()
