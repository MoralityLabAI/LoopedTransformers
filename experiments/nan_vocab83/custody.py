"""Read immutable WRITER77 observations with an isolated oracle."""

import argparse
import json
from pathlib import Path
import re
import sys

from experiments.nan_phase35.run import lease
from experiments.nan_readout34 import runtime
from experiments.nan_readout34.runtime_plan import write_new
from experiments.nan_readout34.tokenizer_check import sha, tokenizer
from experiments.nan_readpath37.run import bound
from experiments.nan_vocab82.heads import partition, partition_sha, verify_head
from experiments.nan_vocab82.host import score
from experiments.nan_writer77.host import qualify
from experiments.nan_writer77.source import CONFIG, load


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
EXPERIMENT = "NAN-VOCAB-83"
MANIFEST = HERE / "RUNTIME_MANIFEST.json"
COMMISSIONED = HERE / "RUNTIME_CHECK.json"
OBSERVER = REPO / "experiments/nan_vocab82/VERIFICATION.json"
HOST_BUDGET = {"minimum_start_available_RAM_bytes": 5 * 2**30,
               "minimum_available_RAM_bytes": 4 * 2**30,
               "maximum_private_bytes": 128 * 2**20, "maximum_seconds": 120}


def config():
    return json.loads(CONFIG.read_bytes())


def vocabulary(snapshot, check, expected_pins):
    registered = json.loads(OBSERVER.read_bytes())["full_width"]
    fixed, pins = tokenizer(snapshot)
    if pins != expected_pins:
        raise ValueError("Vocabulary tokenizer differs from the frozen source")
    tok = fixed.tok
    classes = partition(tok.backend_tokenizer, registered["vocab"], check)
    if partition_sha(classes) != registered["partition_sha256"]:
        raise ValueError("Token classes differ from the verified full vocabulary")
    return tok, classes


def load_baseline(path, expected=None):
    if expected is None:
        expected = sha(path)
    return json.loads(bound(path, expected)), expected


def qualification(source, rows, baseline_sha, classes, tok, cfg):
    compact = [{k: v for k, v in row.items() if k != "class_maxima"} for row in rows]
    labels = source["tokens"]["bare_label_token_ids"]
    legacy, _ = qualify(source, compact, labels, tok, len(classes), cfg)
    public = {p["id"]: p for p in source["public"]}
    erased = {}
    for row in rows:
        for extrema, native in zip(row["class_maxima"], row["phases"], strict=True):
            verify_head(extrema, native, classes, labels)
        item = public[row["id"]]
        if item["arm"] == "focused_erased":
            packet = row["token_sha256"], row["phases"], row["class_maxima"]
            if erased.setdefault(item["root"], packet) != packet:
                raise ValueError("Identical erased inputs produced different physical heads")
    return {"experiment": EXPERIMENT, "baseline_sha256": baseline_sha,
            "partition_sha256": partition_sha(classes), "vocab": len(classes), "gate": legacy}


def collect(work, source, classes, tok, frozen, *, mode="pretrained_cuda"):
    terminal_path = work / "terminal.json"
    terminal_sha = sha(terminal_path)
    terminal = json.loads(bound(terminal_path, terminal_sha))
    expected = runtime.TARGETS["bf16"]
    if (terminal["experiment"] != EXPERIMENT or terminal["status"] != "completed"
            or terminal["mode"] != mode or terminal["backend"] != "bf16" or terminal["runtime_sha256"] != frozen
            or terminal["checkpoint_verified_after"] is not True
            or terminal["target_before"] != expected or terminal["target_after"] != expected
            or terminal["quant_before"] != terminal["quant_after"]
            or terminal["partition_sha256"] != partition_sha(classes) or terminal["vocab"] != len(classes)):
        raise ValueError("The terminal does not certify the frozen model and readout")
    admission = json.loads(bound(work / "admission.json", terminal["admission_sha256"]))
    published = terminal["published_commit"]
    if (re.fullmatch(r"[0-9a-f]{40}", published) is None
            or (admission["published_commit"], admission["runtime_sha256"], admission["mode"]) != (published, frozen, mode)):
        raise ValueError("The terminal and admitted publication disagree")
    preflight = json.loads(bound(work / "preflight.json", terminal["preflight_sha256"]))
    maximum = json.loads(MANIFEST.read_bytes())["max_position"]
    if (preflight["both_heads_exact"] is not True or preflight["partition_sha256"] != partition_sha(classes)
            or set(preflight["depths"]) != {"1", "2"}
            or any(value["tested_max_position"] != maximum for value in preflight["depths"].values())):
        raise ValueError("Device preflight does not bind the admitted B2 head")
    rows, baseline_sha = load_baseline(work / "baseline.json", terminal["baseline_sha256"])
    cfg = config()
    gate = qualification(source, rows, baseline_sha, classes, tok, cfg)
    recorded = json.loads(bound(work / "qualification.json", terminal["qualification_sha256"]))
    if gate != recorded:
        raise ValueError("Independent native competence qualification changed")
    journal = bound(work / "closed.jsonl", terminal["journal_sha256"]).splitlines()
    roots = sorted({p["root"] for p in source["public"]})
    if len(journal) != len(roots) or terminal["root_records"] != len(roots):
        raise ValueError("The closed journal omits registered graph roots")
    if len(rows) != 2 * terminal["b2_baseline_batches"]:
        raise ValueError("The baseline is missing a physical B2 row")
    episodes = []
    for ordinal, (root, line) in enumerate(zip(roots, journal, strict=True)):
        entry = json.loads(line)
        path = work / f"root_{ordinal:03d}.json"
        if entry["file"] != path.name:
            raise ValueError("Root record filename is out of sequence")
        record = json.loads(bound(path, entry["sha256"]))
        if record["root"] != root:
            raise ValueError("Root record does not bind the frozen order")
        episodes.append(record)
    aggregate, private = score(source, rows, episodes, classes, tok, cfg)
    return aggregate | {"experiment": EXPERIMENT, "terminal_sha256": terminal_sha,
                        "runtime_sha256": frozen, "mode": mode}, private


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--source-dir", type=Path, required=True)
    cli.add_argument("--snapshot", type=Path, required=True)
    cli.add_argument("--work", type=Path, required=True)
    cli.add_argument("--qualification", action="store_true")
    cli.add_argument("--fixture", action="store_true")
    args = cli.parse_args()
    from .lane import verified

    frozen = verified()
    if "torch" in sys.modules:
        cli.error("Use the independent tokenizer-only host")
    with lease(HOST_BUDGET) as (check, samples):
        source, source_receipt = load(args.source_dir, ("public", "gold", "tokens"))
        if args.snapshot.name != config()["checkpoint_revision"]:
            raise ValueError("Wrong checkpoint revision for the host oracle")
        tok, classes = vocabulary(args.snapshot, check, source_receipt["tokenizer_pins"])
        mode = "synthetic_fixture" if args.fixture else "pretrained_cuda"
        if args.qualification:
            admission = json.loads((args.work / "admission.json").read_bytes())
            if (admission["runtime_sha256"], admission["mode"]) != (frozen, mode):
                raise ValueError("The baseline lacks a matching admitted run")
            rows, baseline_sha = load_baseline(args.work / "baseline.json")
            result = qualification(source, rows, baseline_sha, classes, tok, config())
            write_new(args.work / "qualification.json", result)
        else:
            result, private = collect(args.work, source, classes, tok, frozen, mode=mode)
            write_new(args.work / "private_outcomes.json", private)
            if "torch" in sys.modules:
                raise RuntimeError("The independent host imported Torch")
    if not args.qualification:
        write_new(args.work / "aggregate.json", result | {
            "private_outcomes_sha256": sha(args.work / "private_outcomes.json"),
            "max_sampled_private_bytes": max(s["private_bytes"] for s in samples),
            "min_sampled_available_RAM_bytes": min(s["available_physical_bytes"] for s in samples),
            "max_sampled_working_set_bytes": max(s["working_set_bytes"] for s in samples),
        })


if __name__ == "__main__":
    main()
