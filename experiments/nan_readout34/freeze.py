"""Create or verify the source-only registration. This module cannot run inference."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
MANIFEST = ROOT / "SOURCE_MANIFEST.json"
FILES = (
    ".gitattributes", "PROTOCOL.md", "__init__.py", "config.json",
    "freeze.py", "oracle.py", "parser.py", "source.py", "worlds.py",
)
SUPPORT = (
    "experiments/nan_ram28/resources.py",
    "experiments/nan_ranksite13/resources.py",
)


def hash_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run():
    config = json.loads((ROOT / "config.json").read_bytes())
    budget = config["source_engineering_only"]
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(budget["CPU_threads"])

    from experiments.nan_ram28.resources import memory
    from experiments.nan_ranksite13.resources import below_normal

    below_normal()
    start = time.monotonic()
    samples = [memory()]
    if samples[0]["available_physical_bytes"] < budget["minimum_start_available_RAM_bytes"]:
        raise RuntimeError("Source engineering admission declined")

    def check_budget():
        sample = memory()
        samples.append(sample)
        if (time.monotonic() - start > budget["maximum_seconds"]
                or sample["private_bytes"] > budget["maximum_private_bytes"]
                or sample["available_physical_bytes"] < budget["minimum_available_RAM_bytes"]):
            raise RuntimeError("Source engineering budget exceeded")

    def watch():
        while not stop.wait(1):
            try:
                check_budget()
            except Exception:
                print("Source engineering watchdog stopped its own process", file=sys.stderr, flush=True)
                os._exit(3)

    stop = threading.Event()
    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    try:
        check_budget()
        from .source import audit, build

        rows = build()
        summary = audit(rows, check_budget)
        check_budget()
        forbidden = {"torch", "numpy", "tokenizers", "transformers"}
        assert not forbidden.intersection(sys.modules)
        assert not config["runtime_frozen"] and not config["inference_allowed"]
        assert not config["sealed_test_accessed_or_used_as_exclusion"]
        manifest = {
            "experiment": config["experiment"],
            "status": "source_only",
            "checkpoint_revision": config["checkpoint_revision"],
            "runtime_frozen": False,
            "inference_allowed": False,
            "sealed_test_accessed_or_used_as_exclusion": False,
            "critical_files_sha256": {name: hash_file(ROOT / name) for name in FILES},
            "support_files_sha256": {name: hash_file(REPO / name) for name in SUPPORT},
            "audit": summary,
        }
    finally:
        stop.set()
        watcher.join()

    stats = {
        "maximum_sampled_private_bytes": max(sample["private_bytes"] for sample in samples),
        "minimum_sampled_available_physical_bytes": min(sample["available_physical_bytes"] for sample in samples),
        "samples": len(samples),
        "elapsed_seconds": round(time.monotonic() - start, 3),
    }
    return manifest, stats


def main():
    if not __debug__:
        raise RuntimeError("Run the source audit with Python assertions enabled")
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--write", action="store_true", help="Create the source manifest once")
    args = cli.parse_args()
    manifest, stats = run()
    data = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("ascii")
    if args.write:
        with MANIFEST.open("xb") as handle:
            handle.write(data)
    elif data != MANIFEST.read_bytes():
        raise RuntimeError("Source manifest or audit drifted")
    print(json.dumps({"manifest_sha256": hashlib.sha256(data).hexdigest(), **manifest["audit"], **stats}))


if __name__ == "__main__":
    main()
