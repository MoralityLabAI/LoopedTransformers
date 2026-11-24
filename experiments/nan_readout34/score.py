"""Post-model tokenizer-only audit and aggregate release for the two depth arms."""
import argparse
import json
from pathlib import Path
import sys
import threading
import time

from experiments.nan_gym05.common import prompt_ids
from experiments.nan_ram28.resources import memory

from . import oracle, source
from .runtime import ARTIFACT, MANIFEST, TARGETS, exclusive_lane, predecessors_closed, verify_manifest
from .runtime_audit import collect, summarize
from .runtime_plan import context_bindings, schedule, write_new
from .tokenizer_check import PRIVATE, RESERVE, START, id_hash, settings, sha, tokenizer


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--tokenizer-receipt", type=Path, required=True)
    cli.add_argument("--snapshot", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    settings()
    verify_manifest()
    with exclusive_lane():
        predecessors_closed()
        if memory()["available_physical_bytes"] < START:
            raise RuntimeError("READOUT34 scoring requires 5 GiB available before starting")
        deadline = time.monotonic() + 600
        stopped = threading.Event()

        def check():
            s = memory()
            if s["available_physical_bytes"] < RESERVE or s["private_bytes"] > PRIVATE or time.monotonic() > deadline:
                raise RuntimeError(f"READOUT34 host resource limit: {s}")

        def watch():
            import os

            while not stopped.wait(1):
                try:
                    check()
                except Exception as error:
                    print(error, file=sys.stderr, flush=True)
                    os._exit(3)

        watchdog = threading.Thread(target=watch, daemon=True)
        watchdog.start()
        try:
            receipt = context_bindings(args.tokenizer_receipt)
            snapshot = args.snapshot.resolve()
            if snapshot.name != source.CONFIG["checkpoint_revision"]:
                raise RuntimeError("Incorrect pinned tokenizer parent revision")
            tok, _ = tokenizer(snapshot)
            rows = source.build()
            for row in rows:
                ids = prompt_ids(tok, snapshot, row["prompt"])
                binding = receipt["contexts"][row["id"]]
                if len(ids) != binding["tokens"] or id_hash(ids) != binding["ids_sha256"]:
                    raise RuntimeError("Host input token replay disagrees with its frozen binding")
                if row["arm"] == "compute" and row["schema"]["readout"] == "final":
                    if oracle.replay_text(row["prompt"]) != row["label"]:
                        raise RuntimeError("Independent source-oracle replay differs")
                check()
            slots = schedule(rows)
            data = {}
            for backend, target in TARGETS.items():
                data[backend] = collect(ARTIFACT / backend, slots, bindings=receipt["contexts"],
                                        tok=tok, manifest_sha=sha(MANIFEST), expected_target=target,
                                        backend=backend)
                check()
            output = summarize(slots, data)
            check()
            if "torch" in sys.modules:
                raise RuntimeError("Host scorer imported Torch")
            write_new(args.output, {
                "experiment": "NAN-READOUT-34",
                "source_roots": {"CAL": 3, "DEV": 9},
                "runtime_manifest_sha256": sha(MANIFEST),
                "registered_slots_per_backend": len(slots),
                "verified_slots": {backend: len(rows) for backend, rows in data.items()},
                "formal_comparisons": 0,
                "verdict": "REDESIGN_OR_PILOT_ONLY",
                **output,
            })
        finally:
            stopped.set()
            watchdog.join()


if __name__ == "__main__":
    main()
