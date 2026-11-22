"""Verify all fixed READOUT34 prompts and answer budgets, with model imports disabled."""
import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
import threading
import time

from experiments.nan_ram28.resources import memory
from experiments.nan_ranksite13.resources import below_normal


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SOURCE = HERE / "SOURCE_MANIFEST.json"
TOKENIZER_PIN = REPO / "experiments/nan_full26/tokenizer_receipt.json"
PACKAGES = {"transformers": "4.46.3", "tokenizers": "0.20.3"}
START = 5 * 2**30
RESERVE = 4 * 2**30
PRIVATE = 768 * 2**20
DEADLINE = 300
MAX_POSITION = 4608


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def id_hash(ids):
    return hashlib.sha256(json.dumps(ids).encode("ascii")).hexdigest()


def settings():
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "RAYON_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ.update(USE_TORCH="0", USE_TF="0", USE_FLAX="0", HF_HUB_OFFLINE="1",
                      TOKENIZERS_PARALLELISM="false", CUDA_VISIBLE_DEVICES="")
    below_normal()


def verify_source():
    manifest = json.loads(SOURCE.read_bytes())
    assert manifest["status"] == "source_only"
    for key, expected in manifest["critical_files_sha256"].items():
        assert sha(HERE / key) == expected, key
    for key, expected in manifest["support_files_sha256"].items():
        assert sha(REPO / key) == expected, key
    return manifest


def tokenizer(snapshot):
    pins = json.loads(TOKENIZER_PIN.read_bytes())["tokenizer_files_sha256"]
    assert set(pins) == {"tokenizer.json", "tokenizer_config.json"}
    for name, expected in pins.items():
        assert sha(snapshot / name) == expected, name
    assert {name: version(name) for name in PACKAGES} == PACKAGES
    from types import SimpleNamespace
    from transformers import PreTrainedTokenizerFast

    return SimpleNamespace(tok=PreTrainedTokenizerFast(tokenizer_file=str(snapshot / "tokenizer.json"))), pins


def audit(snapshot, check):
    from experiments.nan_gym05.common import prompt_ids
    from . import oracle, parser, source, worlds

    manifest = verify_source()
    assert snapshot.name == manifest["checkpoint_revision"]
    rows = source.build()
    assert source.digest([source.public(row) for row in rows]) == manifest["audit"]["public_roster_sha256"]
    assert source.digest([{"id": row["id"], "label": row["label"]} for row in rows]) == manifest["audit"]["label_roster_sha256"]
    tok, pins = tokenizer(snapshot)
    cap = source.CONFIG["proposed_generated_token_cap"]
    contexts = {}
    maxima = {}
    erased = {}
    for row in rows:
        check()
        ids = prompt_ids(tok, snapshot, row["prompt"])
        assert 0 < len(ids) <= MAX_POSITION - cap
        assert type(tok.tok.eos_token_id) is int
        text = worlds.canonical(row["label"], row["schema"]["readout"])
        reply_ids = tok.tok.encode(text, add_special_tokens=False)
        decoded = tok.tok.decode(reply_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        assert 0 < len(reply_ids) <= cap and tok.tok.eos_token_id not in reply_ids
        assert decoded == text and parser.parse(decoded, row["schema"]) is not None
        assert oracle.grade(decoded, row["schema"], row["label"])["joint_exact"] == 1
        key = row["schema"]["readout"]
        maxima[key] = max(maxima.get(key, 0), len(reply_ids))
        binding = {"tokens": len(ids), "ids_sha256": id_hash(ids)}
        contexts[row["id"]] = binding
        if row["arm"] == "erased":
            pair = (row["root"], row["event_count"])
            assert erased.setdefault(pair, binding) == binding
    assert len(contexts) == 432 and len(erased) == 24
    assert not any(name == "torch" or name.startswith("torch.") for name in sys.modules)
    assert {name: sha(snapshot / name) for name in pins} == pins
    return {
        "source_manifest_sha256": sha(SOURCE),
        "tokenizer_files_sha256": pins,
        "tokenizer_check_sha256": sha(__file__),
        "prompt_ids_source_sha256": sha(REPO / "experiments/nan_gym05/common.py"),
        "packages": PACKAGES,
        "contexts": contexts,
        "max_prompt_tokens": max(item["tokens"] for item in contexts.values()),
        "max_answer_tokens_by_readout": maxima,
        "generated_token_cap": cap,
        "eos_token_id": tok.tok.eos_token_id,
        "model_calls": 0,
    }


def main():
    if not __debug__:
        raise RuntimeError("Tokenizer verification requires assertions")
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--snapshot", required=True, type=Path)
    cli.add_argument("--attempt", required=True, type=Path)
    args = cli.parse_args()
    settings()
    args.attempt.parent.mkdir(parents=True, exist_ok=True)
    args.attempt.mkdir(exist_ok=False)
    announced = False
    while memory()["available_physical_bytes"] < START:
        if not announced:
            print("Waiting for 5 GiB of available RAM", flush=True)
            announced = True
        time.sleep(2)
    started = time.monotonic()
    stopped = threading.Event()
    samples = []

    def check():
        sample = memory()
        samples.append(sample)
        if (sample["available_physical_bytes"] < RESERVE or sample["private_bytes"] > PRIVATE
                or time.monotonic() - started > DEADLINE):
            raise RuntimeError(f"Tokenizer check exceeded its resource limits: {sample}")

    def watch():
        while not stopped.wait(1):
            try:
                check()
            except Exception as error:
                print(error, file=sys.stderr, flush=True)
                os._exit(3)

    print("RAM admitted; checking pinned tokenizer without a model", flush=True)
    watchdog = threading.Thread(target=watch, daemon=True)
    watchdog.start()
    try:
        check()
        receipt = audit(args.snapshot.resolve(), check)
        check()
    finally:
        stopped.set()
        watchdog.join()
    receipt["resources"] = {
        "start_RAM_bytes": START, "reserve_RAM_bytes": RESERVE, "private_cap_bytes": PRIVATE,
        "max_sampled_private_bytes": max(s["private_bytes"] for s in samples),
        "min_sampled_available_RAM_bytes": min(s["available_physical_bytes"] for s in samples),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "sample_count": len(samples),
    }
    receipt["completed_UTC"] = datetime.now(timezone.utc).isoformat()
    private = args.attempt / "tokenizer_receipt.json"
    with private.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(receipt, stream, sort_keys=True, indent=2)
        stream.write("\n")
    print(json.dumps({
        **{key: value for key, value in receipt.items() if key != "contexts"},
        "context_count": len(receipt["contexts"]),
        "context_bindings_sha256": hashlib.sha256(json.dumps(
            receipt["contexts"], sort_keys=True, separators=(",", ":")).encode("ascii")).hexdigest(),
        "private_receipt_sha256": sha(private),
    }), flush=True)


if __name__ == "__main__":
    main()
