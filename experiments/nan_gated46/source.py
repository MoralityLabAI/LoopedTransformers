"""Generate an independent GATED46 roster and bind the pinned tokenizer."""
import argparse
import json
from pathlib import Path

from experiments.nan_delay39.source import digest, independently_verify, source_cases
from experiments.nan_gym05.common import prompt_ids
from experiments.nan_phase35.run import lease
from experiments.nan_readout34.runtime_plan import write_new
from experiments.nan_readout34.tokenizer_check import id_hash, sha, tokenizer, verify_source


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CONFIG = HERE / "registration.json"
PIN = HERE / "SOURCE_CHECK.json"
PRIOR = REPO / "runs/nan_delay39/source"


def inputs(tok, public, snapshot, check):
    bindings = {}
    ids = {c: tok.tok.encode(c, add_special_tokens=False) for c in "ABCDEFGHIJKLMNOP"}
    if any(len(x) != 1 for x in ids.values()) or len({v[0] for v in ids.values()}) != len(ids):
        raise ValueError("Bare labels do not uniquely identify vocabulary rows")
    for row in public:
        check()
        tokens = prompt_ids(tok, snapshot, row["prompt"])
        bindings[row["id"]] = {"tokens": len(tokens), "sha256": id_hash(tokens), "ids": tokens}
    return {"row_bindings": bindings, "bare_label_token_ids": {c: v[0] for c, v in ids.items()}}


def prepare(snapshot, out):
    cfg = json.loads(CONFIG.read_bytes())
    if snapshot.name != cfg["checkpoint_revision"]:
        raise ValueError("Wrong tokenizer snapshot revision")
    old = REPO / "experiments/nan_comp14/fixtures/public.jsonl"
    previous = json.loads((PRIOR / "public.json").read_bytes())
    pre_pins = json.loads((REPO / "experiments/nan_delay39/SOURCE_CHECK.json").read_bytes())
    if digest(previous) != pre_pins["source_digest"]:
        raise ValueError("Previously generated source roster changed")
    exposed = [json.loads(line) for line in old.read_bytes().splitlines()] + previous
    public, gold = source_cases(cfg)
    for item in public + gold:
        for key in ("id", "root"):
            item[key] = item[key].replace("delay39:", "gated46:", 1)
    count = independently_verify(public, gold, cfg["source_roots"],
                                 {r["prompt"] for r in exposed if r["arm"] == "focused"})
    limits = {"minimum_start_available_RAM_bytes": 5 * 2**30, "minimum_available_RAM_bytes": 4 * 2**30,
              "maximum_private_bytes": 768 * 2**20, "maximum_seconds": 120}
    with lease(limits) as (check, samples):
        verify_source()
        tok, pinned = tokenizer(snapshot)
        encoded = inputs(tok, public, snapshot, check)
        bindings = encoded["row_bindings"]
        groups = {(r["root"], r["arm"], r["world"]): bindings[r["id"]] for r in public}
        for root in {r["root"] for r in public}:
            for arm in cfg["arms"]:
                a, b = groups[root, arm, "A"], groups[root, arm, "B"]
                if a["tokens"] != b["tokens"] or (arm == "focused_erased" and a != b):
                    raise ValueError("The raw roster did not preserve the paired token layout")
    out.mkdir(parents=True, exist_ok=False)
    for name, rows in (("public", public), ("gold", gold), ("tokens", encoded)):
        write_new(out / (name + ".json"), rows)
    write_new(out / "SOURCE_CHECK.json", {
        "public_sha256": digest(public), "gold_sha256": digest(gold), "tokens_sha256": digest(encoded),
        "tokenizer_pins": pinned, "registered_roots": cfg["source_roots"], "prompts": len(public),
        "independent_focus_replay": count, "max_token_length": max(x["tokens"] for x in bindings.values()),
        "registered_config_sha256": sha(CONFIG),
        "max_sampled_private_bytes": max(x["private_bytes"] for x in samples),
        "min_sampled_available_RAM_bytes": min(x["available_physical_bytes"] for x in samples),
    })


def bound_rows(folder, subset):
    pins = json.loads(PIN.read_bytes())
    value = json.loads((folder / (subset + ".json")).read_bytes())
    if digest(value) != pins[subset + "_sha256"] or sha(CONFIG) != pins["registered_config_sha256"]:
        raise ValueError(f"Frozen GATED46 {subset} source changed")
    return value


def load_gpu_source(folder, snapshot, check):
    public, tokens = (bound_rows(folder, key) for key in ("public", "tokens"))
    tok, actual = tokenizer(snapshot)
    if actual != json.loads(PIN.read_bytes())["tokenizer_pins"] or inputs(tok, public, snapshot, check) != tokens:
        raise ValueError("GPU tokenizer does not reproduce the independent frozen input receipt")
    indexed = {(r["root"], r["arm"], r["world"]): (r["id"], tokens["row_bindings"][r["id"]]["ids"]) for r in public}
    if len(indexed) != json.loads(PIN.read_bytes())["prompts"]:
        raise ValueError("GATED46 prompt roster is incomplete")
    return indexed, tokens["bare_label_token_ids"], tok.tok


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--snapshot", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    if not __debug__:
        raise RuntimeError("Use assertions for the independent source oracle")
    prepare(args.snapshot.resolve(), args.output)


if __name__ == "__main__":
    main()
