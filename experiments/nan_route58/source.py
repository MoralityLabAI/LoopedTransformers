"""Generate and verify fresh ROUTE58 sources with the pinned host tokenizer."""
import argparse
import json
from pathlib import Path

from experiments.nan_delay39.source import digest, independently_verify, source_cases
from experiments.nan_gated46.source import bound_rows, inputs
from experiments.nan_phase35.run import lease
from experiments.nan_readout34.runtime_plan import write_new
from experiments.nan_readout34.tokenizer_check import sha, tokenizer, verify_source


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CONFIG = HERE / "registration.json"
SHAM = REPO / "experiments/nan_sham56"
RUNTIME_INPUTS = ("PROTOCOL.md", "registration.json", "source.py", ".gitattributes", "__init__.py")


def seal():
    files = [HERE / p for p in RUNTIME_INPUTS]
    files += [SHAM / p for p in ("geometry.py", "MINIMAX_MANIFEST.json", "NATIVE_MANIFEST.json")]
    files += [REPO / p for p in ("experiments/nan_delay39/source.py", "experiments/nan_delay39/SOURCE_CHECK.json",
        "experiments/nan_gated46/source.py", "experiments/nan_gated46/SOURCE_CHECK.json",
        "experiments/nan_comp14/fixtures/public.jsonl", "experiments/nan_comp14/sources.py")]
    return {p.relative_to(REPO).as_posix(): sha(p) for p in files}


def prepare(cfg, snapshot, path, manifest):
    if snapshot.name != cfg["checkpoint_revision"]:
        raise ValueError("Tokenizer snapshot revision differs")
    comp14 = [json.loads(line) for line in
              (REPO / "experiments/nan_comp14/fixtures/public.jsonl").read_bytes().splitlines()]
    delay = json.loads((REPO / "runs/nan_delay39/source/public.json").read_bytes())
    expected = json.loads((REPO / "experiments/nan_delay39/SOURCE_CHECK.json").read_bytes())["source_digest"]
    if digest(delay) != expected:
        raise ValueError("DELAY39 prior roster changed")
    gated = bound_rows(REPO / "runs/nan_gated46/source", "public")
    public, gold = source_cases(cfg)
    for item in public + gold:
        for key in ("id", "root"):
            item[key] = item[key].replace("delay39:", "route58:", 1)
    earlier = {r["prompt"] for group in (comp14, delay, gated) for r in group if r["arm"] == "focused"}
    checked = independently_verify(public, gold, cfg["source_roots"], earlier)
    limits = {"minimum_start_available_RAM_bytes": 5*2**30, "minimum_available_RAM_bytes": 4*2**30,
              "maximum_private_bytes": 768*2**20, "maximum_seconds": 120}
    with lease(limits) as (check, samples):
        verify_source()
        tok, pins = tokenizer(snapshot)
        tokens = inputs(tok, public, snapshot, check)
        bindings = tokens["row_bindings"]
        groups = {(row["root"], row["arm"], row["world"]): bindings[row["id"]] for row in public}
        for root in {row["root"] for row in public}:
            for arm in cfg["arms"]:
                a, b = (groups[root, arm, world] for world in ("A", "B"))
                if a["tokens"] != b["tokens"] or (arm == "focused_erased" and a != b):
                    raise ValueError("Paired source tokenizer layout changed")
        check()
    path.mkdir(parents=True, exist_ok=False)
    for name, value in (("public", public), ("gold", gold), ("tokens", tokens)):
        write_new(path / f"{name}.json", value)
    goal = {"experiment": cfg["experiment"], "status": "source_only", "roots": cfg["source_roots"],
            "prompts": len(public), "independently_checked_focused_worlds": checked,
            "public_sha256": digest(public), "gold_sha256": digest(gold), "token_bindings_sha256": digest(tokens),
            "source_manifest_sha256": sha(manifest), "max_prompt_length": max(b["tokens"] for b in bindings.values()),
            "tokenizer_pins": pins, "prior_focused_prompt_count": len(earlier), "sealed_test_accessed": False,
            "model_calls": 0, "max_sampled_private_bytes": max(s["private_bytes"] for s in samples),
            "min_sampled_available_RAM_bytes": min(s["available_physical_bytes"] for s in samples)}
    write_new(path / "SOURCE_CHECK.json", goal)
    print(json.dumps(goal), flush=True)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--freeze", action="store_true")
    cli.add_argument("--snapshot", type=Path)
    cli.add_argument("--output", type=Path)
    args = cli.parse_args()
    manifest = HERE / "SOURCE_MANIFEST.json"
    if args.freeze:
        write_new(manifest, seal())
        return
    if not __debug__ or json.loads(manifest.read_bytes()) != seal():
        raise ValueError("Source manifest changed")
    if not args.snapshot or not args.output:
        cli.error("--snapshot and --output required")
    prepare(json.loads(CONFIG.read_bytes()), args.snapshot, args.output, manifest)


if __name__ == "__main__":
    main()
