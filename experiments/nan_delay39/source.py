"""Fresh paired pointers, independent last-query oracle, and immutable token IDs."""
import argparse
import hashlib
import json
from pathlib import Path
import re

from experiments.nan_comp14.sources import materialize
from experiments.nan_phase35.run import lease
from experiments.nan_readout34.runtime_plan import write_new
from experiments.nan_readout34.tokenizer_check import id_hash, tokenizer, verify_source
from experiments.nan_gym05.common import prompt_ids


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PROTECTED_OMITTED = True
ARMS = ["focused", "copy", "focused_erased"]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def source_cases(cfg):
    assert cfg["arms"] == ARMS
    public, gold = materialize({"source_seed": cfg["source_seed"], "roots": {"calibration": cfg["source_roots"]},
                                "arms": ARMS})
    for row in public + gold:
        for field in ("id", "root"):
            row[field] = row[field].replace("comp14:", "delay39:", 1)
    return public, gold


def independently_verify(public, gold, roots, prior_focused):
    assert len(public) == len(gold) == roots * 2 * len(ARMS)
    positions = {(p["root"], p["world"], p["arm"]): (p, g) for p, g in zip(public, gold)}
    assert len(positions) == len(public)
    focused = set()
    for root in sorted({p["root"] for p in public}):
        a, b = [positions[root, world, "focused"] for world in ("A", "B")]
        ga, gb = a[1], b[1]
        assert ga["answer"] != gb["answer"] and ga["query"] == gb["query"]
        assert {x for x in ga["edges"] if ga["edges"][x] != gb["edges"][x]} == set(ga["paired_changed_sources"])
        erased = []
        for world, (p, g) in zip(("A", "B"), (a, b)):
            graph, start, hops, suffix = p["prompt"].splitlines()[-4:]
            assert start == "Start: " + g["query"] and hops == "Hops: 1" and suffix == "Answer:"
            read = dict(re.findall(r"\b([A-P])->([A-P])\b", graph))
            assert len(read) == 16 and list(read.values()).count(g["answer"]) == 1 and read[g["query"]] == g["answer"]
            assert p["prompt"] not in prior_focused and p["prompt"] not in focused
            focused.add(p["prompt"])
            copy, copy_gold = positions[root, world, "copy"]
            assert copy["prompt"].splitlines()[-1] == copy_gold["answer"] == g["answer"]
            missing, missing_gold = positions[root, world, "focused_erased"]
            assert missing_gold["answer"] == g["answer"] and missing["prompt"].splitlines()[-4] == "Edges: [not provided]"
            erased.append(missing["prompt"])
        assert erased[0] == erased[1]
    for p, g in zip(public, gold):
        assert all(p[key] == g[key] for key in ("id", "root", "split", "world", "arm"))
    return len(focused)


def main():
    if not __debug__:
        raise RuntimeError("Use assertions for registered source verification")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads((HERE / "registration.json").read_bytes())
    if args.snapshot.name != cfg["checkpoint_revision"]:
        raise ValueError("Wrong frozen tokenizer snapshot")
    public, gold = source_cases(cfg)
    prior = REPO / "experiments/nan_comp14/fixtures/public.jsonl"
    prior_focused = {p["prompt"] for p in [json.loads(line) for line in prior.read_bytes().splitlines()] if p["arm"] == "focused"}
    n_focus = independently_verify(public, gold, cfg["source_roots"], prior_focused)
    limits = {"minimum_start_available_RAM_bytes": 5*2**30, "minimum_available_RAM_bytes": 4*2**30,
              "maximum_private_bytes": 768*2**20, "maximum_seconds": 120}
    with lease(limits) as (check, samples):
        verify_source()
        tok, pins = tokenizer(args.snapshot)
        encoded = {}
        labels = {}
        for char in "ABCDEFGHIJKLMNOP":
            ids = tok.tok.encode(char, add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError("Bare-label tokenizer binding changed")
            labels[char] = ids[0]
        assert len(set(labels.values())) == 16
        for row in public:
            check()
            ids = prompt_ids(tok, args.snapshot, row["prompt"])
            encoded[row["id"]] = {"tokens": len(ids), "sha256": id_hash(ids), "ids": ids}
        roots = {row["root"] for row in public}
        for root in roots:
            for arm in ARMS:
                a, b = [encoded[f"{root}:{world}:{arm}"] for world in ("A", "B")]
                assert a["tokens"] == b["tokens"]
                if arm == "focused_erased":
                    assert a == b
    from pathlib import Path as P
    from hashlib import file_digest
    with P(prior).open("rb") as stream:
        prior_digest = file_digest(stream, "sha256").hexdigest()
    args.output_dir.mkdir(exist_ok=False, parents=True)
    write_new(args.output_dir / "public.json", public)
    write_new(args.output_dir / "gold.json", gold)
    write_new(args.output_dir / "tokens.json", {"row_bindings": encoded, "bare_label_token_ids": labels})
    rows = {
        "roots": len(roots), "prompts": len(public), "independently_checked_focused": n_focus,
        "maximum_prompt_tokens": max(v["tokens"] for v in encoded.values()),
        "source_digest": digest(public), "gold_digest": digest(gold), "token_digest": digest(encoded),
        "tokenizer_sha256": pins, "prior_exposed_exclusion_sha256": prior_digest,
        "max_sampled_private_bytes": max(v["private_bytes"] for v in samples),
        "min_sampled_available_RAM_bytes": min(v["available_physical_bytes"] for v in samples),
        "sealed_test_accessed_or_used_for_exclusion": not PROTECTED_OMITTED,
    }
    write_new(args.output_dir / "SOURCE_CHECK.json", rows)
    print(json.dumps(rows), flush=True)


if __name__ == "__main__":
    main()
