"""Independent, CPU-only audit of published RSI checkpoint selection."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

UPSTREAM = "765453d13878804c49876a656f39b57a4384dca4"
MODES = ("R3", "R4", "RK2")
N = 64


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def counts(gate):
    def count(m):
        value = gate[m]
        if isinstance(value, dict):
            return int(value["correct"])
        return round(value * N / 100)
    per = {m: count(m) for m in MODES}
    if any(not 0 <= v <= N for v in per.values()) or sum(per.values()) != gate["total_correct"]:
        raise ValueError("inconsistent gate counts")
    return per


def acceptance(previous, candidate):
    """Replay the public gate semantics against rounded percentages, not training."""
    p, c = counts(previous), counts(candidate)
    drops = {m: p[m] - c[m] for m in MODES}
    repetition_ok = candidate["max_rep"] <= max(3.12, previous["max_rep"] + 1.6)
    mode_ok = max(drops.values()) <= 2
    completion_gain = round(candidate["mean_comp"] * 3 * N / 100) - round(previous["mean_comp"] * 3 * N / 100)
    total_gain = candidate["total_correct"] - previous["total_correct"]
    accepted = repetition_ok and mode_ok and (total_gain > 0 or (total_gain == 0 and completion_gain >= 4))
    return {"accepted_by_rule": accepted, "drops": drops, "total_gain": total_gain,
            "completion_gain": completion_gain, "repetition_ok": repetition_ok, "mode_ok": mode_ok}


def replay(record):
    if record["gate_size_per_mode"] != N or record["total_gate_evals"] != 3 * N:
        raise ValueError("unexpected gate size")
    previous = record["init_gate"]
    rows = []
    for generation in record["generations"]:
        if not generation["accepted"]:
            continue
        matches = [c for c in generation["candidates"] if c["scale"] == generation["winner_scale"]]
        if len(matches) != 1:
            raise ValueError("winner missing or ambiguous")
        candidate = matches[0]
        rows.append({"generation": generation["gen"], "previous": previous["total_correct"],
                     "candidate": candidate["total_correct"], "reported_accepted": True,
                     **acceptance(previous, candidate)})
        # Compare with the reported trajectory, even after a disagreement.
        previous = candidate
    return {"rows": rows, "disagreements": sum(not r["accepted_by_rule"] for r in rows),
            "evidence_class": "published_record_replay", "counterfactual_training_replay": False}


def roots(seed=729001, n=8):
    """Public development-only pointer chase, independent RNG seed per source root."""
    if not 1 <= n <= 256:
        raise ValueError("n must be 1..256")
    out = []
    for i in range(n):
        rng = random.Random(seed + i)
        perm = list(range(11))
        rng.shuffle(perm)
        start = rng.randrange(11)
        hops = rng.randint(3, 8)
        answer = start
        for _ in range(hops):
            answer = perm[answer]
        table = ", ".join(f"{k}->{v}" for k, v in enumerate(perm))
        prompt = (f"Follow the mapping {table}. Start at {start} and apply the mapping "
                  f"exactly {hops} times. Finish with a separate line ANSWER: <integer>.")
        out.append({"root_id": f"pointer-{seed+i}", "prompt": prompt, "answer": answer})
    return out


def grade(text, answer):
    # Strip only known generation terminators; demand one explicit final answer line.
    cleaned = text.replace("<|message_end|>", "").replace("<|endoftext|>", "").strip()
    matches = re.findall(r"(?m)^ANSWER:\s*([+-]?\d+)[ \t]*$", cleaned)
    final = re.search(r"(?:^|\n)ANSWER:[ \t]*([+-]?\d+)[ \t]*$", cleaned)
    words = cleaned.split()
    grams = {}
    for i in range(max(0, len(words) - 7)):
        g = tuple(words[i:i+8])
        grams[g] = grams.get(g, 0) + 1
    repeated = any(v >= 5 for v in grams.values())
    completed = len(matches) == 1 and final is not None
    return {"completed": completed, "repetition": repeated,
            "correct": bool(completed and not repeated and int(final.group(1)) == answer)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--record", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    result = replay(json.loads(args.record.read_text(encoding="utf-8")))
    result["input_sha256"] = digest(args.record)
    result["upstream_commit"] = UPSTREAM
    with args.out.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
