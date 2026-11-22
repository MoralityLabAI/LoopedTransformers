"""Extract graded proof choices from the audited original MathCan training archive."""

import argparse
from functools import lru_cache
import gzip
from pathlib import Path

from ppt_gym.data import canonical, digest, sha256_file
from ppt_gym.hybrid_rl.mathcan import MathCanReplay
from ppt_gym.live_lean import MAX_MENU
from ppt_gym.two_slot import MAX_CONTEXT_CHUNKS, goal_text, text_chunks


CHECKS = 8
SCHEMA = "ppt.two-slot-mathcan-costs.v1"


def cost_roots(replay):
    @lru_cache(maxsize=None)
    def public_goals(key):
        goals = [{"target": {"pp": goal}, "hypotheses": []} for goal in replay.observations[key]["goals"] or ()]
        if not 1 <= len(goals) <= 2 or any(len(text_chunks(goal_text(goal), 384)) > MAX_CONTEXT_CHUNKS for goal in goals):
            return ()
        return goals

    @lru_cache(maxsize=None)
    def outcomes(key):
        return {action: replay.step(key, action) for action in sorted(replay.states[key])[:MAX_MENU]}

    @lru_cache(maxsize=None)
    def paths(key):
        remaining = CHECKS - len(key[2])
        if remaining < 1 or not public_goals(key):
            return {}
        complete = {}
        for action, (status, child) in outcomes(key).items():
            if status == "archived_closure":
                complete[action] = (action,)
            elif status == "accepted_nonterminal" and remaining > 1:
                tails = paths(child)
                if tails:
                    complete[action] = (action, *min(tails.values(), key=lambda tail: (len(tail), tail)))
        return complete

    def visit(key):
        complete = paths(key)
        if len({len(path) for path in complete.values()}) > 1:
            yield {"prefix": key[2], "goals": public_goals(key), "checks_remaining": CHECKS - len(key[2]),
                   "completed": complete, "unscored": {a: outcome[0] for a, outcome in outcomes(key).items()
                                                        if a not in complete}}
        for action in complete:
            status, child = outcomes(key)[action]
            if status == "accepted_nonterminal":
                yield from visit(child)

    roots = []
    for start in replay.starts:
        states = list(visit(start))
        if states:
            archive_row = next(iter(replay.states[start].values()))
            roots.append({"root": digest(start), "module": start[0], "theorem": start[1],
                          "source_sha256": archive_row["source_sha256"], "states": states})
    return roots


def prepare(archive):
    replay = MathCanReplay(archive)
    roots = cost_roots(replay)
    if not roots:
        raise ValueError("the original training archive has no supported proof-cost contrasts")
    return {"schema": SCHEMA, "archive_sha256": sha256_file(archive), "training_only": True,
            "tactic_checks": CHECKS, "archived_training_starts": len(replay.starts), "roots": roots}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = prepare(args.archive)
    raw = (canonical(result) + "\n").encode("utf-8")
    with args.output.open("xb") as output:
        output.write(gzip.compress(raw, mtime=0) if args.output.suffix == ".gz" else raw)


if __name__ == "__main__":
    main()
