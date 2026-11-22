"""Measure frozen BPD rewrite checkpoints on verified eight-check root choices."""

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path

from configs.probe_ppt_bpd_rewrite_components import hop_components
from configs.probe_ppt_bpd_rewrite_revisits import audited_policies
from ppt_gym.data import canonical, digest, sha256_file
from ppt_gym.hybrid_rl.corpus import load
from ppt_gym.ram_limits import TreeBudget


def audit(args, tree):
    import torch

    from lcgrt.lean_repl import LeanRepl
    from ppt_gym.live_lean import candidate_menu, state_input
    from ppt_gym.ram_campaign import configure

    configure(tree.threads, 151)
    cases = {case.root: case for case in load(args.deadline_cohort)["dev"]}
    cohort_sha = sha256_file(args.deadline_cohort)
    witness = json.loads(gzip.decompress(args.witness.read_bytes()))
    if witness["cohort_sha256"] != cohort_sha:
        raise ValueError("the native deadline witness names a different cohort")
    groups = witness["pairs"]
    worlds = {world["root"]: world for pair in groups.values() for world in pair}
    if set(cases) != set(worlds) or len(cases) != 2 * len(groups):
        raise ValueError("the native witness has incomplete deadline coverage")
    for pair in groups.values():
        if len(pair) != 2 or {row["winner"] for row in pair} != {0, 1}:
            raise ValueError("the root choice does not switch within the native pair")
        for row in pair:
            for proposal in row["proposals"]:
                win = proposal["action"] == row["winner"]
                cert = proposal["certificate"]
                if (not proposal["replay"]["replay_passed"] or proposal["checks"] != 8
                        or proposal["certified"] != win
                        or (win and (not cert["passed"] or cert["axioms"]
                                    or digest(cert["source"]) != cert["source_digest"]))):
                    raise ValueError("a root choice lacks its matching native proof witness")
    source_sha = sha256_file(args.source_cohort)
    source_reference_sha = sha256_file(args.source_reference)
    results = {}
    with torch.no_grad(), LeanRepl(timeout=15) as repl:
        if repl.version != witness["native_lean"]:
            raise ValueError("the native deadline toolchain changed")
        packets = {}
        for root, case in cases.items():
            repl.reset()
            state = repl.init(case.statement, pp=True)
            actions = candidate_menu(repl, state, rewrite=True)
            if actions != ("rw [h0]", "rw [h1]"):
                raise ValueError("the cold deadline root action menu changed")
            packet, _ = state_input(repl, state, (), 8)
            packets[root] = (packet, actions)
        for arm, model, source, _ in audited_policies(args.sessions, args.audit, source_sha, source_reference_sha):
            choices = {}
            for root, (packet, actions) in packets.items():
                passes = 4 if model.architecture == "hybrid" else 1
                logits, _ = model(packet, actions, passes)
                scores = {"full": logits}
                if model.graph.transport is not None:
                    terms = logits.new_tensor(hop_components(model, packet, actions)).sum(dim=1)
                    scores["transport"] = terms.sum(dim=0)
                    # The root rewrite itself costs one of the eight available checks.
                    scores["transport_through_seven_hops"] = terms[:8].sum(dim=0)
                if any(not score.isfinite().all() for score in scores.values()):
                    raise ValueError("the frozen root policy produced a nonfinite score")
                choices[root] = {condition: int(score.argmax()) for condition, score in scores.items()}
            conditions = next(iter(choices.values()))
            counts = {name: Counter(aligned=0, misleading=0) for name in conditions}
            for pair in groups.values():
                for name in conditions:
                    counts[name][pair[0]["stratum"]] += int(
                        all(choices[row["root"]][name] == row["winner"] for row in pair)
                    )
            results[arm] = {"checkpoint": source, "complete_root_choice_pairs": counts, "choices": choices}
            print(arm, {name: dict(values) for name, values in counts.items()}, flush=True)
    return {"scope": "root choices on fresh equality deadline development pairs",
            "native_witness_sha256": sha256_file(args.witness), "deadline_cohort_sha256": cohort_sha,
            "source_audit_sha256": sha256_file(args.audit), "results": results, "memory": tree.stats()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("deadline_cohort", "witness", "source_cohort", "source_reference", "sessions", "audit", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    tree = TreeBudget(2, 1)
    try:
        report = audit(args, tree)
        payload = gzip.compress(canonical(report).encode("utf-8"), mtime=0)
        with args.output.open("xb") as output:
            output.write(payload)
    finally:
        tree.close()


if __name__ == "__main__":
    main()
