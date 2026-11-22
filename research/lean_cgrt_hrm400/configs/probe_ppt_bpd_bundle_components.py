"""Measure which frozen scores distinguish viable bundle openings."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

from configs.probe_ppt_bpd_rewrite_revisits import audited_policies, warm_policies
from ppt_gym.data import canonical, sha256_file
from ppt_gym.hybrid_rl.corpus import load
from ppt_gym.ram_limits import TreeBudget


def score_components(model, packet, menu, passes):
    import torch

    from ppt_gym.hybrid_rl.graph_model import visible_graph
    from ppt_gym.hybrid_rl.model import ProofPolicy

    full, _ = model(packet, menu, passes)
    lexical, value = ProofPolicy.forward(model, packet, menu, passes)
    preview, _ = model.score_openings(packet, menu, passes, torch.zeros_like(lexical),
                                      torch.zeros_like(value), visible_graph(packet, menu))
    if not torch.allclose(full, lexical + preview, atol=1e-6, rtol=1e-5):
        raise ValueError("the lexical and preview scores do not reconstruct the model decision")
    variants = {"full": full, "lexical": lexical, "preview": preview}
    if model.graph.transport is not None:
        with model.graph.transport.register_forward_hook(lambda _module, _args, value: torch.zeros_like(value)):
            without_transport, _ = model(packet, menu, passes)
        variants["without_transport"] = without_transport
    if not all(torch.isfinite(scores).all() for scores in variants.values()):
        raise ValueError("a bundle opening score was nonfinite")
    return {mode: {"opening": int(scores.argmax()), "margin": float(scores[0] - scores[1])}
            for mode, scores in variants.items()}


def audit(args, tree):
    import torch

    from lcgrt.lean_repl import LeanRepl
    from ppt_gym.live_lean import candidate_menu, state_input
    from ppt_gym.ram_campaign import configure

    cohort_sha, reference_sha = map(sha256_file, (args.cohort, args.reference))
    registration = json.loads(args.cohort.read_bytes())["registration"]
    reference = json.loads(args.reference.read_bytes())
    witnesses = reference["rows"]
    cases = load(args.cohort)["dev"]
    if (reference["cohort_sha256"] != cohort_sha or reference["kind"] != "bundled-equality"
            or set(witnesses) != {case.root for case in cases}):
        raise ValueError("the bundle probe requires the matching native reference")
    seed, = registration["model_seeds"]
    configure(tree.threads, seed)
    groups, inputs = defaultdict(list), {}
    with LeanRepl(timeout=15) as repl:
        if repl.version != reference["lean"]:
            raise ValueError("the native bundle toolchain changed")
        for case in cases:
            repl.reset()
            initial = repl.init(case.statement, pp=True)
            menu = candidate_menu(repl, initial, rewrite=True, bundles=True)
            if list(menu) != witnesses[case.root]["proposals"][0]["menu"]:
                raise ValueError("the bundle probe observed a different public opening menu")
            inputs[case.root] = state_input(repl, initial, (), registration["tactic_checks"])[0], menu
            groups[case.cluster].append(case.root)

    previous = json.loads(args.audit.read_bytes()) if args.audit else None
    policies = (audited_policies(args.sessions, args.audit, cohort_sha, reference_sha) if previous
                else warm_policies(args.initialization, registration, seed))
    results = {}
    with torch.no_grad():
        for arm, model, checkpoint, _ in policies:
            passes = 4 if model.architecture == "hybrid" else 1
            rows = {root: score_components(model, *observation, passes) for root, observation in inputs.items()}
            if previous:
                audited = previous["trials"][arm]["bundled_evaluation"]["final"]
                if any(item["opening"] is not None and item["opening"] != rows[root]["full"]["opening"]
                       for root, item in audited.items()):
                    raise ValueError("the frozen probe did not reproduce the audited opening")
            counts = {}
            for mode in next(iter(rows.values())):
                viable = {root for root, row in rows.items() if row[mode]["opening"] == witnesses[root]["winner"]}
                counts[mode] = {"viable_worlds": len(viable),
                                "viable_pairs": sum(viable.issuperset(worlds) for worlds in groups.values())}
            results[arm] = {"checkpoint": checkpoint, "counts": counts, "opening_scores": rows}
            print(arm, counts, flush=True)
    return {"scope": "development opening choices scored against native branch feasibility",
            "cohort_sha256": cohort_sha, "reference_sha256": reference_sha,
            "audit_sha256": sha256_file(args.audit) if args.audit else None,
            "results": results, "memory": tree.stats()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cohort", "reference", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--initialization", type=Path)
    parser.add_argument("--sessions", type=Path)
    parser.add_argument("--audit", type=Path)
    args = parser.parse_args()
    if bool(args.sessions) != bool(args.audit) or bool(args.initialization) == bool(args.audit):
        parser.error("supply --initialization or both --sessions and --audit")
    tree = TreeBudget(2, 1)
    try:
        report = audit(args, tree)
        with args.output.open("x", encoding="utf-8", newline="\n") as output:
            output.write(canonical(report) + "\n")
    finally:
        tree.close()


if __name__ == "__main__":
    main()
