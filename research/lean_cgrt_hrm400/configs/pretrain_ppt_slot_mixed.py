"""Warm the two-slot policy and TRM on the same audited source training traces."""

import argparse
from collections import Counter, defaultdict
import json
from itertools import zip_longest
from pathlib import Path

from ppt_gym.data import digest, sha256_file
from ppt_gym.hybrid_rl.corpus import load
from ppt_gym.ram_limits import TreeBudget

from configs.run_ppt_two_slot import TRM_SCHEMA, read_fit


SCHEMA = "ppt.two-slot-mixed-warm.v1"
MATHCAN = "mathcan_ram08_audited_train"
ACCEPTED = {"accepted_nonterminal", "kernel_closed", "archived_closure"}


def corpus(cohort, audit_path, ledger_path):
    from ppt_gym import ram_store
    from ppt_gym.two_slot import MAX_CONTEXT_CHUNKS, goal_text, text_chunks

    audit = json.loads(audit_path.read_bytes())
    trial = audit["trials"]["hybrid_rl"]
    if sha256_file(cohort) != audit["cohort_sha256"] or sha256_file(ledger_path) != trial["session_sha256"]:
        raise ValueError("the mixed warm start requires its completed, audited training ledger")
    train = {case.root: case.source for case in load(cohort)["train"]}
    by_source, seen = defaultdict(list), defaultdict(set)
    exposures, native, archived = Counter(), Counter(), Counter()
    with ram_store.open_store(ledger_path) as db:
        if ram_store.read_manifest(db) != trial["manifest"]:
            raise ValueError("the mixed training session changed its manifest")
        for row in ram_store.ledger(db):
            if row["phase"] not in ("train", "mathcan"):
                continue
            for episode in row["episodes"]:
                root = episode["root"]
                source = train[root] if row["phase"] == "train" else episode["source"]
                exposures[source] += 1
                if row["phase"] == "mathcan":
                    if source != MATHCAN:
                        raise ValueError("unexpected source in the archived MathCan phase")
                    if not episode.get("archived_closure"):
                        continue
                    archived[source] += 1
                else:
                    if not episode["verified_closed"] or not episode["within_cpu_budget"]:
                        continue
                    cert = episode["certificate"]
                    if (not episode["live_kernel"] or not cert["passed"] or cert["axioms"]
                            or digest(cert["source"]) != cert["source_digest"]):
                        raise ValueError("a native warm-start proof lost its certification")
                    native[source] += 1
                if root in seen[source]:
                    continue
                examples = []
                for event in episode["events"]:
                    if event["kind"] != "tactic" or event["status"] not in ACCEPTED:
                        continue
                    packet = event["input"]
                    goals = packet["state"]["goals"]
                    if not 1 <= len(goals) <= 2 or any(
                            len(text_chunks(goal_text(goal), 384)) > MAX_CONTEXT_CHUNKS for goal in goals):
                        continue
                    actions = event["candidate_menu"]
                    indices = event.get("rewrite_choices", list(range(len(actions))))
                    selected = event["selected_index"]
                    if selected not in indices:
                        raise ValueError("a selected training proof tactic was excluded from its own menu")
                    remaining = episode["tactic_checks"] - event["attempt"] + 1
                    examples.append((goals, [actions[i] for i in indices], indices.index(selected),
                                     packet["checks_remaining"], 1. - .025 * remaining))
                if examples:
                    by_source[source].append((root, examples))
                    seen[source].add(root)

    for source, expected in trial["source_exposures"].items():
        if (exposures[source] != expected["episodes"] or native[source] != expected.get("native_certificates", 0)
                or archived[source] != expected.get("archived_closures", 0) or not by_source[source]):
            raise ValueError(f"the warm start lost audited training coverage for {source}")
    return by_source, dict(exposures)


def pretrain(policy, sources):
    import torch
    from torch.nn import functional as F

    optimizer = torch.optim.AdamW(policy.parameters(), lr=.0005)
    groups = [sources[key] for key in sorted(sources)]
    for round_ in zip_longest(*groups):
        losses = []
        for entry in round_:
            if entry is None:
                continue
            _, examples = entry
            terms = []
            for goals, actions, selected, checks, target in examples:
                scores, _ = policy(goals, actions, checks)
                value = scores[selected]
                terms.append(-scores.log_softmax(0)[selected]
                             + .25 * F.smooth_l1_loss(value, value.new_tensor(target)))
            losses.append(torch.stack(terms).mean())
        optimizer.zero_grad()
        torch.stack(losses).mean().backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()


def main():
    from ppt_gym.ram_campaign import configure

    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "trm", "cohort", "audit", "ledger", "output"):
        cli.add_argument(name, type=Path)
    args = cli.parse_args()
    if args.output.exists():
        cli.error("output already exists")
    tree = TreeBudget(2, 1)
    try:
        configure(tree.threads, 831)
        import torch

        from ppt_gym.ram_learning import tensor_hash
        from ppt_gym.two_slot import TwoSlotPolicy, TwoSlotTRM

        fit = read_fit(args.source)
        trm = torch.load(args.trm, weights_only=True, map_location="cpu")
        parents = {"hybrid": sha256_file(args.source), "trm": sha256_file(args.trm)}
        if not fit.get("bindings") or trm["schema"] != TRM_SCHEMA or trm["parent_sha256"] != parents["hybrid"]:
            raise ValueError("the warm start requires a matched typed policy pair")
        sources, exposures = corpus(args.cohort, args.audit, args.ledger)
        counts = {key: {"roots": len(rows), "decisions": sum(len(examples) for _, examples in rows)}
                  for key, rows in sorted(sources.items())}
        weights, hashes = {}, {}
        for name, model, state in (("hybrid", TwoSlotPolicy(bindings=True), fit["learned"]),
                                   ("trm", TwoSlotTRM(bindings=True), trm["weights"])):
            model.load_state_dict(state)
            pretrain(model, sources)
            weights[name] = model.state_dict()
            hashes[name] = tensor_hash(weights[name])
            print(name, hashes[name], counts, flush=True)
        with args.output.open("xb") as output:
            torch.save({"schema": SCHEMA, "parents": parents, "ledger_sha256": sha256_file(args.ledger),
                        "audit_sha256": sha256_file(args.audit), "cohort_sha256": sha256_file(args.cohort),
                        "examples_sha256": digest(sources), "exposures": exposures, "counts": counts,
                        "weights": weights, "weight_hashes": hashes, "memory": tree.stats()}, output)
    finally:
        tree.close()


if __name__ == "__main__":
    main()
