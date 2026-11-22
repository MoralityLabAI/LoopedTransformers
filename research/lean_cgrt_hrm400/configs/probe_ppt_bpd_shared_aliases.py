"""Audit paired local definitions and test raw-only opening-score invariance."""

import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path

from configs.check_ppt_bpd_shared_aliases import SCHEMA, layouts, scripts, world
from configs.run_ppt_shared_rewrite import model
from ppt_gym.data import canonical, digest, sha256_file
from ppt_gym.hybrid_rl.corpus import SearchCase
from ppt_gym.ram_limits import TreeBudget


def read_witness(path, split):
    from ppt_gym.alias_rewrite import alias_menu, local_definitions

    data = json.loads(gzip.decompress(path.read_bytes()))
    if data["schema"] != SCHEMA or data["split"] != split or data["tactic_checks"] != 8:
        raise ValueError("unexpected local-definition witness")
    groups = defaultdict(list)
    for key, row in data["rows"].items():
        shape = tuple(row["shape"])
        case, definitions, _, rewrites, targets, aliases = world(split, shape, row["swapped"])
        if SearchCase(**row["case"]) != case or case.root != key:
            raise ValueError("a local-definition case changed from its registered layout")
        groups[shape].append(row)
        plans = scripts(definitions, rewrites, targets, aliases)
        packet = json.loads(row["packet"])
        current = packet["state"]
        menu = alias_menu(current)
        raw = [{key: value for key, value in goal.items() if key != "pp"} for goal in current["goals"]]
        if (local_definitions(current["goals"][0]) != definitions or row["raw_state_sha256"] != digest({"goals": raw, "menu": menu})
                or set(plans) != set(row["paths"]) or row["viable_openings"] != sorted(name for name, path in plans.items() if len(path) <= 8)):
            raise ValueError("the alias definitions, raw observation, or feasible opening set changed")
        for action, plan in row["paths"].items():
            events, cert = plan["events"], plan["certificate"]
            if (events[0]["packet"] != row["packet"] or tuple(events[0]["menu"]) != menu or len(events) != plan["checks"]
                    or plans[action] != [step["tactic"] for step in events] or bool(events[7]["targets"]) != (action not in row["viable_openings"])
                    or not cert["passed"] or cert["axioms"] or digest(cert["source"]) != cert["source_digest"]):
                raise ValueError("a local-definition path lost its decoded observation, proof budget, or certificate")
            for i, step in enumerate(events):
                if "packet" not in step:
                    if split == "train":
                        raise ValueError("an alias training tactic lacks its public state")
                    continue
                obs = json.loads(step["packet"])
                if (digest(obs["state"]) != step["before"] or obs["checks_remaining"] != max(0, 8 - i)
                        or alias_menu(obs["state"]) != tuple(step["menu"]) or step["tactic"] not in step["menu"]):
                    raise ValueError("an alias observation no longer matches the native tactic receipt")
    if set(groups) != set(layouts()[split]) or any({row["swapped"] for row in pair} != {False, True} or len(pair) != 2 for pair in groups.values()):
        raise ValueError("the witness lost a registered alias pair")
    for left, right in groups.values():
        if (left["raw_state_sha256"] != right["raw_state_sha256"] or left["packet"] == right["packet"]
                or not set(left["viable_openings"]).isdisjoint(right["viable_openings"])):
            raise ValueError("a pair lost its identical raw state or disjoint viable openings")
    return data, groups


def audit(args, tree):
    import torch

    from ppt_gym.ram_learning import tensor_hash
    from ppt_gym.shared_rewrite import ConjunctiveGraphPolicy

    witness, groups = read_witness(args.witness, args.split)
    hybrid, _ = model(args.warm)
    trm_checkpoint = torch.load(args.trm, map_location="cpu", weights_only=True)
    policies = {"adapted_hybrid": hybrid, "continued_graph_trm": ConjunctiveGraphPolicy(**trm_checkpoint["architecture"]).eval()}
    paths = {"adapted_hybrid": args.hybrid, "continued_graph_trm": args.trm}
    report = {}
    for identity, policy in policies.items():
        checkpoint = torch.load(paths[identity], map_location="cpu", weights_only=True)
        policy.load_state_dict(checkpoint["state_dict"])
        if tensor_hash(policy.state_dict()) != checkpoint["weights_sha256"]:
            raise ValueError("the existing raw-only checkpoint changed on reload")
        viable = 0
        with torch.no_grad():
            for left, right in groups.values():
                menu = tuple(left["paths"]["rw [h0]"]["events"][0]["menu"])
                depth = 4 if policy.architecture == "hybrid" else 1
                a, b = (policy(row["packet"], menu, depth)[0] for row in (left, right))
                if not torch.equal(a, b):
                    raise ValueError("the raw-only model unexpectedly distinguished a local-definition pair")
                viable += sum(menu[int(a.argmax())] in row["viable_openings"] for row in (left, right))
        report[identity] = {"identical_logits_pairs": len(groups), "viable_openings": viable,
                            "certified_pair_upper_bound": 0, "checkpoint_sha256": sha256_file(paths[identity])}
    print(args.split, len(witness["rows"]), "worlds", report, flush=True)
    return {"witness_sha256": sha256_file(args.witness), "split": args.split,
            "certified_paths": sum(len(row["paths"]) for row in witness["rows"].values()),
            "results": report, "memory": tree.stats()}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("split", choices=("train", "dev"))
    for name in ("witness", "warm", "hybrid", "trm", "output"):
        cli.add_argument(name, type=Path)
    args = cli.parse_args()
    tree = TreeBudget(2, 1)
    try:
        from ppt_gym.ram_campaign import configure

        configure(tree.threads, 103)
        result = audit(args, tree)
        with args.output.open("xb") as handle:
            handle.write(gzip.compress(canonical(result).encode("utf-8"), mtime=0))
    finally:
        tree.close()


if __name__ == "__main__":
    main()
