"""Replay frozen recovery controllers on a fresh pending-obligation cohort."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import gzip
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from configs.freeze_ppt_pending_recovery import ARMS, CHECKS, EDITS, SCHEMA, SEEDS
from ppt_gym.data import canonical, digest, sha256_file
from ppt_gym.hybrid_rl.corpus import SearchCase
from ppt_gym.ram_limits import TreeBudget


def frozen_cohort(path):
    document = json.loads(path.read_bytes())
    reg = document["registration"]
    if (document["schema"] != SCHEMA or tuple(reg["seeds"]) != SEEDS or tuple(reg["arms"]) != ARMS
            or tuple(reg["edits"]) != EDITS or reg["train_updates"] or reg["confirmatory"]):
        raise ValueError("unregistered pending-goal transfer")
    groups = {}
    roots = set()
    for raw in document["cases"]:
        case = SearchCase(**raw)
        if case.root in roots:
            raise ValueError("repeated transfer theorem")
        roots.add(case.root)
        groups.setdefault(case.cluster, []).append(case)
    layouts = {digest([SCHEMA, shape]) for shape in document["layouts"]}
    if (len(roots) != reg["worlds"] or len(groups) != reg["pairs"] or set(groups) != layouts
            or any(len(pair) != 2 for pair in groups.values())):
        raise ValueError("incomplete native transfer cohort")
    return reg, groups


def recovery_event(episode):
    event, = (item for item in episode["events"] if item["kind"] == "recovery")
    packet = event["input"]
    before = packet["parent"]["state"]["goals"]
    after = packet["current"]["state"]["goals"]
    if (event["checks_used"] != 1 or packet["explored"] != 0 or len(packet["parent_menu"]) != 2
            or len(before) != 1 or len(after) != 2 or episode["tactic_checks"] > CHECKS):
        raise ValueError("expected two outstanding obligations after the shared tactic")
    return event


def verified(episode):
    certificate = episode["certificate"]
    if not episode["native_replay"]["replay_passed"] or not episode["within_cpu_budget"]:
        raise ValueError("transfer episode was not replayed within budget")
    if episode["verified_closed"]:
        if (not certificate or not certificate["passed"] or certificate["axioms"] != []
                or digest(certificate["source"]) != certificate["source_digest"]):
            raise ValueError("native transfer success lacks a standalone certificate")
        return True
    return False


def witness(groups, tree):
    from configs.check_ppt_conjunction_pilot import ProofLengthWitness
    from lcgrt.lean_repl import LeanRepl
    from ppt_gym.hybrid_rl.__main__ import rollout
    from ppt_gym.live_search import replay_search

    rows = {}
    options = SimpleNamespace(cpu_seconds=6, max_macs=1_200_000_000)
    policy = ProofLengthWitness()
    with LeanRepl(timeout=15) as repl:
        if repl.version != "4.19.0":
            raise ValueError("expected pinned Lean 4.19.0")
        for cluster, cases in groups.items():
            entries, observations, active_paths = [], [], []
            for case in cases:
                episodes = []
                for choice in (0, 1):
                    ep = rollout(repl, tree, case, policy, options, recovery_choice=choice)
                    ep["native_replay"] = replay_search(repl, case, ep)
                    event = recovery_event(ep)
                    if event["selected_index"] != choice:
                        raise ValueError("witness did not force the registered recovery action")
                    episodes.append(ep)
                winners = [action for action, ep in enumerate(episodes) if verified(ep)]
                if len(winners) != 1 or episodes[winners[0]]["tactic_checks"] != 7 + winners[0]:
                    raise ValueError(f"recovery did not have exactly one viable choice in {case.tid}")
                event = recovery_event(episodes[0])
                observation = event["input"]
                observations.append((observation["parent_menu"],
                                     [g["target"] for g in observation["current"]["state"]["goals"]]))
                # Both worlds work through the same active proof before touching the pending target.
                pending = observations[-1][1][1]
                path = []
                for tactic in episodes[0]["events"][1:]:
                    if tactic["kind"] == "rollback":
                        break
                    if tactic["kind"] != "tactic":
                        continue
                    target = tactic["input"]["state"]["goals"][0]["target"]
                    if target == pending:
                        break
                    path.append(target)
                active_paths.append(path)
                entries.append({"root": case.root, "winner": winners[0], "forced": episodes})
            if (entries[0]["winner"] == entries[1]["winner"] or observations[0] != observations[1]
                    or active_paths[0] != active_paths[1] or len(active_paths[0]) < 2):
                raise ValueError("pending facts did not swap the viable action at the same branch point")
            rows[cluster] = entries
            if len(rows) % 16 == 0:
                print(f"verified {len(rows)}/{len(groups)} witness pairs", flush=True)
    return {"rows": rows}


def score(groups, session, reg, seed, arm, tree):
    from lcgrt import config
    from lcgrt.lean_repl import LeanRepl
    from ppt_gym import ram_store
    from ppt_gym.hybrid_rl.__main__ import rollout
    from ppt_gym.hybrid_rl.graph_model import GraphProofPolicy
    from ppt_gym.hybrid_rl.search_control import DirectFactContinuation
    from ppt_gym.live_lean import fingerprint
    from ppt_gym.live_search import replay_search
    from ppt_gym.ram_learning import tensor_hash

    session_sha = sha256_file(session)
    if session_sha != reg["source_session_sha256"][str(seed)][arm]:
        raise ValueError("frozen evaluation checkpoint bytes changed")
    with ram_store.open_store(session) as db:
        manifest = ram_store.read_manifest(db)
        weights = ram_store.checkpoint(db)["model"]
        last = None
        for row in ram_store.ledger(db):
            last = row
    arch = "trm" if arm.startswith("trm_") else "hybrid"
    objective = "imitation" if arm == "hybrid_imitation" else "rl"
    weights_sha = tensor_hash(weights)
    if (manifest["architecture"] != arch or manifest["objective"] != objective or manifest["seed"] != seed
            or manifest["head_only"] != arm.endswith("_calibration")
            or manifest["cohort"] != reg["previous_cohort_sha256"]
            or not manifest["branch_recovery"] or not manifest["prefer_direct_fact"]
            or last is None or last["phase"] != "final" or last["weights_after"] != weights_sha):
        raise ValueError("transfer requires a complete original recovery policy")
    hybrid = arch == "hybrid"
    model = GraphProofPolicy(arch, manifest["width"], residual_transport=hybrid, premise_interaction=hybrid,
                             branch_recovery=True).eval()
    model.load_state_dict(weights)
    policy = DirectFactContinuation(model)
    options = SimpleNamespace(max_macs=manifest["max_macs"], cpu_seconds=manifest["cpu_seconds"])
    budget = {"max_macs": options.max_macs, "cpu_seconds": options.cpu_seconds, "tactic_checks": CHECKS}
    modes = EDITS if arm == ARMS[0] else (EDITS[0],)
    rows = {}
    with LeanRepl(timeout=15) as repl:
        if fingerprint(config.REPL_EXE, config.LEAN_EXE, repl.version) != manifest["verifier"]:
            raise ValueError("transfer requires the training verifier")
        for cluster, cases in groups.items():
            direction = 1 if int(digest([SCHEMA, cluster, seed, "signed-margin"])[0], 16) % 2 == 0 else -1
            for case in cases:
                trials = {}
                for mode in modes:
                    edits = []

                    def edit(_module, _args, scores):
                        size = scores.norm() / (scores.numel() ** 0.5)
                        if mode == "ablate-history":
                            changed = scores * 0
                        elif mode == "common-shift":
                            changed = scores - size
                        else:
                            changed = scores + direction * size * scores.new_tensor([1, -1])
                        edits.append({"before": scores.tolist(), "after": changed.tolist()})
                        return changed

                    hook = model.recovery_head.history.register_forward_hook(edit) if mode != EDITS[0] else nullcontext()
                    with hook:
                        ep = rollout(repl, tree, case, policy, options)
                    ep["native_replay"] = replay_search(repl, case, ep)
                    recovery_event(ep)
                    if len(edits) != int(mode != EDITS[0]) or ep["model_macs"] > options.max_macs:
                        raise ValueError("transfer policy violated the intervention dose or compute budget")
                    verified(ep)
                    trials[mode] = {"episode": ep, "edit": edits}
                rows[case.root] = trials
            if len(rows) % 32 == 0:
                print(f"scored {len(rows)}/{2 * len(groups)} worlds", flush=True)
    return {"rows": rows, "seed": seed, "arm": arm, "session_sha256": session_sha, "weights_sha256": weights_sha,
            "verifier": manifest["verifier"], "budget": budget, "parameters": manifest["parameters"]}


def worker(args, registration, groups):
    tree = TreeBudget(2, 2)
    try:
        from ppt_gym.ram_campaign import configure

        configure(tree.threads, 0)
        if args.worker == "witness":
            data = witness(groups, tree)
        else:
            seed, arm = args.worker.split(":", 1)
            seed = int(seed)
            directory = args.sessions[SEEDS.index(seed)]
            data = score(groups, directory / f"{arm}.sqlite", registration, seed, arm, tree)
        data.update(cohort_sha256=sha256_file(args.cohort), memory=tree.stats())
        name = args.worker.replace(":", "-")
        target = args.output / f"{name}.json.gz"
        partial = target.with_suffix(".partial")
        partial.write_bytes(gzip.compress(canonical(data).encode("utf-8"), mtime=0))
        partial.replace(target)
        print(name, "verified", len(data["rows"]), flush=True)
    finally:
        tree.close()


def campaign(args):
    args.output.mkdir(exist_ok=True)
    prefix = [sys.executable, "-u", "-m", "configs.run_ppt_pending_recovery",
              str(args.cohort), str(args.output), *map(str, args.sessions), "--worker"]

    def run(task):
        name = task.replace(":", "-")
        output = args.output / f"{name}.json.gz"
        if output.exists():
            with gzip.open(output, "rt", encoding="utf-8") as stream:
                if json.load(stream)["cohort_sha256"] != sha256_file(args.cohort):
                    raise ValueError("cannot resume a changed transfer cohort")
            print(name, "already complete", flush=True)
        else:
            with (args.output / f"{name}.log").open("ab") as log:
                status = subprocess.run([*prefix, task], stdout=log, stderr=subprocess.STDOUT).returncode
            if status:
                raise RuntimeError(f"transfer worker {name} exited with {status}")
            print(name, "complete", flush=True)

    tree = TreeBudget(6, 2)
    try:
        run("witness")
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(run, (f"{seed}:{arm}" for seed in SEEDS for arm in ARMS)))
        (args.output / "resources.json").write_text(canonical(tree.stats()) + "\n", encoding="utf-8")
    finally:
        tree.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cohort", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("sessions", type=Path, nargs=len(SEEDS))
    parser.add_argument("--workers", type=int, choices=range(1, len(ARMS) + 1), default=len(ARMS))
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args()
    registration, groups = frozen_cohort(args.cohort)
    if args.worker:
        worker(args, registration, groups)
    else:
        campaign(args)


if __name__ == "__main__":
    main()
