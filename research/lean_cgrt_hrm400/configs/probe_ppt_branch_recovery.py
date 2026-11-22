"""Remove the learned recurrent branch comparison and test the matched sham in Lean."""

import argparse
from contextlib import nullcontext
import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from configs.analyze_ppt_hybrid_replication import read_arm
from lcgrt import config
from lcgrt.lean_repl import LeanRepl
from ppt_gym import ram_store
from ppt_gym.data import canonical, sha256_file
from ppt_gym.hybrid_rl.__main__ import rollout
from ppt_gym.hybrid_rl.corpus import load
from ppt_gym.hybrid_rl.graph_model import GraphProofPolicy
from ppt_gym.hybrid_rl.search_control import DirectFactContinuation
from ppt_gym.live_lean import fingerprint
from ppt_gym.live_search import replay_search
from ppt_gym.ram_campaign import configure
from ppt_gym.ram_learning import tensor_hash
from ppt_gym.ram_limits import TreeBudget


MODES = ("pristine", "ablate-recovery-history", "matched-common-shift")


def run(cohort, session, tree):
    configure(2, 0)
    cases = {case.root: case for case in load(cohort)["dev"]}
    cohort_sha = sha256_file(cohort)
    trial = read_arm(session, cases, cohort_sha)
    manifest = trial["manifest"]
    if not (manifest["architecture"] == "hybrid" and manifest["branch_recovery"] and manifest["prefer_direct_fact"]):
        raise ValueError("expected complete hybrid recovery trial")
    with ram_store.open_store(session) as db:
        weights = ram_store.checkpoint(db)["model"]
        original = {ep["root"]: ep for row in ram_store.ledger(db) if row["phase"] == "final" for ep in row["episodes"]}
    if tensor_hash(weights) != trial["final_weights_sha256"]:
        raise ValueError("recovery checkpoint changed")
    model = GraphProofPolicy("hybrid", manifest["width"], residual_transport=True, premise_interaction=True,
                             branch_recovery=True)
    model.load_state_dict(weights)
    policy = DirectFactContinuation(model)
    params = SimpleNamespace(max_macs=manifest["max_macs"], cpu_seconds=manifest["cpu_seconds"])
    report = {"cohort_sha256": cohort_sha, "session_sha256": trial["session_sha256"],
              "weights_sha256": trial["final_weights_sha256"], "roots": {}}
    with LeanRepl(timeout=15) as repl:
        if fingerprint(config.REPL_EXE, config.LEAN_EXE, repl.version) != manifest["verifier"]:
            raise ValueError("recovery intervention requires the original native verifier")
        for root, case in cases.items():
            rows = {}
            for mode in MODES:
                hits = 0

                def edit(_module, _inputs, scores):
                    nonlocal hits
                    hits += 1
                    return (torch.zeros_like(scores) if mode == "ablate-recovery-history"
                            else scores - scores.norm() / (scores.numel() ** 0.5))

                ctx = model.recovery_head.history.register_forward_hook(edit) if mode != "pristine" else nullcontext()
                with ctx:
                    ep = rollout(repl, tree, case, policy, params)
                native = replay_search(repl, case, ep)
                event, = (e for e in ep["events"] if e["kind"] == "recovery")
                if hits != int(mode != "pristine") or not native["replay_passed"]:
                    raise ValueError("recovery edit or native replay departed from the declared single intervention")
                if mode == "pristine" and (ep["accepted_script"] != original[root]["accepted_script"]
                                           or ep["tactic_checks"] != original[root]["tactic_checks"]):
                    raise ValueError("unperturbed recovery policy did not reproduce the recorded proof")
                rows[mode] = {"choice": event["selected_index"], "checks": ep["tactic_checks"],
                              "success": ep["verified_closed"] and ep["within_cpu_budget"], "hits": hits,
                              "certificate": ep["certificate"], "native_replay": native}
            report["roots"][root] = rows
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cohort", "session", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    tree = TreeBudget(2, 2)
    try:
        report = run(args.cohort, args.session, tree)
        report["memory"] = tree.stats()
        args.output.write_bytes(gzip.compress(canonical(report).encode("utf-8"), mtime=0))
        for mode in MODES:
            print(mode, sum(row[mode]["success"] for row in report["roots"].values()))
    finally:
        tree.close()


if __name__ == "__main__":
    main()
