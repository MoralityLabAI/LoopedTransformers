"""Measure continuation performance when Lean's witnessed faster first action is fixed."""
import argparse
from collections import defaultdict
from contextlib import nullcontext
import gzip
import json
from pathlib import Path
from types import SimpleNamespace

from configs.analyze_ppt_deadline import validate_witness
from configs.analyze_ppt_graph_alias_pilot import pair_counts
from configs.freeze_ppt_deadline import ARMS
from configs.probe_ppt_hybrid_bank import ARRIVAL_EDITS, EditedTransportPolicy
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


DIAGNOSTIC_ARMS = ("arrival_rl", "trm_rl")


def audit(args, tree):
    configure(tree.threads, 0)
    cases = {case.root: case for case in load(args.cohort)["dev"]}
    registry = json.loads(args.cohort.read_bytes())["registration"]
    sha = sha256_file(args.cohort)
    clusters, fastest, _ = validate_witness(args.sessions / "native-witness.json", sha, cases, registry)
    trials, counts = {}, {}
    with LeanRepl(timeout=15) as repl:
        verifier = fingerprint(config.REPL_EXE, config.LEAN_EXE, repl.version)
        for arm in args.arms:
            source = args.sessions / (arm + ".sqlite")
            with ram_store.open_store(source) as db:
                m = ram_store.read_manifest(db)
                original, seen, last = {}, 0, None
                for last in ram_store.ledger(db):
                    seen += 1
                    if last["phase"] == "final":
                        episode, = last["episodes"]
                        original[episode["root"]] = episode
                weights = ram_store.checkpoint(db)["model"]
            trained_deadline = None if arm == "all_proofs_imitation" else registry["proof_deadline"]
            if (m["cohort"] != sha or m["verifier"] != verifier or m["proof_deadline"] != trained_deadline
                    or seen != m["entry_count"] or last is None or tensor_hash(weights) != last["weights_after"]):
                raise ValueError("continuation probe requires the completed frozen deadline policy")
            if set(original) != set(cases):
                raise ValueError("missing original deadline trajectories")
            model = GraphProofPolicy(m["architecture"], m["width"], residual_transport=m["residual_fact_transport"],
                                     premise_interaction=m["premise_interaction"], root_fact_adapter=m["root_fact_adapter"],
                                     arrival_transport=m["arrival_transport"])
            model.load_state_dict(weights)
            editor = EditedTransportPolicy(model, arrival_only=True) if args.root_edit else None
            actor = editor if editor is not None else model
            policy = DirectFactContinuation(actor) if args.prefer_direct_fact else actor
            opts = SimpleNamespace(cpu_seconds=m["cpu_seconds"], max_macs=m["max_macs"])
            forced = {}
            observed = {}
            certified = {}
            paths_changed = 0
            after_first = defaultdict(int)
            deadline = registry["proof_deadline"]
            for root, case in cases.items():
                witnessed = fastest[root]
                initial = original[root]
                expected_first = initial["events"][0]["selected_index"] if args.prefer_direct_fact else witnessed["early_first"]
                with editor.edit(args.root_edit) if editor is not None else nullcontext():
                    ep = rollout(repl, tree, case, policy, opts,
                                 first_choice=None if args.prefer_direct_fact else expected_first)
                a, b = initial["events"][0], ep["events"][0]
                native = replay_search(repl, case, ep)
                if (a["candidate_menu"] != b["candidate_menu"]
                        or (args.root_edit != "ablate-arrival" and b["selected_index"] != expected_first)
                        or (editor is not None and editor.hits != 1)):
                    raise ValueError("probe observed the wrong first decision")
                if not native["replay_passed"]:
                    raise ValueError("forced first-decision trace failed replay")
                if not args.prefer_direct_fact and a["selected_index"] == b["selected_index"] and (
                    initial["accepted_script"] != ep["accepted_script"] or initial["tactic_checks"] != ep["tactic_checks"]
                ):
                    raise ValueError("probe changed continuation despite the original first action")
                paths_changed += initial["accepted_script"] != ep["accepted_script"]
                on_time = int(ep["verified_closed"] and ep["within_cpu_budget"] and ep["tactic_checks"] <= deadline)
                if not on_time:
                    after_first[ep["status"]] += 1
                forced[root] = {"certified": on_time}
                observed[root] = {"certified": int(initial["verified_closed"] and initial["within_cpu_budget"]
                                                    and initial["tactic_checks"] <= deadline)}
                certified[root] = {"checks": ep["tactic_checks"], "initial_index": a["selected_index"],
                                   "forced_index": b["selected_index"], "terminal_status": ep["status"],
                                   "certificate": ep["certificate"], "native": native}
            label = args.root_edit or ("direct_fact_preference" if args.prefer_direct_fact else "forced")
            counts[arm] = {"original": pair_counts(observed, clusters), label: pair_counts(forced, clusters),
                           "miss_status_after_intervention": dict(after_first), "paths_changed": paths_changed}
            trials[arm] = {"session_sha256": sha256_file(source), "roots": certified}
    scope = ("development diagnostic, original roots with shared direct-fact continuation"
             if args.prefer_direct_fact else "development diagnostic, forced known first tactic and unchanged continuation")
    if args.root_edit:
        scope = "development diagnostic, root readout intervention with shared direct-fact continuation"
    return {"scope": scope,
            "cohort_sha256": sha, "counts": counts, "trials": trials}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cohort", "sessions", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--prefer-direct-fact", action="store_true")
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=DIAGNOSTIC_ARMS)
    parser.add_argument("--root-edit", choices=ARRIVAL_EDITS[1:])
    args = parser.parse_args()
    if len(set(args.arms)) != len(args.arms):
        parser.error("repeated continuation probe arm")
    if args.root_edit and (not args.prefer_direct_fact or list(args.arms) != ["arrival_rl"]):
        parser.error("the root readout edit requires direct-fact continuation and the arrival RL arm")
    tree = TreeBudget(2, 2)
    try:
        report = audit(args, tree)
        report["memory"] = tree.stats()
        args.output.write_bytes(gzip.compress(canonical(report).encode("utf-8"), mtime=0))
        print(report["counts"])
    finally:
        tree.close()


if __name__ == "__main__":
    main()
