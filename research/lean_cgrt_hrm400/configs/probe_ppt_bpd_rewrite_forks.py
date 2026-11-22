"""Collect paired native proof outcomes at interior BPD rewrite decisions."""

import argparse
from collections import Counter, defaultdict
import gzip
from itertools import combinations
import json
from pathlib import Path
from types import SimpleNamespace

from configs.probe_ppt_bpd_rewrite_revisits import PreferNewEquality, RewriteVisits, audited_policies
from ppt_gym import ram_store
from ppt_gym.data import canonical, digest, sha256_file
from ppt_gym.hybrid_rl.corpus import load
from ppt_gym.ram_limits import TreeBudget
from ppt_gym.rewrite import BPD_REWRITE_FAMILY


def comparable(a, b):
    return (a["public_packet_sha256"] == b["public_packet_sha256"]
            and a["candidate_menu"] == b["candidate_menu"] and a.get("rewrite_choices") == b.get("rewrite_choices"))


def training_forks(cohort, audit_path, session):
    from ppt_gym.hybrid_rl.__main__ import proof_return

    audit = json.loads(audit_path.read_bytes())
    info = audit["trials"]["hybrid_rl"]
    if audit["cohort_sha256"] != sha256_file(cohort) or info["session_sha256"] != sha256_file(session):
        raise ValueError("expected the completed original rewrite RL training ledger")
    cases = {case.root: case for case in load(cohort)["train"] if case.family == BPD_REWRITE_FAMILY}
    forks, counts, depth = {}, Counter(), Counter()
    with ram_store.open_store(session) as db:
        for row in ram_store.ledger(db):
            if row["phase"] != "train":
                continue
            by_root = defaultdict(list)
            for ep in row["episodes"]:
                if ep["root"] in cases and ep["within_cpu_budget"] and ep["native_replay"]["replay_passed"]:
                    by_root[ep["root"]].append(ep)
            for root, episodes in by_root.items():
                counts["training_roots"] += 1
                for a, b in combinations(episodes, 2):
                    other = {event["public_packet_sha256"]: event for event in b["events"] if event["kind"] == "tactic"}
                    for event in a["events"]:
                        if event["kind"] != "tactic":
                            continue
                        match = other.get(event["public_packet_sha256"])
                        if match is None or not comparable(event, match) or event["selected_index"] == match["selected_index"]:
                            continue
                        counts["observed_forks"] += 1
                        if proof_return(a) != proof_return(b):
                            counts["unequal_returns"] += 1
                            counts["unequal_returns_without_certificate"] += int(not a["verified_closed"] and not b["verified_closed"])
                            depth[event["attempt"]] += 1
                        break
                candidates = []
                for episode in episodes:
                    prefix = []
                    for event in episode["events"]:
                        if event["kind"] != "tactic":
                            break
                        allowed = event.get("rewrite_choices", list(range(len(event["candidate_menu"]))))
                        if prefix and len(allowed) > 1:
                            candidates.append((list(prefix), event, allowed))
                            break
                        if event["status"] != "accepted_nonterminal":
                            break
                        prefix.append(event)
                if candidates:
                    forks[root] = min(candidates, key=lambda item: (len(item[0]), item[1]["public_packet_sha256"]))
    return audit, cases, forks, {**counts, "unequal_returns_by_attempt": dict(depth), "interior_forks_available": len(forks)}


class ForcedPrefix(PreferNewEquality):
    def __init__(self, model, visits, events):
        super().__init__(model, visits)
        self.events = events

    def __call__(self, packet, menu, passes, **options):
        import torch

        attempt = 9 - json.loads(packet)["checks_remaining"]
        if attempt <= len(self.events):
            event, index = self.events[attempt - 1]
            if digest(packet) != event["public_packet_sha256"] or list(menu) != event["candidate_menu"]:
                raise ValueError("the native fork no longer shares the archived public prefix")
            logits = torch.full((len(menu),), -1.)
            logits[index] = 1.
            return logits, logits.new_zeros(())
        return super().__call__(packet, menu, passes, **options)


def collect(args, tree, audit, cases, forks):
    import torch

    from lcgrt import config
    from lcgrt.lean_repl import LeanRepl
    from ppt_gym.hybrid_rl.__main__ import proof_return, rollout
    from ppt_gym.live_lean import fingerprint
    from ppt_gym.live_search import replay_search
    from ppt_gym.ram_campaign import configure

    configure(tree.threads, audit["seed"])
    _, model, checkpoint, _ = next(audited_policies(args.sessions, args.audit, audit["cohort_sha256"],
                                                   audit["reference_sha256"], selected=("hybrid_rl",)))
    manifest = audit["trials"]["hybrid_rl"]["manifest"]
    options = SimpleNamespace(cpu_seconds=manifest["cpu_seconds"], max_macs=manifest["max_macs"])
    rows = {}
    with torch.no_grad(), LeanRepl(timeout=15) as repl:
        if fingerprint(config.REPL_EXE, config.LEAN_EXE, repl.version) != manifest["verifier"]:
            raise ValueError("the rewrite experiment uses the pinned Lean verifier")
        for root, (prefix, fork, allowed) in sorted(forks.items()):
            traces = []
            for choice in allowed:
                proposals = [*((event, event["selected_index"]) for event in prefix), (fork, choice)]
                visits = RewriteVisits(repl)
                episode = rollout(visits, tree, cases[root], ForcedPrefix(model, visits, proposals), options)
                replay = replay_search(repl, cases[root], episode)
                if (not replay["replay_passed"] or not episode["within_cpu_budget"]
                        or len(episode["events"]) < len(proposals)):
                    raise ValueError("the interior rewrite fork did not replay within its budget")
                for actual, (expected, index) in zip(episode["events"][:len(proposals)], proposals, strict=True):
                    if (actual["kind"] != "tactic" or actual["public_packet_sha256"] != expected["public_packet_sha256"]
                            or actual["candidate_menu"] != expected["candidate_menu"] or actual["selected_index"] != index):
                        raise ValueError("a collected branch changed the native common prefix")
                certificate = episode["certificate"]
                if episode["verified_closed"] and (not certificate["passed"] or certificate["axioms"]
                        or digest(certificate["source"]) != certificate["source_digest"]):
                    raise ValueError("a collected fork proof lacks an independent certificate")
                traces.append({"choice": choice, "return": proof_return(episode), "native_replay": replay, "episode": episode})
            rows[root] = {"fork": fork, "eligible_choices": allowed, "traces": traces}
    return {"checkpoint": checkpoint, "rows": rows, "memory": tree.stats()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cohort", "audit", "sessions", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--collect", action="store_true")
    args = parser.parse_args()
    audit, cases, forks, observed = training_forks(args.cohort, args.audit, args.sessions / "hybrid_rl.sqlite")
    report = {"cohort_sha256": sha256_file(args.cohort), "audit_sha256": sha256_file(args.audit), "seed": audit["seed"],
              "observed": observed}
    print(audit["seed"], observed, flush=True)
    if args.collect:
        tree = TreeBudget(2, 1)
        try:
            report["interventions"] = collect(args, tree, audit, cases, forks)
        finally:
            tree.close()
    with args.output.open("xb") as output:
        output.write(gzip.compress(canonical(report).encode("utf-8"), mtime=0))


if __name__ == "__main__":
    main()
