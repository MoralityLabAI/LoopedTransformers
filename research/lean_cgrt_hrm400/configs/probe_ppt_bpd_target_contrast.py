"""Compare target-conditioned recurrent scores on natively verified BPD rewrites."""

import argparse
from collections import defaultdict
from dataclasses import replace
import gzip
import json
from pathlib import Path
from types import SimpleNamespace

from configs.evaluate_ppt_bpd_rewrite_transfer import prepare, read_arm
from configs.freeze_ppt_bpd_rewrite import ARMS
from configs.probe_ppt_bpd_rewrite_revisits import PreferNewEquality, RewriteVisits, audited_policies
from ppt_gym.data import canonical, digest, sha256_file
from ppt_gym.ram_limits import TreeBudget


class RewriteGraphReadout:
    def __init__(self, model, mode):
        self.graph = model.graph
        self.architecture = model.architecture
        self.mode = mode

    def macs(self, _candidates, passes):
        views = 2 if self.mode in ("contrast", "learned_contrast") else 1
        return views * self.graph.upper_macs(passes)

    def __call__(self, packet, actions, passes, **_options):
        import torch

        from ppt_gym.hybrid_rl.graph_model import visible_graph

        observed = visible_graph(packet, actions)
        if observed is None or not all(observed.actions) or sum(row[0] for row in observed.features) != 1:
            raise ValueError("the target contrast requires a public equality graph with one target")
        if self.mode == "transport":
            features = self.graph.node.weight.new_tensor(observed.features)
            premises = torch.as_tensor(observed.premises, dtype=torch.long, device=features.device)
            heads = torch.as_tensor(observed.heads, dtype=torch.long, device=features.device)
            transport = self.graph.transport(features[:, :1], premises, heads, 2 * passes)
            selected = torch.as_tensor(observed.actions, dtype=torch.long, device=features.device) - 1
            return transport[selected], transport.new_zeros(())
        if self.mode == "graph":
            return self.graph(observed, passes)
        neutral = replace(observed, features=[[0., active, pending] for _, active, pending in observed.features])
        if self.mode == "learned_contrast":
            with self.graph.transport.register_forward_hook(lambda _module, _args, scores: torch.zeros_like(scores)):
                scores, value = self.graph(observed, passes)
                baseline, _ = self.graph(neutral, passes)
        else:
            scores, value = self.graph(observed, passes)
            baseline, _ = self.graph(neutral, passes)
        return scores - baseline, value


def diagnostic_cases(cases, witness):
    by_depth = defaultdict(list)
    for case in cases.values():
        length = witness["rows"][case.root]["checks"]
        if 4 <= length <= 7:
            by_depth[length].append(case)
    if set(by_depth) != {4, 5, 6, 7} or any(len(cases) < 3 for cases in by_depth.values()):
        raise ValueError("the target-contrast diagnostic needs three layouts from each registered depth")
    return [case for depth in sorted(by_depth)
            for case in sorted(by_depth[depth], key=lambda item: digest(["target-contrast-v1", item.root]))[:3]]


def audit(args, tree):
    import torch

    from lcgrt import config
    from lcgrt.lean_repl import LeanRepl
    from ppt_gym.hybrid_rl.__main__ import rollout
    from ppt_gym.live_lean import fingerprint
    from ppt_gym.live_search import replay_search
    from ppt_gym.ram_campaign import configure

    doc, witness, candidates = prepare(args.cohort, args.witness, args.source_audit)
    cohort_sha, witness_sha = map(sha256_file, (args.cohort, args.witness))
    summary_path = args.receipts / "summary.json"
    summary = json.loads(summary_path.read_bytes())
    if summary["cohort_sha256"] != cohort_sha or summary["witness_sha256"] != witness_sha:
        raise ValueError("the baseline receipts describe different development layouts")
    cases = diagnostic_cases(candidates, witness)
    registration = doc["registration"]
    configure(tree.threads, registration["seed"])
    options = SimpleNamespace(cpu_seconds=registration["cpu_seconds"], max_macs=registration["max_macs"])
    source = json.loads(args.source_audit.read_bytes())
    arms = (args.arm,) if args.arm else ARMS
    policies = audited_policies(args.sessions, args.source_audit, doc["source_cohort_sha256"], doc["reference_sha256"],
                                selected=arms)
    results = {}
    with torch.no_grad(), LeanRepl(timeout=15) as repl:
        verifier = fingerprint(config.REPL_EXE, config.LEAN_EXE, repl.version)
        for arm, model, checkpoint, _ in policies:
            receipt = args.receipts / f"{arm}.json.gz"
            if sha256_file(receipt) != summary["arm_evidence_sha256"][arm]:
                raise ValueError("a baseline receipt has changed since the matched native readout")
            prior = read_arm(receipt, arm, doc, candidates, cohort_sha, witness_sha)["rows"]
            if verifier != source["trials"][arm]["manifest"]["verifier"]:
                raise ValueError("the native verifier differs from the original rewrite readout")
            modes = ("graph", "contrast", "learned_contrast", "transport") if model.graph.transport is not None else (
                "graph", "contrast",
            )
            rows = {}
            for case in cases:
                trials = {name: {"certified": row["certified"], "checks": row["checks"], "macs": row["macs"]}
                          for name, row in prior[case.root].items()}
                for mode in modes:
                    visits = RewriteVisits(repl)
                    policy = PreferNewEquality(RewriteGraphReadout(model, mode), visits)
                    episode = rollout(visits, tree, case, policy, options)
                    replay = replay_search(repl, case, episode)
                    certificate = episode["certificate"]
                    if (not replay["replay_passed"] or episode["tactic_checks"] > registration["tactic_checks"]
                            or episode["model_macs"] > options.max_macs
                            or (episode["verified_closed"] and (not certificate["passed"] or certificate["axioms"]
                                or digest(certificate["source"]) != certificate["source_digest"]))):
                        raise ValueError("a target-contrast trajectory failed its native proof audit")
                    trials[mode] = {"certified": episode["verified_closed"] and episode["within_cpu_budget"],
                                    "cpu_censored": not episode["within_cpu_budget"], "macs": episode["model_macs"],
                                    "checks": episode["tactic_checks"], "script": episode["accepted_script"],
                                    "certificate": certificate, "native_replay": replay}
                rows[case.root] = trials
            counts = {mode: sum(row[mode]["certified"] for row in rows.values())
                      for mode in ("prefer_new", "pristine", *modes)}
            results[arm] = {"checkpoint": checkpoint, "counts": counts, "rows": rows}
            print(arm, counts, flush=True)
    return {"scope": "exploratory target-contrast diagnosis on a depth-stratified subset of the fresh BPD readout",
            "cohort_sha256": cohort_sha, "witness_sha256": witness_sha,
            "baseline_summary_sha256": sha256_file(summary_path), "results": results, "memory": tree.stats()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cohort", "witness", "source_audit", "sessions", "receipts", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--arm", choices=ARMS)
    args = parser.parse_args()
    tree = TreeBudget(2, 1)
    try:
        report = audit(args, tree)
        with args.output.open("xb") as output:
            output.write(gzip.compress(canonical(report).encode("utf-8"), mtime=0))
    finally:
        tree.close()


if __name__ == "__main__":
    main()
