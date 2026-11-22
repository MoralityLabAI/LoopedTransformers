"""Intervene on the lexical, graph, and fact-transport scores during BPD rewrites."""

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
from types import SimpleNamespace

from configs.probe_ppt_bpd_rewrite_revisits import PreferNewEquality, RewriteVisits, audited_policies
from ppt_gym.data import canonical, digest, sha256_file
from ppt_gym.hybrid_rl.corpus import load
from ppt_gym.ram_limits import TreeBudget


class RewriteScoreEdit:
    def __init__(self, model, component):
        self.model = model
        self.component = component

    def __getattr__(self, name):
        return getattr(self.model, name)

    def __call__(self, packet, actions, passes, **kwargs):
        import torch

        from ppt_gym.hybrid_rl.graph_model import visible_graph

        if self.component == "pristine":
            return self.model(packet, actions, passes, **kwargs)
        outputs = []
        if self.component == "graph":
            module = self.model.graph
        else:
            module = self.model.graph.transport

        def capture(_module, _args, output):
            outputs.append(output)

        with module.register_forward_hook(capture):
            full, value = self.model(packet, actions, passes, **kwargs)
        observed, = outputs
        if self.component == "graph":
            return observed[0], value
        graph = visible_graph(packet, actions)
        rows = torch.tensor(graph.actions, device=full.device) - 1
        return observed[rows], value


def hop_components(model, packet, actions):
    """Exactly split the typed linear recurrence into native channel and hop terms."""
    import torch

    from ppt_gym.hybrid_rl.graph_model import visible_graph

    transport = model.graph.transport
    graph = visible_graph(packet, actions)
    weight = transport.action.weight[0] * transport.source.weight[:, 0]
    retention = transport.retention.sigmoid()
    features = weight.new_tensor(graph.features)
    index = torch.tensor(graph.premises, device=weight.device)[:, 0] - 1
    heads = torch.tensor(graph.heads, device=weight.device)
    choices = torch.tensor(graph.actions, device=weight.device) - 1
    if not all(graph.actions) or any(sum(i != 0 for i in row) != 1 for row in graph.premises):
        raise ValueError("the exact basis requires single-premise rewrite edges")
    mass = features[:, 0]
    degree = torch.bincount(heads, minlength=len(mass)).clamp_min(1)
    parts = []
    for hop in range(9):
        selected = mass[index[choices]]
        parts.append(8 * (weight * retention.pow(hop))[:, None] * selected)
        mass = torch.zeros_like(mass).index_add(0, heads, mass[index]) / degree
    parts = torch.stack(parts)
    measured, _ = RewriteScoreEdit(model, "transport")(packet, actions, 4)
    if not torch.allclose(parts.sum(dim=(0, 1)), measured, atol=2e-6, rtol=2e-5):
        raise ValueError("the exact transport basis did not reconstruct the native score")
    return parts.tolist()


def audit(args, tree):
    import torch

    from lcgrt.lean_repl import LeanRepl
    from ppt_gym.hybrid_rl.__main__ import rollout
    from ppt_gym.live_lean import candidate_menu, state_input
    from ppt_gym.live_search import replay_search
    from ppt_gym.ram_campaign import configure

    configure(tree.threads, 151)
    cohort_sha = sha256_file(args.cohort)
    reference_sha = sha256_file(args.reference)
    audit_sha = sha256_file(args.audit)
    reference = json.loads(args.reference.read_bytes())
    visit_control = json.loads(gzip.decompress(args.revisit_audit.read_bytes()))
    cases = {case.root: case for case in load(args.cohort)["dev"]}
    if (reference["cohort_sha256"] != cohort_sha or set(cases) != set(reference["rows"])
            or visit_control["cohort_sha256"] != cohort_sha or visit_control["reference_sha256"] != reference_sha
            or visit_control["audit_sha256"] != audit_sha):
        raise ValueError("the component intervention requires the matching native rewrite and revisit audits")
    options = SimpleNamespace(cpu_seconds=6, max_macs=1_200_000_000)
    results = {}
    with torch.no_grad(), LeanRepl(timeout=15) as repl:
        if repl.version != reference["lean"]:
            raise ValueError("the native rewrite toolchain changed")
        policies = audited_policies(args.sessions, args.audit, cohort_sha, reference_sha)
        for arm, model, source, _ in policies:
            if source != visit_control["results"][arm]["checkpoint"]:
                raise ValueError("the component intervention must use the audited revisit checkpoint")
            modes = ("pristine", "graph", "transport") if model.graph.transport is not None else ("pristine", "graph")
            counts = {mode: Counter(certified=0, checks=0) for mode in modes}
            rows = {}
            for root, case in cases.items():
                world = {"modes": {}}
                rows[root] = world
                if model.graph.transport is not None:
                    repl.reset()
                    initial = repl.init(case.statement, pp=True)
                    actions = candidate_menu(repl, initial, rewrite=True)
                    packet, _ = state_input(repl, initial, (), 8)
                    world["root_actions"] = actions
                    world["transport_parts"] = hop_components(model, packet, actions)
                for mode in modes:
                    visits = RewriteVisits(repl)
                    policy = PreferNewEquality(RewriteScoreEdit(model, mode), visits)
                    episode = rollout(visits, tree, case, policy, options)
                    replay = replay_search(repl, case, episode)
                    cert = episode["certificate"]
                    success = episode["verified_closed"]
                    if (not episode["within_cpu_budget"] or not replay["replay_passed"]
                            or (success and (not reference["rows"][root]["certified"] or not cert["passed"]
                                             or cert["axioms"] or digest(cert["source"]) != cert["source_digest"]))):
                        raise ValueError("a component edit produced an invalid native outcome")
                    pristine = visit_control["results"][arm]["roots"][root]["prefer_new"]
                    if mode == "pristine" and (success != pristine["certified"]
                                               or episode["accepted_script"] != pristine["script"]):
                        raise ValueError("pristine component replay changed the audited rewrite trajectory")
                    counts[mode].update(certified=int(success), checks=episode["tactic_checks"])
                    world["modes"][mode] = {
                        "certified": success, "checks": episode["tactic_checks"], "script": episode["accepted_script"],
                        "certificate": cert, "native_replay": replay, "macs": episode["model_macs"],
                    }
            results[arm] = {"checkpoint": source, "counts": counts, "roots": rows}
            print(arm, {mode: dict(row) for mode, row in counts.items()}, flush=True)
    return {"scope": "original BPD rewrite development with shared revisit control",
            "cohort_sha256": cohort_sha, "reference_sha256": reference_sha,
            "audit_sha256": audit_sha, "revisit_audit_sha256": sha256_file(args.revisit_audit),
            "results": results, "memory": tree.stats()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cohort", "reference", "sessions", "audit", "revisit_audit", "output"):
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
