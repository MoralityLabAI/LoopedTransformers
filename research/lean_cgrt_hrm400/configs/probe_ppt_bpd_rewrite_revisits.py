"""Measure a shared rewrite revisit control on the matched warm checkpoints."""

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
from types import SimpleNamespace

from configs.freeze_ppt_bpd_rewrite import ARMS, CELL_ARMS, rewrites
from ppt_gym.data import canonical, digest, sha256_file
from ppt_gym.hybrid_rl.corpus import load
from ppt_gym.ram_limits import TreeBudget
from ppt_gym.rewrite import (
    BPD_REWRITE_FAMILY, remember_rewrite_target, restrict_rewrite_scores, unvisited_rewrites,
)


class RewriteVisits:
    """Track equality targets actually reached by the current Lean search."""

    def __init__(self, repl):
        self.repl = repl
        self.targets = set()

    def __getattr__(self, name):
        return getattr(self.repl, name)

    def reset(self):
        self.targets.clear()
        return self.repl.reset()

    def remember(self, state):
        from ppt_gym.live_lean import public_state

        remember_rewrite_target(public_state(self.repl, state), self.targets)

    def init(self, *args, **kwargs):
        state = self.repl.init(*args, **kwargs)
        self.remember(state)
        return state

    def run(self, sid, tactic):
        reply = self.repl.run(sid, tactic)
        if reply["ok"]:
            self.remember(reply["state"])
        return reply


class PreferNewEquality:
    def __init__(self, policy, visits):
        self.policy = policy
        self.visits = visits
        self.interventions = 0

    def __getattr__(self, name):
        return getattr(self.policy, name)

    def __call__(self, packet, menu, passes, **kwargs):
        goal = json.loads(packet)["state"]["goals"][0]
        logits, value = self.policy(packet, menu, passes, **kwargs)
        choices = unvisited_rewrites(goal, menu, self.visits.targets)
        if choices is not None:
            self.interventions += int(int(logits.argmax()) not in choices)
        return restrict_rewrite_scores(logits, choices), value


def revisit_count(episode):
    seen = set()
    count = 0
    for event in episode["events"]:
        if event["kind"] == "tactic":
            seen.add(event["state_before_sha256"])
            if event["status"] == "accepted_nonterminal" and event["state_after_sha256"] in seen:
                count += 1
    return count


def warm_policies(initialization, reg, seed=151):
    from ppt_gym.hybrid_rl.__main__ import initialize_from_checkpoint
    from ppt_gym.hybrid_rl.graph_model import GraphProofPolicy

    for architecture, warm in (("hybrid", "hybrid_imitation"), ("trm", "trm")):
        hybrid = architecture == "hybrid"
        model = GraphProofPolicy(architecture, residual_transport=hybrid, premise_interaction=hybrid).eval()
        options = SimpleNamespace(
            initialize_from=initialization / f"{seed}-{warm}.sqlite",
            architecture=architecture, premise_interaction=hybrid, width=256, seed=seed,
            root_fact_adapter=False, branch_recovery=False,
        )
        source = initialize_from_checkpoint(model, options, reg["initialization_cohort_sha256"])
        yield architecture, model, source, None


def audited_policies(sessions, audit_path, cohort_sha, reference_sha):
    from ppt_gym import ram_store
    from ppt_gym.hybrid_rl.graph_model import GraphProofPolicy
    from ppt_gym.ram_learning import tensor_hash

    audit = json.loads(audit_path.read_bytes())
    arms = CELL_ARMS if audit.get("cell_study_sha256") else ARMS
    if (audit["cohort_sha256"] != cohort_sha or audit["reference_sha256"] != reference_sha
            or set(audit["trials"]) != set(arms)):
        raise ValueError("the final rewrite policies require the complete matched audit")
    for arm in arms:
        trial = audit["trials"][arm]
        path = sessions / f"{arm}.sqlite"
        if sha256_file(path) != trial["session_sha256"]:
            raise ValueError("the audited rewrite session bytes changed")
        with ram_store.open_store(path) as db:
            manifest = ram_store.read_manifest(db)
            weights = ram_store.checkpoint(db)["model"]
        if manifest != trial["manifest"] or tensor_hash(weights) != trial["final_weights_sha256"]:
            raise ValueError("the final rewrite checkpoint differs from the audit")
        model = GraphProofPolicy(
            manifest["architecture"], manifest["width"], residual_transport=manifest["residual_fact_transport"],
            premise_interaction=manifest["premise_interaction"],
        ).eval()
        model.load_state_dict(weights)
        source = {"session_sha256": trial["session_sha256"], "weights_sha256": trial["final_weights_sha256"]}
        yield arm, model, source, trial["stages"]["final"]


def audit(args, tree):
    import torch

    from lcgrt.lean_repl import LeanRepl
    from ppt_gym.hybrid_rl.__main__ import rollout
    from ppt_gym.live_search import replay_search
    from ppt_gym.ram_campaign import configure

    configure(tree.threads, 151)
    cohort_sha = sha256_file(args.cohort)
    reg = json.loads(args.cohort.read_bytes())["registration"]
    data = load(args.cohort)
    cases = {case.root: case for case in data["dev"]}
    reference = json.loads(args.reference.read_bytes())
    reference_sha = sha256_file(args.reference)
    if reference["cohort_sha256"] != cohort_sha or set(reference["rows"]) != set(cases):
        raise ValueError("the reference and frozen rewrite development roots differ")
    valid = {root: row["certified"] for root, row in reference["rows"].items()}
    options = SimpleNamespace(cpu_seconds=reg["cpu_seconds"], max_macs=1_200_000_000)
    policies = (audited_policies(args.sessions, args.audit, cohort_sha, reference_sha) if args.sessions
                else warm_policies(args.initialization, reg, args.warm_seed))
    if args.sample_training:
        cases = {case.root: case for case in data["train"] if case.family == BPD_REWRITE_FAMILY}
        expected = {case.root: (case, label) for case, label in rewrites() if case.split == "train"}
        if len(cases) != 64 or cases != {root: case for root, (case, _) in expected.items()}:
            raise ValueError("sampled exploration requires all original rewrite training declarations")
        valid = {root: label for root, (_, label) in expected.items()}
        policies = [next(policies)]
    results = {}
    with LeanRepl(timeout=15) as repl:
        if repl.version != reference["lean"]:
            raise ValueError("the native rewrite toolchain changed")
        for arm, model, source, baseline in policies:
            visits = RewriteVisits(repl)
            novel = PreferNewEquality(model, visits)
            counts = {"pristine": Counter(), "prefer_new": Counter()}
            rows = {}
            for root, case in cases.items():
                rows[root] = {}
                for name, verifier, policy in (("pristine", repl, model), ("prefer_new", visits, novel)):
                    before = novel.interventions
                    seed = int(digest(["bpd-rewrite-exploration", root])[:16], 16)
                    rng = torch.Generator().manual_seed(seed) if args.sample_training else None
                    episode = rollout(verifier, tree, case, policy, options, rng=rng)
                    replay = replay_search(repl, case, episode)
                    cert = episode["certificate"]
                    certified = episode["verified_closed"] and episode["within_cpu_budget"]
                    if (not episode["within_cpu_budget"] or not replay["replay_passed"]
                            or (certified and (not valid[root] or not cert["passed"] or cert["axioms"]
                                               or digest(cert["source"]) != cert["source_digest"]))):
                        raise ValueError("rewrite intervention lacked a replayed in-budget outcome")
                    if baseline is not None and name == "pristine":
                        tactics = [event["tactic"] for event in episode["events"] if event["kind"] == "tactic"]
                        if tactics != baseline[root]["tactics"] or certified != baseline[root]["certified"]:
                            raise ValueError("the pristine replay differs from the audited final search")
                    revisits = revisit_count(episode)
                    interventions = novel.interventions - before
                    rows[root][name] = {"certified": certified, "checks": episode["tactic_checks"],
                                        "revisits": revisits, "overrides": interventions,
                                        "script": episode["accepted_script"], "certificate": cert,
                                        "macs": episode["model_macs"], "replay": replay}
                    counts[name].update(certified=certified, revisits=revisits, overrides=interventions,
                                        tactic_checks=episode["tactic_checks"],
                                        valid_world_revisits=revisits if valid[root] else 0,
                                        valid_worlds_revisited=int(valid[root] and revisits > 0))
            key = "checkpoint" if args.audit else "initialization"
            results[arm] = {key: source, "counts": counts, "roots": rows}
            print(arm, {name: dict(values) for name, values in counts.items()}, flush=True)
    scope = "audited final checkpoints" if args.audit else "frozen warm checkpoints"
    report = {"scope": f"original BPD rewrite development with {scope}",
              "cohort_sha256": cohort_sha, "reference_sha256": reference_sha,
              "results": results, "memory": tree.stats()}
    if args.audit:
        report["audit_sha256"] = sha256_file(args.audit)
    elif args.warm_seed != 151:
        report["warm_seed"] = args.warm_seed
    if args.sample_training:
        report.update(scope="paired sampled exploration on original BPD rewrite training declarations",
                      sampling="same per-root random seed in both modes",
                      rank_overrides="number of masks that move the highest-scoring action")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cohort", "reference", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--initialization", type=Path)
    parser.add_argument("--warm-seed", type=int, choices=(103, 151, 197), default=151)
    parser.add_argument("--sessions", type=Path)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--sample-training", action="store_true")
    args = parser.parse_args()
    if args.initialization and (args.sessions or args.audit):
        parser.error("choose either warm initialization or audited final sessions")
    if not args.initialization and (not args.sessions or not args.audit):
        parser.error("provide warm initialization or both final sessions and their audit")
    if args.sample_training and not args.initialization:
        parser.error("sampled training exploration uses the frozen hybrid warm initialization")
    if args.warm_seed != 151 and (not args.initialization or args.sample_training):
        parser.error("additional warm seeds use the frozen rewrite development comparison")
    tree = TreeBudget(2, 1)
    try:
        report = audit(args, tree)
        with args.output.open("xb") as output:
            output.write(gzip.compress(canonical(report).encode("utf-8"), mtime=0))
    finally:
        tree.close()


if __name__ == "__main__":
    main()
