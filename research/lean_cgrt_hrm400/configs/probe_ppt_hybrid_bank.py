"""Probe value content in a frozen hybrid bank, with equal-dose control edits."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path

from ppt_gym.data import canonical, sha256_file
from ppt_gym.hybrid_rl.corpus import load
from ppt_gym.ram_limits import TreeBudget


EDITS = {"pristine": ((), False), "ablate-pass-2": ((2,), False),
         "control-pass-2": ((2,), True), "ablate-all": ((2, 3, 4), False),
         "control-all": ((2, 3, 4), True)}
TRANSPORT_EDITS = ("pristine", "ablate-transport", "control-common-shift")
INTERACTION_EDITS = ("pristine", "ablate-interaction", "control-common-shift")


class EditedPolicy:
    """Keep the attention lookup fixed and replace only value content at selected reads."""

    def __init__(self, policy):
        self.policy = policy
        self.architecture = policy.architecture
        self.macs = policy.macs
        self.mode = "pristine"

    @contextmanager
    def edit(self, mode):
        self.mode = mode
        self.hits = 0
        try:
            yield self
        finally:
            self.mode = "pristine"

    def __call__(self, *args, **kwargs):
        passes, control = EDITS[self.mode]
        pass_number = 1

        def modify(_module, inputs):
            nonlocal pass_number
            pass_number += 1
            if pass_number not in passes:
                return None
            query, keys, values = inputs
            # Subtract a channel-rotated bank with the same Frobenius norm as
            # subtracting the true values. Preserve keys, query, and shape.
            delta = values.roll(1, dims=-1) if control else values
            self.hits += 1
            return query, keys, values - delta

        if self.mode == "pristine":
            return self.policy(*args, **kwargs)
        with self.policy.recall.register_forward_pre_hook(modify):
            return self.policy(*args, **kwargs)


class EditedTransportPolicy(EditedPolicy):
    """Cut the selected logits' transport contribution or apply an equal-norm common shift."""

    def __init__(self, policy, *, interaction_only=False):
        super().__init__(policy)
        self.interaction_only = interaction_only
        self.target = policy.graph.transport.interaction_readout if interaction_only else policy.graph.transport
        self.ablation = INTERACTION_EDITS[1] if interaction_only else TRANSPORT_EDITS[1]

    def __call__(self, packet, actions, *args, **kwargs):
        from torch import long

        from ppt_gym.hybrid_rl.graph_model import visible_graph

        if self.mode == "pristine" or (self.interaction_only and json.loads(packet)["checks_remaining"] != 8):
            return self.policy(packet, actions, *args, **kwargs)
        graph = visible_graph(packet, actions)
        if graph is None or (len(set(graph.actions)) != len(actions)) or not all(graph.actions):
            raise ValueError("transport intervention requires distinct visible premise actions")

        def modify(_module, _inputs, scores):
            ids = scores.new_tensor(graph.actions, dtype=long) - 1
            chosen = scores[ids]
            if self.mode == self.ablation:
                edits = chosen.new_zeros(chosen.shape)
            else:
                edits = chosen - chosen.norm() / len(actions) ** 0.5
            self.hits += 1
            return scores.index_copy(0, ids, edits)

        with self.target.register_forward_hook(modify):
            return self.policy(packet, actions, *args, **kwargs)


def probe(args, tree):
    import torch

    from lcgrt import config
    from lcgrt.lean_repl import LeanRepl
    from ppt_gym import ram_store
    from ppt_gym.hybrid_rl.__main__ import rollout
    from ppt_gym.hybrid_rl.graph_model import GraphProofPolicy
    from ppt_gym.hybrid_rl.model import ProofPolicy
    from ppt_gym.live_lean import fingerprint
    from ppt_gym.live_search import replay_search
    from ppt_gym.ram_campaign import configure
    from ppt_gym.ram_learning import tensor_hash

    configure(args.threads, 0)
    cases = load(args.cohort)["dev"]
    cohort_sha = sha256_file(args.cohort)
    transport = args.transport or args.interaction
    with ram_store.open_store(args.session) as db:
        manifest = ram_store.read_manifest(db)
        if (manifest["cohort"] != cohort_sha or manifest["architecture"] != "hybrid"
                or bool(manifest.get("graph_context")) != transport
                or bool(manifest.get("residual_fact_transport")) != transport
                or (args.interaction and not manifest.get("premise_interaction"))):
            raise ValueError("probe type does not match the frozen hybrid checkpoint")
        recorded = {}
        last = None
        for row in ram_store.ledger(db):
            last = row
            if row["phase"] == "final":
                ep, = row["episodes"]
                recorded[ep["root"]] = ep
        if last is None or last["seq"] + 1 != manifest["entry_count"] or len(recorded) != len(cases):
            raise ValueError("hybrid training or frozen development evaluation is incomplete")
        params = ram_store.checkpoint(db)["model"]
        if last["weights_after"] != tensor_hash(params):
            raise ValueError("frozen weights differ from the proof ledger")
    model = (GraphProofPolicy("hybrid", manifest["width"], residual_transport=True,
                              premise_interaction=manifest.get("premise_interaction", False)) if transport
             else ProofPolicy("hybrid", manifest["width"]))
    model.load_state_dict(params)
    edit = EditedTransportPolicy(model, interaction_only=args.interaction) if transport else EditedPolicy(model)
    modes = INTERACTION_EDITS if args.interaction else TRANSPORT_EDITS if transport else tuple(EDITS)
    args.cpu_seconds, args.max_macs = manifest["cpu_seconds"], manifest["max_macs"]
    methods = {"primary": "all-read value removal versus equal-norm channel-subtraction control",
               "secondary": "pass-two value removal versus the same control",
               "scientific_minimum": 0.05, "scope": "reused development roots, diagnostic only"}
    if transport:
        methods = {"primary": "transport removal versus equal-norm common shift of all root action logits",
                   "scientific_minimum": 0.05, "scope": "reused development roots, diagnostic only"}
    if args.interaction:
        methods["primary"] = "remove learned premise interactions at the first decision versus equal-norm common shift"
    result = {"cohort": cohort_sha, "session_sha256": sha256_file(args.session),
              "weights_sha256": last["weights_after"], "model_seed": manifest["seed"],
              "registration": methods, "roots": {}}

    with LeanRepl(timeout=15) as repl:
        if fingerprint(config.REPL_EXE, config.LEAN_EXE, repl.version) != manifest["verifier"]:
            raise ValueError("the original native proof verifier changed")
        for case in cases:
            info = {"family": case.family, "cluster": case.cluster, "arms": {}, "logit_diagnostics": {}}
            result["roots"][case.root] = info
            for mode in modes:
                with edit.edit(mode):
                    ep = rollout(repl, tree, case, edit, args)
                    hits = edit.hits
                replay = replay_search(repl, case, ep)
                info["arms"][mode] = {
                    "certified": ep["verified_closed"] and ep["within_cpu_budget"],
                    "path": [e["tactic"] for e in ep["events"] if e["kind"] == "tactic"],
                    "checks": ep["tactic_checks"], "bank_reads_edited": hits,
                    "standalone_certificate": ep["certificate"], "native_replay": replay,
                }
            pristine = info["arms"]["pristine"]
            original = recorded[case.root]
            path = [e["tactic"] for e in original["events"] if e["kind"] == "tactic"]
            original_certified = original["verified_closed"] and original["within_cpu_budget"]
            if pristine["certified"] != original_certified or pristine["path"] != path:
                raise ValueError("unedited policy failed to reproduce the recorded proof search")
            for mode in modes[1:]:
                max_change, changes, decisions = 0.0, 0, 0
                for event in original["events"]:
                    if event["kind"] != "tactic" or len(event["candidate_menu"]) == 1:
                        continue
                    packet, menu = canonical(event["input"]), event["candidate_menu"]
                    with torch.no_grad():
                        a, _ = edit(packet, menu, 4)
                        with edit.edit(mode):
                            b, _ = edit(packet, menu, 4)
                    max_change = max(max_change, float((a - b).abs().max()))
                    changes += int(a.argmax() != b.argmax())
                    decisions += 1
                info["logit_diagnostics"][mode] = {
                    "max_absolute_change": max_change, "argmax_changes": changes, "decisions": decisions}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cohort", type=Path)
    parser.add_argument("session", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--ram", type=int, choices=(2, 4, 6), default=2)
    parser.add_argument("--threads", type=int, default=2)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--transport", action="store_true", help="probe residual fact transport instead of the lexical bank")
    mode.add_argument("--interaction", action="store_true", help="ablate only the learned interaction at the first decision")
    args = parser.parse_args()
    tree = TreeBudget(args.ram, args.threads)
    try:
        result = probe(args, tree)
        result["memory"] = tree.stats()
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    finally:
        tree.close()


if __name__ == "__main__":
    main()
