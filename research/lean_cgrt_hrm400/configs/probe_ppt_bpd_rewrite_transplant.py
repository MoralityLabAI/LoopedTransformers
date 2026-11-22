"""Swap disjoint learned policy components at the original rewrite inference cost."""

import argparse
from collections import Counter
import copy
import gzip
import json
from pathlib import Path
from types import SimpleNamespace

from configs.probe_ppt_bpd_rewrite_components import RewriteScoreEdit
from configs.probe_ppt_bpd_rewrite_revisits import PreferNewEquality, RewriteVisits, audited_policies
from ppt_gym.data import canonical, digest, sha256_file
from ppt_gym.hybrid_rl.corpus import load
from ppt_gym.ram_limits import TreeBudget


CONTROLS = ("hybrid_calibration", "hybrid_imitation")
PARTS = {"graph": "graph.", "transport": "graph.transport."}
MESSAGE_BANK = ("recall.", "read_gate.", "write_gate.", "write_content.")
LEXICAL_PARTS = {
    "action_head": "actor.",
    "observation_features": ("text.", "roles", "context.", "norm.", "feature."),
    "tied_recurrence_and_bank": ("cell.", *MESSAGE_BANK),
    "tied_cell": "cell.",
    "message_bank": MESSAGE_BANK,
}


def substitute(recipient, donor, prefix):
    from ppt_gym.ram_learning import tensor_hash

    recipient_weights = recipient.state_dict()
    donor_weights = donor.state_dict()
    if not any(name.startswith(prefix) for name in recipient_weights):
        raise ValueError("the recipient has no matching component")
    weights = {name: donor_weights[name] if name.startswith(prefix) else value
               for name, value in recipient_weights.items()}
    composed = copy.deepcopy(recipient).eval()
    composed.load_state_dict(weights)
    return composed, tensor_hash(weights)


def check_scores(model, recipient, donor, part, packet, menu):
    import torch

    native, _ = model(packet, menu, 4)
    original, _ = recipient(packet, menu, 4)
    removed, _ = RewriteScoreEdit(recipient, part)(packet, menu, 4)
    inserted, _ = RewriteScoreEdit(donor, part)(packet, menu, 4)
    if not torch.allclose(native, original - removed + inserted, atol=2e-6, rtol=2e-5):
        raise ValueError("the parameter substitution did not reproduce the intended score")


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
    previous = json.loads(gzip.decompress(args.revisit_audit.read_bytes()))
    cases = {case.root: case for case in load(args.cohort)["dev"]}
    if (reference["cohort_sha256"] != cohort_sha or set(cases) != set(reference["rows"])
            or previous["cohort_sha256"] != cohort_sha or previous["reference_sha256"] != reference_sha
            or previous["audit_sha256"] != audit_sha):
        raise ValueError("the component substitution requires matching audited rewrite outcomes")
    options = SimpleNamespace(cpu_seconds=6, max_macs=1_200_000_000)
    candidate = "hybrid_rl"
    controls = CONTROLS[:1] if args.lexical else CONTROLS
    parts = LEXICAL_PARTS if args.lexical else PARTS
    if args.part is not None:
        if args.part not in parts:
            raise ValueError("the requested parameter group is unavailable in this diagnostic")
        parts = {args.part: parts[args.part]}
    models = {arm: model for arm, model, source, _ in audited_policies(
        args.sessions, args.audit, cohort_sha, reference_sha
    ) if arm in (candidate, *controls) and source == previous["results"][arm]["checkpoint"]}
    if set(models) != {candidate, *controls}:
        raise ValueError("the requested checkpoints differ from the previous revisit audit")
    result = {}
    with torch.no_grad(), LeanRepl(timeout=15) as repl:
        if repl.version != reference["lean"]:
            raise ValueError("the native rewrite toolchain changed")
        first = next(iter(cases.values()))
        initial = repl.init(first.statement, pp=True)
        menu = candidate_menu(repl, initial, rewrite=True)
        packet, _ = state_input(repl, initial, (), 8)
        for control in controls:
            result[control] = {}
            for part, prefix in parts.items():
                for destination, source in ((candidate, control), (control, candidate)):
                    model, weights = substitute(models[destination], models[source], prefix)
                    if part in PARTS:
                        check_scores(model, models[destination], models[source], part, packet, menu)
                    stats = Counter(certified=0, gains=0, losses=0, checks=0)
                    rows = {}
                    for root, case in cases.items():
                        visits = RewriteVisits(repl)
                        policy = PreferNewEquality(model, visits)
                        episode = rollout(visits, tree, case, policy, options)
                        replay = replay_search(repl, case, episode)
                        cert = episode["certificate"]
                        success = episode["verified_closed"]
                        if (not episode["within_cpu_budget"] or not replay["replay_passed"]
                                or (success and (not reference["rows"][root]["certified"] or not cert["passed"]
                                                 or cert["axioms"] or digest(cert["source"]) != cert["source_digest"]))):
                            raise ValueError("the component substitution produced an invalid native outcome")
                        pristine = previous["results"][destination]["roots"][root]["prefer_new"]["certified"]
                        stats.update(certified=int(success), gains=int(success and not pristine),
                                     losses=int(pristine and not success), checks=episode["tactic_checks"])
                        rows[root] = {
                            "certified": success, "checks": episode["tactic_checks"], "script": episode["accepted_script"],
                            "certificate": cert, "native_replay": replay, "macs": episode["model_macs"],
                        }
                    label = f"{part}_from_{source}_into_{destination}"
                    result[control][label] = {
                        "composed_weights_sha256": weights, "counts": stats, "roots": rows,
                    }
                    print(label, dict(stats), flush=True)
    return {"scope": "original BPD rewrites with a shared revisit control and matched inference cost",
            "parameters_substituted": parts,
            "cohort_sha256": cohort_sha, "reference_sha256": reference_sha, "audit_sha256": audit_sha,
            "revisit_audit_sha256": sha256_file(args.revisit_audit), "results": result, "memory": tree.stats()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cohort", "reference", "sessions", "audit", "revisit_audit", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--lexical", action="store_true")
    parser.add_argument("--part", choices=(*PARTS, *LEXICAL_PARTS))
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
