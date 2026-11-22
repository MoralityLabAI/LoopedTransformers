"""Export audited CPU rewrite weights and verify both certified inference modes."""

import argparse
import gzip
import io
import json
from pathlib import Path
from types import SimpleNamespace

from configs.probe_ppt_bpd_rewrite_revisits import PreferNewEquality, RewriteVisits, audited_policies
from ppt_gym.data import digest, sha256_file
from ppt_gym.hybrid_rl.corpus import load
from ppt_gym.ram_limits import TreeBudget


def export(args, tree):
    import torch

    from lcgrt import config
    from lcgrt.lean_repl import LeanRepl
    from ppt_gym.hybrid_rl.__main__ import rollout
    from ppt_gym.hybrid_rl.graph_model import GraphProofPolicy
    from ppt_gym.live_lean import fingerprint
    from ppt_gym.live_search import replay_search
    from ppt_gym.ram_campaign import configure
    from ppt_gym.ram_learning import tensor_hash

    configure(tree.threads, 0)
    cohort_sha, reference_sha, audit_sha = map(sha256_file, (args.cohort, args.reference, args.audit))
    audit = json.loads(args.audit.read_bytes())
    revisit = json.loads(gzip.decompress(args.revisit_audit.read_bytes()))
    arm, model, checkpoint, stage = next(audited_policies(args.sessions, args.audit, cohort_sha, reference_sha))
    if arm != "hybrid_rl" or revisit["audit_sha256"] != audit_sha or revisit["reference_sha256"] != reference_sha:
        raise ValueError("the rewrite export needs the matched audited RL checkpoint")
    revisit_trial = revisit["results"][arm]
    if revisit_trial["checkpoint"] != checkpoint:
        raise ValueError("the rewrite export differs from the native revisit checkpoint")
    manifest = audit["trials"][arm]["manifest"]
    settings = SimpleNamespace(max_macs=manifest["max_macs"], cpu_seconds=manifest["cpu_seconds"])
    architecture = {
        "architecture": manifest["architecture"], "width": manifest["width"],
        "residual_transport": manifest["residual_fact_transport"], "premise_interaction": manifest["premise_interaction"],
    }
    package = {
        "schema": "ppt.bpd-rewrite.checkpoint.v1", "architecture": architecture, "state_dict": model.state_dict(),
        "seed": manifest["seed"], "objective": manifest["objective"], "weights_sha256": checkpoint["weights_sha256"],
        "source_session_sha256": checkpoint["session_sha256"], "cohort_sha256": cohort_sha,
        "audit_sha256": audit_sha, "revisit_audit_sha256": sha256_file(args.revisit_audit),
        "search": {"recurrent_passes": 4, "tactic_checks": 8, "cpu_seconds": settings.cpu_seconds,
                   "max_macs": settings.max_macs, "revisit_modes": ["pristine", "prefer_new"]},
    }
    buffer = io.BytesIO()
    torch.save(package, buffer)
    data = buffer.getvalue()
    restored = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    reloaded = GraphProofPolicy(**restored["architecture"]).eval()
    reloaded.load_state_dict(restored["state_dict"])
    if tensor_hash(reloaded.state_dict()) != checkpoint["weights_sha256"]:
        raise ValueError("the portable rewrite weights changed during serialization")
    cases = {case.root: case for case in load(args.cohort)["dev"]}
    records = revisit_trial["roots"]
    roots = {
        "pristine": min(root for root, row in stage.items() if row["certified"]),
        "prefer_new": min(root for root, row in records.items()
                          if row["prefer_new"]["certified"] and not row["pristine"]["certified"]),
    }
    proofs = {}
    with LeanRepl(timeout=15) as repl:
        if fingerprint(config.REPL_EXE, config.LEAN_EXE, repl.version) != manifest["verifier"]:
            raise ValueError("checkpoint replay requires the original native verifier")
        for mode, root in roots.items():
            visits = RewriteVisits(repl)
            verifier, policy = (repl, reloaded) if mode == "pristine" else (visits, PreferNewEquality(reloaded, visits))
            episode = rollout(verifier, tree, cases[root], policy, settings)
            replay = replay_search(repl, cases[root], episode)
            original = records[root][mode]
            cert = episode["certificate"]
            if (not episode["verified_closed"] or not episode["within_cpu_budget"] or not replay["replay_passed"]
                    or episode["accepted_script"] != original["script"] or episode["tactic_checks"] != original["checks"]
                    or episode["model_macs"] != original["macs"] or digest(cert["source"]) != original["certificate"]["source_digest"]):
                raise ValueError("the checkpoint did not reproduce its audited native proof")
            proofs[mode] = {"root": root, "certificate_sha256": cert["source_digest"], "native_replay": replay,
                            "tactic_checks": episode["tactic_checks"], "model_macs": episode["model_macs"]}
    return data, {"seed": manifest["seed"], "weights_sha256": checkpoint["weights_sha256"],
                  "source_session_sha256": checkpoint["session_sha256"], "proofs": proofs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cohort", "reference", "sessions", "audit", "revisit_audit", "output", "receipt"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    tree = TreeBudget(2, 1)
    try:
        weights, receipt = export(args, tree)
        with args.output.open("xb") as output:
            output.write(weights)
        receipt["artifact_sha256"] = sha256_file(args.output)
        with args.receipt.open("x", encoding="utf-8", newline="\n") as output:
            output.write(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        print(json.dumps(receipt, sort_keys=True))
    finally:
        tree.close()


if __name__ == "__main__":
    main()
