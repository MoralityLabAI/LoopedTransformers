"""Export a routed hybrid checkpoint and verify a native proof with the reloaded model."""

import argparse
import gzip
import io
import json
from pathlib import Path
from types import SimpleNamespace

from configs.run_ppt_pending_recovery import frozen_policy, verified
from ppt_gym.data import sha256_file
from ppt_gym.hybrid_rl.corpus import load
from ppt_gym.ram_limits import TreeBudget


def export(cohort, session, audit, tree):
    import torch

    from lcgrt import config
    from lcgrt.lean_repl import LeanRepl
    from ppt_gym.hybrid_rl.__main__ import rollout
    from ppt_gym.hybrid_rl.graph_model import GraphProofPolicy
    from ppt_gym.hybrid_rl.search_control import DirectFactContinuation
    from ppt_gym.live_lean import fingerprint
    from ppt_gym.live_search import replay_search
    from ppt_gym.ram_campaign import configure
    from ppt_gym.ram_learning import tensor_hash

    configure(tree.threads, 0)
    cohort_sha = sha256_file(cohort)
    audited = json.loads(gzip.decompress(audit.read_bytes()))
    if audited["summary"]["cohort_sha256"] != cohort_sha:
        raise ValueError("checkpoint export requires its audited development cohort")
    seed = audited["summary"]["seed"]
    arm = "hybrid_recovery_rl"
    source = audited["trials"][arm]
    expected = {"previous_cohort_sha256": cohort_sha,
                "source_session_sha256": {str(seed): {arm: source["trial"]["session_sha256"]}}}
    model, manifest, session_sha, weight_sha = frozen_policy(session, expected, seed, arm)
    if weight_sha != source["trial"]["final_weights_sha256"] or not manifest.get("pending_arrival_router"):
        raise ValueError("checkpoint export requires the audited routed hybrid weights")
    architecture = {
        "architecture": manifest["architecture"], "width": manifest["width"],
        "branch_recovery": manifest["branch_recovery"], "residual_transport": manifest["residual_fact_transport"],
        "premise_interaction": manifest["premise_interaction"], "pending_arrival_router": True,
    }
    package = {
        "schema": "ppt.routed-recovery.checkpoint.v1", "architecture": architecture,
        "state_dict": model.state_dict(), "seed": seed, "objective": manifest["objective"],
        "source_session_sha256": session_sha, "audited_evidence_sha256": sha256_file(audit),
        "cohort_sha256": cohort_sha, "weights_sha256": weight_sha,
        "search": {"recurrent_passes": 4, "tactic_checks": 8, "prefer_direct_fact": True,
                   "max_macs": manifest["max_macs"], "cpu_seconds": manifest["cpu_seconds"]},
    }
    buffer = io.BytesIO()
    torch.save(package, buffer)
    data = buffer.getvalue()
    loaded = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    reconstructed = GraphProofPolicy(**loaded["architecture"]).eval()
    reconstructed.load_state_dict(loaded["state_dict"])
    if tensor_hash(reconstructed.state_dict()) != weight_sha:
        raise ValueError("portable weights changed during serialization")
    stage = source["trial"]["stages"]["final"]
    root = min(root for root, row in stage.items() if row["certified"])
    case = next(item for item in load(cohort)["dev"] if item.root == root)
    settings = SimpleNamespace(max_macs=manifest["max_macs"], cpu_seconds=manifest["cpu_seconds"])
    with LeanRepl(timeout=15) as repl:
        if fingerprint(config.REPL_EXE, config.LEAN_EXE, repl.version) != manifest["verifier"]:
            raise ValueError("checkpoint smoke test requires the original native Lean verifier")
        episode = rollout(repl, tree, case, DirectFactContinuation(reconstructed), settings)
        episode["native_replay"] = replay_search(repl, case, episode)
    recovery, = (event for event in episode["events"] if event["kind"] == "recovery")
    tactics = [event["tactic"] for event in episode["events"] if event["kind"] == "tactic"]
    if (not verified(episode) or recovery["selected_index"] != source["decisions"]["final"][root]
            or tactics != stage[root]["tactics"] or episode["model_macs"] != stage[root]["macs"]):
        raise ValueError("reloaded checkpoint did not reproduce its audited proof decision")
    return data, {
        "seed": seed, "weights_sha256": weight_sha, "root": root,
        "certificate_sha256": episode["certificate"]["source_digest"], "native_replay": episode["native_replay"],
        "tactic_checks": episode["tactic_checks"], "model_macs": episode["model_macs"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cohort", "session", "audit", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    tree = TreeBudget(2, 2)
    try:
        data, receipt = export(args.cohort, args.session, args.audit, tree)
        with args.output.open("xb") as out:
            out.write(data)
        receipt["artifact_sha256"] = sha256_file(args.output)
        print(json.dumps(receipt, sort_keys=True))
    finally:
        tree.close()


if __name__ == "__main__":
    main()
