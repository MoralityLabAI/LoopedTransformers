"""Decompose arrival weights by channel and hop, then swap matched world contributions."""

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path

from configs.analyze_ppt_branch_recovery import verify_witness
from configs.run_ppt_pending_recovery import frozen_policy
from ppt_gym.data import canonical, sha256_file
from ppt_gym.hybrid_rl.corpus import load
from ppt_gym.ram_limits import TreeBudget


PASSES = 4
HOP_WINDOWS = {"early": slice(0, 4), "late": slice(4, 8), "all": slice(0, 8)}
ARMS = ("hybrid_recovery_rl", "hybrid_calibration", "hybrid_imitation")


def arrival_trace(model, observation):
    import torch
    from torch.func import functional_call

    from ppt_gym.hybrid_rl.graph_model import visible_graph

    transport = model.recovery_head.transport
    context, actual = [], []

    def capture_input(_module, args, kwargs):
        context.append((args, kwargs))

    def capture_output(_module, _args, output):
        actual.append(output)

    with (transport.register_forward_pre_hook(capture_input, with_kwargs=True),
          transport.arrival_readout.register_forward_hook(capture_output)):
        logits, _ = model.recover(canonical(observation), PASSES)
    (args, kwargs), = context
    clean, = actual
    features, premises, heads, total = args
    if total != 2 * PASSES:
        raise ValueError("unexpected recurrent arrival horizon")
    graph = visible_graph(canonical(observation["parent"]), observation["parent_menu"])
    if not all(graph.actions):
        raise ValueError("the arrival probe requires two public Horn rule actions")
    indices = torch.tensor(graph.actions, device=logits.device) - 1
    gain = kwargs["arrival_gain"]
    gain = logits.new_tensor(1.0) if gain is None else gain
    channels = transport.arrival
    masks = torch.eye(channels.numel(), device=channels.device, dtype=channels.dtype)
    accumulated = [logits.new_zeros((len(channels), len(indices)))]
    with transport.arrival_readout.register_forward_hook(capture_output):
        for hop in range(1, total + 1):
            partials = []
            for mask in masks:
                actual.clear()
                functional_call(transport, {"arrival": channels * mask}, (features, premises, heads, hop))
                projected, = actual
                partials.append(projected[indices] * (hop / total))
            accumulated.append(torch.stack(partials))
    parts = torch.diff(torch.stack(accumulated), dim=0)
    if not torch.allclose(parts.sum(dim=(0, 1)) * gain, clean[indices], atol=2e-6, rtol=2e-5):
        raise ValueError("the channel-by-hop decomposition failed to reconstruct the native arrival term")
    return logits, gain, parts


def donor_edits(logits, gain, own, donor, explored):
    import torch

    edits = {}
    for name, window in HOP_WINDOWS.items():
        delta = (donor[window] - own[window]).sum(dim=(0, 1)) * gain
        margin = delta[explored] - delta[1 - explored]
        patch = torch.stack((margin / 2, -margin / 2))
        # The equal-norm common shift leaves the decision margin invariant.
        sham = torch.full_like(patch, float(margin.abs()) / 2)
        edited = logits + patch
        sham_logits = logits + sham
        if (not torch.isfinite(edited).all()
                or not torch.allclose(sham_logits[0] - sham_logits[1], logits[0] - logits[1],
                                      atol=2e-6, rtol=2e-5)):
            raise ValueError("nonfinite donor edit or dose-matched control changed the decision margin")
        edits[name] = {"choice": int(edited.argmax()), "margin": float(edited[0] - edited[1]),
                       "sham_choice": int(sham_logits.argmax()), "sham_magnitude": float(sham.norm())}
    return edits


def evaluate(args, tree):
    import torch

    from ppt_gym.ram_campaign import configure

    configure(tree.threads, 0)
    cohort_sha = sha256_file(args.cohort)
    registration = json.loads(args.cohort.read_bytes())["registration"]
    audited = json.loads(gzip.decompress(args.audit.read_bytes()))
    summary, trials = audited["summary"], audited["trials"]
    seed = summary["seed"]
    if (summary["cohort_sha256"] != cohort_sha or summary["confirmation_scored"]
            or not registration.get("pending_arrival_ablation") or not set(ARMS).issubset(trials)):
        raise ValueError("expected a fully audited routed development cohort")
    cases = {case.root: case for case in load(args.cohort)["dev"]}
    witness_file = args.sessions / "native-witness.json"
    reference, _, _ = verify_witness(witness_file, cases, cohort_sha, registration)
    if reference != audited["witness"]:
        raise ValueError("the matched donor worlds differ from the audited native reference")
    pin = {
        "previous_cohort_sha256": cohort_sha,
        "source_session_sha256": {
            str(seed): {arm: trials[arm]["trial"]["session_sha256"] for arm in ARMS},
        },
    }
    result = {}
    with torch.no_grad():
        for arm in ARMS:
            model, manifest, _, weights = frozen_policy(args.sessions / f"{arm}.sqlite", pin, seed, arm)
            if not manifest.get("pending_arrival_router") or weights != trials[arm]["trial"]["final_weights_sha256"]:
                raise ValueError("the donor edit requires the audited routed weights")
            rows = {}
            totals = {name: Counter(clean_correct=0, donor_correct=0, choice_flips=0,
                                    moves_toward_donor=0, sham_choice_flips=0) for name in HOP_WINDOWS}
            for pair in reference["pairs"].values():
                observations = [world["outcomes"][0]["observation"] for world in pair]
                if (observations[0]["parent_menu"] != observations[1]["parent_menu"]
                        or observations[0]["explored"] != observations[1]["explored"]):
                    raise ValueError("donor and recipient must share the route and action menu")
                traces = [arrival_trace(model, observation) for observation in observations]
                for i, world in enumerate(pair):
                    logits, gain, parts = traces[i]
                    root, winner = world["root"], world["winner"]
                    clean = int(logits.argmax())
                    decision = trials[arm]["decisions"]["final"][root]
                    if decision is not None and decision != clean:
                        raise ValueError("reconstructed decision differs from the audited rollout")
                    edits = donor_edits(logits, gain, parts, traces[1 - i][2], observations[i]["explored"])
                    sign = 1 if winner == 0 else -1
                    clean_margin = float(logits[0] - logits[1])
                    for name, edit in edits.items():
                        totals[name].update(clean_correct=int(clean == winner),
                                            donor_correct=int(edit["choice"] == winner),
                                            choice_flips=int(edit["choice"] != clean),
                                            moves_toward_donor=int(sign * (edit["margin"] - clean_margin) < -1e-6),
                                            sham_choice_flips=int(edit["sham_choice"] != clean))
                    rows[root] = {"donor": pair[1 - i]["root"], "winner": winner, "clean_choice": clean,
                                  "clean_margin": clean_margin, "recipient_gain": float(gain),
                                  "arrival_parts_before_gate": parts.tolist(), "edits": edits}
            result[arm] = {"weights_sha256": weights, "worlds": rows, "windows": totals}
            print(arm, {name: dict(values) for name, values in totals.items()}, flush=True)
    return {"schema": "ppt.recovery-arrival-donor-development.v1", "seed": seed,
            "cohort_sha256": cohort_sha, "audit_sha256": sha256_file(args.audit),
            "witness_sha256": sha256_file(witness_file), "scorer_sha256": sha256_file(Path(__file__)),
            "memory": tree.stats(), "arms": result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cohort", "sessions", "audit", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    tree = TreeBudget(2, 1)
    try:
        report = evaluate(args, tree)
        with args.output.open("xb") as handle:
            handle.write(gzip.compress(canonical(report).encode("utf-8"), mtime=0))
    finally:
        tree.close()


if __name__ == "__main__":
    main()
