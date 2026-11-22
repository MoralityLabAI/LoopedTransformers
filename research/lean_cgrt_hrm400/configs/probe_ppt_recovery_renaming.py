"""Test whether frozen recovery decisions survive consistent Lean atom renaming."""

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import random
import re

from configs.freeze_ppt_pending_recovery import ARMS, SEEDS
from configs.probe_ppt_recovery_components import observed_components, single_goal_witness
from configs.recheck_ppt_pending_certificates import check_batch
from configs.run_ppt_pending_recovery import frozen_policy
from ppt_gym.data import canonical, digest, sha256_file
from ppt_gym.ram_limits import TreeBudget


ATOM = re.compile(r"\bP[0-9]+\b")
EVIDENCE = {103: "branch-recovery-evidence-103.json.gz", 151: "branch-recovery-evidence.json.gz",
            197: "branch-recovery-evidence-197.json.gz"}


def rename(value, mapping):
    if isinstance(value, str):
        return ATOM.sub(lambda token: mapping[token[0]], value)
    if isinstance(value, list):
        return [rename(item, mapping) for item in value]
    if isinstance(value, dict):
        return {key: rename(item, mapping) for key, item in value.items()}
    return value


def proposition_binders(observation):
    hypotheses = observation["parent"]["state"]["goals"][0]["hypotheses"]
    atoms = [row["name"] for row in hypotheses if row["type"] == {"sort": "Prop"}]
    if len(atoms) < 2 or len(set(atoms)) != len(atoms) or not all(ATOM.fullmatch(name) for name in atoms):
        raise ValueError("expected a public set of distinct proposition atoms")
    return atoms


def canonicalize_atoms(observation):
    atoms = proposition_binders(observation)
    stable = sorted(f"P{i}" for i in range(len(atoms)))
    return rename(observation, dict(zip(atoms, stable, strict=True)))


def aliases(cluster, observation):
    atoms = sorted(proposition_binders(observation))
    permutation = atoms.copy()
    rng = random.Random(int(digest(["recovery-renaming-v1", cluster]), 16))
    rng.shuffle(permutation)
    if permutation == atoms:
        raise ValueError("the registered diagnostic requires a nontrivial renaming")
    return dict(zip(atoms, permutation, strict=True))


def certified_renamings(cohort, witness, previous_sha):
    rows = single_goal_witness(cohort, witness, previous_sha)
    old = json.loads(witness.read_bytes())["pairs"]
    groups, renamed, certificates = {}, {}, []
    for root, (cluster, observation, winner) in rows.items():
        mapping = aliases(cluster, observation)
        if mapping != groups.setdefault(cluster, mapping):
            raise ValueError("paired Lean worlds must share the renaming")
        updated = rename(observation, mapping)
        reverse = {new: original for original, new in mapping.items()}
        if rename(updated, reverse) != observation or updated["parent_menu"] != observation["parent_menu"]:
            raise ValueError("renaming changed the recovery protocol")
        if canonicalize_atoms(observation) != canonicalize_atoms(updated):
            raise ValueError("binder-order canonicalization changed under a pure renaming")
        entry, = (record for record in old[cluster] if record["root"] == root)
        certificate = entry["outcomes"][winner]["certificate"]
        name = certificate["source"].splitlines()[0].split()[1]
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise ValueError("expected a named original witness theorem")
        certificates.append((name, rename(certificate["source"], mapping)))
        renamed[root] = (cluster, updated, winner)
    for i in range(0, len(certificates), 24):
        check_batch(certificates[i:i + 24], i // 24)
    return renamed, groups


def measure(rows):
    before, after = Counter(), Counter()
    for row in rows.values():
        before[row["cluster"]] += row["original"] == row["winner"]
        after[row["cluster"]] += row["renamed"] == row["winner"]
    return {
        "original_correct_worlds": sum(before.values()), "renamed_correct_worlds": sum(after.values()),
        "original_viable_choice_pairs": sum(count == 2 for count in before.values()),
        "renamed_viable_choice_pairs": sum(count == 2 for count in after.values()),
        "choice_flips": sum(row["original"] != row["renamed"] for row in rows.values()),
    }


def check_canonicalization(witness, permutations):
    original = json.loads(witness.read_bytes())["pairs"]
    identity = 0
    for cluster, worlds in original.items():
        for world in worlds:
            source = world["outcomes"][0]["observation"]
            reordered = rename(source, permutations[cluster])
            canonical_source = canonicalize_atoms(source)
            if canonical_source != canonicalize_atoms(reordered):
                raise ValueError("alpha-renaming changed the canonical packet")
            identity += int(canonical_source == source)
    return {"matched_under_renaming": 2 * len(original), "original_already_canonical": identity}


def evaluate(args, tree):
    import torch

    from lcgrt import config
    from ppt_gym.live_lean import fingerprint
    from ppt_gym.ram_campaign import configure

    configure(tree.threads, 0)
    registration = json.loads(args.pending_cohort.read_bytes())["registration"]
    renamed, permutations = certified_renamings(args.cohort, args.witness, registration["previous_cohort_sha256"])
    components = json.loads(gzip.decompress(args.component_evidence.read_bytes()))
    if components["cohort_sha256"]["single-goal"] != sha256_file(args.cohort):
        raise ValueError("the component traces do not match the original cohort")
    counts, raw, source_hashes = {}, {}, {}
    verifier = fingerprint(config.REPL_EXE, config.LEAN_EXE, "4.19.0")
    with torch.no_grad():
        for seed, directory in zip(SEEDS, args.sessions, strict=True):
            label = str(seed)
            evidence = args.cohort.parent / EVIDENCE[seed]
            source_hashes[evidence.name] = sha256_file(evidence)
            original = json.loads(gzip.decompress(evidence.read_bytes()))
            if original["summary"]["cohort_sha256"] != registration["previous_cohort_sha256"]:
                raise ValueError("the original policy scores use different layouts")
            counts[label], raw[label] = {}, {}
            for arm in ARMS:
                model, manifest, _, weights = frozen_policy(directory / f"{arm}.sqlite", registration, seed, arm)
                saved = original["trials"][arm]
                if manifest["verifier"] != verifier or weights != saved["trial"]["final_weights_sha256"]:
                    raise ValueError("the renamed proof or policy uses a different verifier or checkpoint")
                passes = 4 if model.architecture == "hybrid" else 1
                decisions = saved["decisions"]["final"]
                results = {}
                if set(decisions) != set(renamed):
                    raise ValueError("renaming evaluation has a different set of theorem roots")
                for root, (cluster, observation, winner) in renamed.items():
                    extra = {}
                    if arm == ARMS[0]:
                        logits, parts = observed_components(model, observation, passes)
                        previous = components["roots"][label]["single-goal"][root]["components"]
                        extra["largest_history_component_change"] = max(
                            float((part - part.new_tensor(previous[name])).abs().max()) for name, part in parts.items()
                        )
                    else:
                        logits, _ = model.recover(canonical(observation), passes)
                    if not torch.isfinite(logits).all():
                        raise ValueError("renamed recovery packet produced a nonfinite decision")
                    results[root] = {"cluster": cluster, "winner": winner, "original": decisions[root],
                                     "renamed": int(logits.argmax()), **extra}
                counts[label][arm] = measure(results)
                raw[label][arm] = results
                print(seed, arm, counts[label][arm], flush=True)
    return {
        "scope": "paired development diagnostic under proposition alpha-renaming",
        "cohort_sha256": sha256_file(args.cohort), "witness_sha256": sha256_file(args.witness),
        "component_evidence_sha256": sha256_file(args.component_evidence),
        "verified_renamed_witness_proofs": len(renamed), "verifier": verifier,
        "original_evidence_sha256": source_hashes, "renamings": permutations, "counts": counts,
        "root_decisions": raw, "canonicalization": check_canonicalization(args.witness, permutations),
        "memory": tree.stats(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cohort", "witness", "pending_cohort", "component_evidence", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("sessions", type=Path, nargs=len(SEEDS))
    args = parser.parse_args()
    tree = TreeBudget(2, 2)
    try:
        report = evaluate(args, tree)
        with args.output.open("xb") as handle:
            handle.write(gzip.compress(canonical(report).encode("utf-8"), mtime=0))
    finally:
        tree.close()


if __name__ == "__main__":
    main()
