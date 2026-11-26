"""Additive AUX-GYM V2 diagnostics. V0 generators and receipts stay byte-identical."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import random

FAMILIES_V2 = ("masked_pointer_chase", "modular_recurrence")
ABSTAIN = 1101
POINTER = 800
MODULAR = 832


def digest_file(path, *, compressed=False):
    h = hashlib.sha256()
    opener = gzip.open if compressed else open
    with opener(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def write_rows(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in data:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def symbolic(family, semantic):
    """Independent trace and terminal oracle; stop permanently at the first hole."""
    if family == "masked_pointer_chase":
        node = semantic["start"]
        trace = [POINTER + node]
        for _ in range(semantic["hops"]):
            node = -1 if node == -1 else semantic["mapping"][node]
            trace.append(ABSTAIN if node == -1 else POINTER + node)
    elif family == "modular_recurrence":
        value = semantic["start"]
        trace = [MODULAR + value]
        for a, b in semantic["operations"]:
            value = (a * value + b) % 17
            trace.append(MODULAR + value)
    else:
        raise ValueError(f"unsupported oracle: {family}")
    return trace


def histogram(data):
    counts = Counter(r["target_token"] for r in data)
    depths = defaultdict(Counter)
    for r in data:
        depths[r["difficulty"]][r["target_token"]] += 1
    return {
        "examples": len(data), "targets": dict(counts),
        "by_depth_targets": {str(d): dict(c) for d, c in sorted(depths.items())},
        "distinct_fingerprints": len({r["fingerprint"] for r in data}),
        "distinct_groups": len({r["group_id"] for r in data}),
        "majority_share": max(counts.values()) / len(data) if data else None,
    }


def audit(root, *, pools=False):
    index = json.loads((root / "dataset_index.json").read_text())
    report = {"shards": {}, "verified_hashes": {}, "pool_payloads": {}}
    for name, entry in sorted(index["files"].items()):
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != entry["bytes"] or digest_file(path) != entry["sha256"]:
            raise ValueError(f"byte/hash mismatch: {name}")
        report["verified_hashes"][name] = entry["sha256"]
        if name.endswith(".jsonl"):
            data = rows(path)
            family = path.stem
            if family in FAMILIES_V2:
                for row in data:
                    oracle = symbolic(family, row["semantic"])
                    if oracle[-1] != row["target_token"] or len(oracle) != row["difficulty"] + 1:
                        raise ValueError(f"oracle disagreement: {row['example_id']}")
            report["shards"][name] = histogram(data)
        elif pools and name.endswith(".pt.gz"):
            got = digest_file(path, compressed=True)
            if got != entry["uncompressed_sha256"]:
                raise ValueError(f"payload hash mismatch: {name}")
            report["pool_payloads"][name] = got
    return report


def majority(data):
    counts = Counter(r["target_token"] for r in data)
    return min(counts, key=lambda k: (-counts[k], k))


def metrics(data, predictions, constant):
    """Unrestricted token predictions. All invalid tokens count as errors."""
    if len(data) != len(predictions) or not data:
        raise ValueError("prediction count mismatch or empty split")
    groups = defaultdict(list)
    by_depth = defaultdict(list)
    for r, p in zip(data, predictions):
        t = r["target_token"]
        groups[t].append(int(t == p))
        by_depth[r["difficulty"]].append((t, p))
    recall = {str(t): sum(v) / len(v) for t, v in sorted(groups.items())}
    out = {
        "n": len(data), "exact": sum(map(sum, groups.values())) / len(data),
        "macro_recall_observed_classes": sum(recall.values()) / len(recall),
        "class_recall": recall,
        "class_support": {str(t): len(v) for t, v in sorted(groups.items())},
        "constant_token": constant,
        "constant_exact": sum(r["target_token"] == constant for r in data) / len(data),
        "by_depth": {},
    }
    for d, pairs in sorted(by_depth.items()):
        supports = Counter(t for t, _ in pairs)
        correct = Counter(t for t, p in pairs if t == p)
        nodes = [(t, p) for t, p in pairs if t != ABSTAIN]
        abstains = [(t, p) for t, p in pairs if t == ABSTAIN]
        out["by_depth"][str(d)] = {
            "macro_recall_observed_classes": sum(correct[t] / n for t, n in supports.items()) / len(supports),
            "node_exact_if_masked": (sum(t == p for t, p in nodes) / len(nodes)
                                     if nodes and "mapping" in data[0]["semantic"] else None),
            "abstain_recall": (sum(t == p for t, p in abstains) / len(abstains)
                               if abstains else None),
            "n": len(pairs),
            "exact": sum(t == p for t, p in pairs) / len(pairs),
            "constant_exact": sum(t == constant for t, _ in pairs) / len(pairs),
        }
    if "mapping" in data[0]["semantic"] and "hops" in data[0]["semantic"]:
        a = [(r["target_token"], p) for r, p in zip(data, predictions)]
        tp = sum(t == ABSTAIN and p == ABSTAIN for t, p in a)
        abstain_n = sum(t == ABSTAIN for t, _ in a)
        predicted_n = sum(p == ABSTAIN for _, p in a)
        node_n = len(a) - abstain_n
        node_correct = sum(t != ABSTAIN and t == p for t, p in a)
        out["masked"] = {
            "abstain_recall": tp / abstain_n if abstain_n else None,
            "abstain_precision": tp / predicted_n if predicted_n else None,
            "abstain_prediction_rate": predicted_n / len(a),
            "node_exact": node_correct / node_n if node_n else None,
            "node_n": node_n, "abstain_n": abstain_n,
            "node_false_abstain": sum(t != ABSTAIN and p == ABSTAIN for t, p in a),
            "node_answer_rate": sum(p != ABSTAIN for _, p in a) / len(a),
            "always_abstain_exact": abstain_n / len(a),
        }
        if abstain_n and node_n:
            out["masked"]["binary_outcome_macro_exact"] = (tp / abstain_n + node_correct / node_n) / 2
    return out


def rescore(shard, predfile, calibration):
    data = rows(shard)
    records = rows(predfile)
    lookup = {r["example_id"]: r for r in records}
    if len(lookup) != len(records) or set(lookup) != {r["example_id"] for r in data}:
        raise ValueError("prediction IDs must exactly match shard")
    for r in data:
        p = lookup[r["example_id"]]
        if p["target"] != r["target_token"] or p["depth"] != r["difficulty"]:
            raise ValueError("prediction receipt disagrees with shard")
    return metrics(data, [lookup[r["example_id"]]["prediction"] for r in data],
                   majority(rows(calibration)))


def split_ops(config):
    from research_gym.interp.bundle import region_of
    from research_gym.interp.tasks import FAMILIES, structure_hash
    family = FAMILIES["modular_recurrence"]
    by_region = defaultdict(list)
    for a in range(1, 17):
        for b in range(17):
            semantic = {"start": 0, "operations": [[a, b]]}
            by_region[region_of(structure_hash(family, semantic), config.regions)].append((a, b))
    return dict(by_region)


def build(out, config_path, seed, train_quota, eval_quota, trial_limit):
    from research_gym.interp.bundle import BundleConfig, build_example, region_of
    from research_gym.interp.tasks import FAMILIES, Solution, structure_hash, fingerprint

    if train_quota < 1 or eval_quota < 1 or trial_limit < 1:
        raise ValueError("positive quotas and trial limit required")
    if out.exists():
        raise FileExistsError(f"choose a fresh output directory: {out}")
    config = BundleConfig.from_mapping(json.loads(config_path.read_text()))
    op_groups = split_ops(config)
    familiar = sorted(op_groups["train"])
    novel = sorted(op for region, ops in op_groups.items() if region != "train" for op in ops)
    manifest = {"seed": seed, "train_quota": train_quota, "eval_quota": eval_quota,
                "trial_limit": trial_limit, "config_sha256": digest_file(config_path),
                "depth1_operations": {k: len(v) for k, v in op_groups.items()}, "files": {}}
    seen = set()
    owner = {}
    specifications = (
        ("train", "train", range(1, 9), train_quota, "familiar"),
        ("validation", "validation", range(1, 9), eval_quota, "mixed"),
        ("locked_familiar", "locked", range(9, 13), eval_quota, "familiar"),
        ("locked_novel", "locked", range(9, 13), eval_quota, "one_novel"),
    )
    staged = {}
    for family_name in FAMILIES_V2:
        family = FAMILIES[family_name]
        for split, region, depths, quota, regime in specifications:
            if family_name == "masked_pointer_chase" and split == "locked_novel":
                continue
            rng = random.Random(f"{seed}:{family_name}:{split}")
            data = []
            for depth in depths:
                buckets = {t: [] for t in family.answer_tokens}
                for trial in range(trial_limit):
                    if all(len(v) == quota for v in buckets.values()):
                        break
                    if family_name == "modular_recurrence" and regime != "mixed":
                        ops = [list(rng.choice(familiar)) for _ in range(depth)]
                        if regime == "one_novel":
                            ops[rng.randrange(depth)] = list(rng.choice(novel))
                        semantic = {"start": rng.randrange(17), "operations": ops}
                    else:
                        semantic = family.sample(rng, depth)
                    group = structure_hash(family, semantic)
                    if region_of(group, config.regions) != region:
                        continue
                    if group in owner and owner[group] != split:
                        continue
                    identity = fingerprint(family, semantic)
                    if identity in seen:
                        continue
                    trace = symbolic(family_name, semantic)
                    bucket = buckets[trace[-1]]
                    if len(bucket) == quota:
                        continue
                    row = build_example(family, semantic, split).to_dict()
                    if row["target_token"] != trace[-1]:
                        raise ValueError("oracle disagreement")
                    row["symbolic_trace_tokens"] = trace
                    seen.add(identity)
                    owner[group] = split
                    bucket.append(row)
                if any(len(v) != quota for v in buckets.values()):
                    raise RuntimeError(f"quota shortfall {family_name}/{split}/{depth}: "
                                       f"{ {k: len(v) for k, v in buckets.items()} }")
                for bucket in buckets.values():
                    data.extend(bucket)
            if split == "train" and family_name == "modular_recurrence":
                # All 17 distinct zero-op identities, training-only. No V0 solver changes.
                for start in range(17):
                    semantic = {"start": start, "operations": []}
                    group = structure_hash(family, semantic)
                    if region_of(group, config.regions) != "train":
                        raise ValueError("identity structure is outside training region")
                    solution = Solution(MODULAR + start, 0, {"acc": [start], "remaining": [0]}, {})
                    row = build_example(family, semantic, split, solution=solution).to_dict()
                    row["symbolic_trace_tokens"] = [MODULAR + start]
                    data.append(row)
            rng.shuffle(data)
            staged[f"{split}/{family_name}.jsonl"] = data
    # Write only after every quota has succeeded.
    for name, data in staged.items():
        path = out / name
        write_rows(path, data)
        manifest["files"][name] = {"sha256": digest_file(path), **histogram(data)}
    dump(out / "manifest.json", manifest)
    return manifest


def trace_loss(model, output, targets, trace_targets, mode, weight):
    import torch
    import torch.nn.functional as F
    final = F.cross_entropy(output.logits[:, -1].float(), targets)
    if mode == "final":
        return final
    if mode not in ("repeated_final", "symbolic_trace", "shuffled_trace"):
        raise ValueError(mode)
    aux = []
    for i, state in enumerate(output.visit_outputs[:-1]):
        desired = targets if mode == "repeated_final" else trace_targets[:, i]
        if mode == "shuffled_trace":
            desired = desired.roll(1)
        aux.append(F.cross_entropy(model.lens(state[:, -1]).float(), desired))
    return final + weight * torch.stack(aux).mean() if aux else final


def run_pilot(args):
    import torch
    from research_gym.interp.models import build_model
    torch.manual_seed(args.seed)
    if args.threads:
        torch.set_num_threads(args.threads)
    rng = torch.Generator().manual_seed(args.seed)
    device = torch.device(args.device)
    manifest = json.loads((args.data / "manifest.json").read_text())

    def checked(split):
        name = f"{split}/{args.family}.jsonl"
        path = args.data / name
        if digest_file(path) != manifest["files"][name]["sha256"]:
            raise ValueError("dataset hash mismatch")
        return rows(path)

    if args.out.exists():
        raise FileExistsError(args.out)
    train = checked("train")
    # Do not read the locked files during development training.
    val = checked("validation")
    model = build_model(args.arm, visits=8).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    tokens = torch.tensor([r["tokens"] for r in train])
    targets = torch.tensor([r["target_token"] for r in train])
    trace_targets = torch.tensor([
        [r["symbolic_trace_tokens"][min(i, r["difficulty"])] for i in range(1, 9)]
        for r in train
    ])
    # Depth-balanced sampling; depth 0 occupies the same share as one other depth.
    buckets = [torch.tensor([i for i, r in enumerate(train) if r["difficulty"] == d])
               for d in sorted({r["difficulty"] for r in train})]
    for _ in range(args.steps):
        choices = torch.randint(len(buckets), (args.batch,), generator=rng)
        idx = torch.tensor([int(buckets[c][torch.randint(len(buckets[c]), (1,), generator=rng)])
                            for c in choices.tolist()])
        optimizer.zero_grad()
        output = model(tokens[idx].to(device))
        loss = trace_loss(model, output, targets[idx].to(device),
                          trace_targets[idx].to(device), args.mode, args.weight)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    model.eval()
    with torch.no_grad():
        pred = []
        for i in range(0, len(val), args.batch):
            x = torch.tensor([r["tokens"] for r in val[i:i + args.batch]], device=device)
            pred.extend(model(x).logits[:, -1].argmax(-1).cpu().tolist())
    args.out.mkdir(parents=True, exist_ok=False)
    torch.save(model.state_dict(), args.out / "model.pt")
    write_rows(args.out / "validation_predictions.jsonl",
               [{"example_id": r["example_id"], "depth": r["difficulty"],
                 "target": r["target_token"], "prediction": p} for r, p in zip(val, pred)])
    dump(args.out / "receipt.json", {
        "arm": args.arm, "mode": args.mode, "family": args.family, "seed": args.seed,
        "steps": args.steps, "batch": args.batch, "visits": 8, "lr": args.lr,
        "aux_weight": args.weight, "device": str(device),
        "parameters": sum(p.numel() for p in model.parameters()),
        "manifest_sha256": digest_file(args.data / "manifest.json"),
        "checkpoint_sha256": digest_file(args.out / "model.pt"),
        "predictions_sha256": digest_file(args.out / "validation_predictions.jsonl"),
        "validation": metrics(val, pred, majority(train)),
    })


def evaluate(args):
    """Evaluate a frozen pilot checkpoint; explicit opt-in required for locked splits."""
    import torch
    from research_gym.interp.models import build_model

    receipt = json.loads((args.run / "receipt.json").read_text())
    manifest = json.loads((args.data / "manifest.json").read_text())
    if digest_file(args.data / "manifest.json") != receipt["manifest_sha256"]:
        raise ValueError("checkpoint and dataset manifest mismatch")
    if digest_file(args.run / "model.pt") != receipt["checkpoint_sha256"]:
        raise ValueError("checkpoint hash mismatch")
    if args.split.startswith("locked") and not args.unlock:
        raise ValueError("locked evaluation requires --unlock and a frozen analysis plan")
    if args.out.exists():
        raise FileExistsError(args.out)
    name = f"{args.split}/{receipt['family']}.jsonl"
    path = args.data / name
    if digest_file(path) != manifest["files"][name]["sha256"]:
        raise ValueError("shard hash mismatch")
    if args.threads:
        torch.set_num_threads(args.threads)
    data = rows(path)
    model = build_model(receipt["arm"], visits=receipt["visits"])
    model.load_state_dict(torch.load(args.run / "model.pt", map_location="cpu", weights_only=True))
    model.eval()
    device = torch.device(args.device)
    model.to(device)
    visits = receipt["visits"]
    if args.visits is not None:
        visits = args.visits
        if visits != receipt["visits"] and receipt["arm"] != "LOOP-T":
            raise ValueError("only tied arm supports the extra-visit diagnostic")
    if visits < 1:
        raise ValueError("positive visit count required")
    pred, trace_hits, trace_count = [], Counter(), Counter()
    pred_rows = []
    with torch.no_grad():
        for offset in range(0, len(data), args.batch):
            chunk = data[offset:offset + args.batch]
            x = torch.tensor([r["tokens"] for r in chunk], device=device)
            output = model(x, depth_visits=visits)
            final = output.logits[:, -1].argmax(-1).cpu().tolist()
            all_states = [model.lens(state[:, -1]).argmax(-1).cpu().tolist()
                          for state in output.visit_outputs]
            pred.extend(final)
            for j, r in enumerate(chunk):
                path_pred = [p[j] for p in all_states]
                for i, prediction in enumerate(path_pred, 1):
                    expected = r["symbolic_trace_tokens"][min(i, r["difficulty"])]
                    key = f"{r['difficulty']}:{i}"
                    trace_count[key] += 1
                    trace_hits[key] += int(prediction == expected)
                pred_rows.append({"example_id": r["example_id"], "depth": r["difficulty"],
                                  "target": r["target_token"], "prediction": final[j],
                                  "visit_predictions": path_pred})
    args.out.mkdir(parents=True, exist_ok=False)
    write_rows(args.out / "predictions.jsonl", pred_rows)
    dump(args.out / "evaluation.json", {
        "run_receipt_sha256": digest_file(args.run / "receipt.json"),
        "checkpoint_sha256": receipt["checkpoint_sha256"], "shard_sha256": digest_file(path),
        "split": args.split, "device": str(device), "visits": visits,
        "training_visits": receipt["visits"],
        "predictions_sha256": digest_file(args.out / "predictions.jsonl"),
        "score": metrics(data, pred, receipt["validation"]["constant_token"]),
        "trace_exact_by_depth_visit": {
            key: {"correct": trace_hits[key], "n": n, "exact": trace_hits[key] / n}
            for key, n in sorted(trace_count.items())
        },
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("audit")
    a.add_argument("--root", type=Path, required=True)
    a.add_argument("--out", type=Path, required=True)
    a.add_argument("--pools", action="store_true")
    r = sub.add_parser("rescore")
    for name in ("shard", "predfile", "calibration", "out"):
        r.add_argument("--" + name, type=Path, required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--config", type=Path, default=Path("configs/rmp_task_bundle_v0.json"))
    b.add_argument("--seed", type=int, default=2026112602)
    b.add_argument("--train-quota", type=int, default=64)
    b.add_argument("--eval-quota", type=int, default=8)
    b.add_argument("--trial-limit", type=int, default=1000000)
    t = sub.add_parser("pilot")
    t.add_argument("--data", type=Path, required=True)
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--family", choices=FAMILIES_V2, required=True)
    t.add_argument("--arm", choices=("LOOP-T", "FF-U"), required=True)
    t.add_argument("--mode", choices=("final", "repeated_final", "symbolic_trace", "shuffled_trace"),
                   required=True)
    t.add_argument("--seed", type=int, required=True)
    t.add_argument("--steps", type=int, default=1000)
    t.add_argument("--batch", type=int, default=64)
    t.add_argument("--weight", type=float, default=0.5)
    t.add_argument("--lr", type=float, default=0.001)
    t.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    t.add_argument("--threads", type=int, default=0)
    e = sub.add_parser("evaluate")
    e.add_argument("--run", type=Path, required=True)
    e.add_argument("--data", type=Path, required=True)
    e.add_argument("--out", type=Path, required=True)
    e.add_argument("--split", choices=("validation", "locked_familiar", "locked_novel"), required=True)
    e.add_argument("--unlock", action="store_true")
    e.add_argument("--batch", type=int, default=64)
    e.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    e.add_argument("--threads", type=int, default=0)
    e.add_argument("--visits", type=int)
    args = parser.parse_args()
    if args.cmd == "evaluate":
        if args.batch < 1:
            parser.error("positive batch required")
        evaluate(args)
    elif args.cmd == "audit":
        dump(args.out, audit(args.root, pools=args.pools))
    elif args.cmd == "rescore":
        dump(args.out, rescore(args.shard, args.predfile, args.calibration))
    elif args.cmd == "build":
        build(args.out, args.config, args.seed, args.train_quota, args.eval_quota, args.trial_limit)
    else:
        if args.steps < 1 or args.batch < 2 or args.weight < 0 or args.lr <= 0:
            parser.error("positive steps/lr, batch >= 2, nonnegative weight required")
        run_pilot(args)


if __name__ == "__main__":
    main()
