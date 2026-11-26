"""AUX-V2 counterfactual data and physical-weight factor memory pilot.

This is an SVD/BPD proxy, not an implementation of BPD's behavioral adversary.
The model still executes its original dense physical weights.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import json
from pathlib import Path
import random
import time

import torch
from torch.nn import functional as F

from research_gym.interp.aux_gym_v2 import (
    FAMILIES_V2, digest_file, dump, metrics, majority, rows, symbolic,
    trace_loss, write_rows, split_ops,
)


def candidate_edits(family, semantic, rng):
    """Change exactly one affine op or exchange two mapping destinations."""
    if family == "modular_recurrence":
        operations = semantic["operations"]
        edits = [(i, a, b) for i in range(len(operations))
                 for a in range(1, 17) for b in range(17)
                 if [a, b] != operations[i]]
        rng.shuffle(edits)
        for i, a, b in edits:
            change = copy.deepcopy(semantic)
            change["operations"][i] = [a, b]
            yield change, {"kind": "op_change", "site": i}
    else:
        mapping = semantic["mapping"]
        edits = [(i, j) for i in range(len(mapping)) for j in range(i + 1, len(mapping))
                 if mapping[i] != mapping[j]]
        rng.shuffle(edits)
        for i, j in edits:
            change = copy.deepcopy(semantic)
            change["mapping"][i], change["mapping"][j] = mapping[j], mapping[i]
            yield change, {"kind": "mask_move" if -1 in (mapping[i], mapping[j])
                           else "edge_rewire", "sites": [i, j]}


def make_pairs(data, out, config_path, *, seed, quota):
    from research_gym.interp.bundle import BundleConfig, build_example, region_of
    from research_gym.interp.tasks import FAMILIES, InvalidInstance, fingerprint, structure_hash
    if quota < 1 or out.exists():
        raise ValueError("positive quota and fresh output directory required")
    manifest = json.loads((data / "manifest.json").read_text())
    if digest_file(config_path) != manifest["config_sha256"]:
        raise ValueError("config mismatch")
    config = BundleConfig.from_mapping(json.loads(config_path.read_text()))
    familiar = set(split_ops(config)["train"])
    datasets, owners, used = {}, {}, set()
    for name, info in sorted(manifest["files"].items()):
        path = data / name
        if digest_file(path) != info["sha256"]:
            raise ValueError("source shard hash mismatch")
        split = name.split("/")[0]
        datasets[name] = rows(path)
        for row in datasets[name]:
            if owners.setdefault(row["group_id"], split) != split:
                raise ValueError("source structure leaked")
            used.add(row["fingerprint"])
    result = {"source_manifest_sha256": digest_file(data / "manifest.json"),
              "seed": seed, "quota_per_depth_kind": quota, "files": {}}
    staged = {}
    for name, examples in datasets.items():
        split, filename = name.split("/")
        family_name = filename.removesuffix(".jsonl")
        family = FAMILIES[family_name]
        region = "locked" if split.startswith("locked") else split
        rng = random.Random(f"{seed}:{name}")
        by_depth = defaultdict(list)
        for row in examples:
            if row["difficulty"] > 0:
                by_depth[row["difficulty"]].append(row)
        pairs = []
        for depth, anchors in sorted(by_depth.items()):
            rng.shuffle(anchors)
            for kind in ("changed_target", "same_target"):
                count = 0
                for anchor in anchors:
                    if count == quota:
                        break
                    original = anchor["symbolic_trace_tokens"]
                    for semantic, edit in candidate_edits(family_name, anchor["semantic"], rng):
                        if family_name == "modular_recurrence":
                            novel = sum(tuple(op) not in familiar for op in semantic["operations"])
                            if (split in ("train", "locked_familiar") and novel
                                    or split == "locked_novel" and novel != 1):
                                continue
                        group = structure_hash(family, semantic)
                        if region_of(group, config.regions) != region or owners.get(group, split) != split:
                            continue
                        identity = fingerprint(family, semantic)
                        if identity in used:
                            continue
                        try:
                            sibling = build_example(family, semantic, split).to_dict()
                        except InvalidInstance:
                            continue
                        trace = symbolic(family_name, semantic)
                        if sibling["target_token"] != trace[-1]:
                            raise ValueError("oracle mismatch")
                        changed = trace[-1] != original[-1]
                        if changed != (kind == "changed_target"):
                            continue
                        first = next((i for i, (a, b) in enumerate(zip(original, trace)) if a != b), None)
                        if changed and (first is None or first < 1):
                            raise ValueError("invalid divergence")
                        # A same-target control must preserve the entire symbolic trace.
                        if not changed and first is not None:
                            continue
                        sibling["symbolic_trace_tokens"] = trace
                        pairs.append({"kind": kind, "first_divergence": first, "edit": edit,
                                      "anchor": anchor, "donor": sibling})
                        used.add(identity)
                        owners[group] = split
                        count += 1
                        break
                if count != quota:
                    raise RuntimeError(f"pair shortfall {name}/{depth}/{kind}: {count}/{quota}")
        staged[name] = pairs
    for name, pairs in staged.items():
        path = out / name
        write_rows(path, pairs)
        result["files"][name] = {"sha256": digest_file(path), "pairs": len(pairs),
                                 "by_kind": dict(Counter(p["kind"] for p in pairs))}
    dump(out / "manifest.json", result)
    return result


class FactorMemory:
    """CPU FIFO bank, fixed slots for (depth, visit, class), with explicit byte accounting."""

    def __init__(self, byte_cap, width, classes, *, depths=9, visits=8):
        self.classes = tuple(classes)
        self.depths, self.visits, self.width = depths, visits, width
        self.groups = depths * visits * len(classes)
        self.byte_cap = byte_cap
        if byte_cap < 0:
            raise ValueError("negative byte cap")
        # Counts and cursors are both int64. No Python per-record allocations.
        metadata_bytes = self.groups * 16
        self.slots = max(0, (byte_cap - metadata_bytes) // (self.groups * width * 4))
        if self.slots:
            self.values = torch.zeros(self.groups, self.slots, width)
            self.counts = torch.zeros(self.groups, dtype=torch.long)
            self.cursors = torch.zeros_like(self.counts)
        else:
            self.values = torch.empty(0)
            self.counts = torch.empty(0, dtype=torch.long)
            self.cursors = torch.empty_like(self.counts)
        self.lookups = self.hits = self.insertions = self.evictions = 0

    @property
    def allocated_bytes(self):
        return sum(t.numel() * t.element_size() for t in (self.values, self.counts, self.cursors))

    def keys(self, depths, visit, targets):
        if not 0 <= visit < self.visits:
            raise ValueError("visit outside map")
        if (depths < 0).any() or (depths >= self.depths).any():
            raise ValueError("depth outside map")
        classes = torch.tensor(self.classes, dtype=torch.long)
        matches = targets.cpu()[:, None] == classes
        if not bool(matches.any(1).all()):
            raise ValueError("unknown state label")
        ids = matches.long().argmax(1)
        group = depths.cpu().long() * self.visits + visit
        return group * len(self.classes) + ids, group * len(self.classes) + (ids + 1) % len(self.classes)

    def retrieve(self, keys, rng, samples):
        self.lookups += len(keys)
        if not self.slots:
            return torch.zeros(len(keys), self.width), torch.zeros(len(keys), dtype=torch.bool)
        counts = self.counts[keys]
        valid = counts > 0
        # Constant K gathers per query, with replacement. No scan through the RAM tier.
        indices = (torch.rand(len(keys), samples, generator=rng)
                   * counts.clamp_min(1)[:, None]).long()
        values = self.values[keys[:, None], indices].mean(1)
        self.hits += int(valid.sum())
        return values, valid

    @torch.no_grad()
    def insert(self, keys, values):
        if not self.slots:
            return
        for key, value in zip(keys.tolist(), values.detach().float().cpu()):
            slot = int(self.cursors[key])
            if int(self.counts[key]) == self.slots:
                self.evictions += 1
            self.values[key, slot].copy_(value)
            self.counts[key] = min(int(self.counts[key]) + 1, self.slots)
            self.cursors[key] = (slot + 1) % self.slots
            self.insertions += 1

    def receipt(self):
        return {"cap_bytes": self.byte_cap, "allocated_tensor_bytes": self.allocated_bytes,
                "slots_per_key": self.slots, "entries": int(self.counts.sum()),
                "lookups": self.lookups, "hits": self.hits, "insertions": self.insertions,
                "evictions": self.evictions}


def physical_basis(weight, rank, mode, generator):
    """One output basis per physical weight; report retained Frobenius energy."""
    w = weight.detach().float().cpu()
    if rank < 1 or rank > min(w.shape):
        raise ValueError("invalid rank")
    if mode == "svd":
        u, _, _ = torch.linalg.svd(w, full_matrices=False)
        basis = u[:, :rank].contiguous()
    elif mode == "random":
        basis = torch.linalg.qr(torch.randn(w.shape[0], rank, generator=generator)).Q
    else:
        raise ValueError(mode)
    energy = float((basis.T @ w).square().sum() / w.square().sum().clamp_min(1e-12))
    return basis, energy


def batch_donors(keys, values):
    """Zero persistent RAM control: same-key current-batch donors excluding self."""
    equal = keys[:, None] == keys[None, :]
    equal.fill_diagonal_(False)
    count = equal.sum(1)
    donor = equal.to(values.dtype) @ values.detach()
    return donor / count.clamp_min(1)[:, None], count > 0


def factor_loss(current, positive, negative, valid, basis):
    """Matched directional contrast, donor tensors detached; only current write receives gradients."""
    q = F.normalize(current.float() @ basis, dim=-1)
    p = F.normalize(positive.detach().float() @ basis, dim=-1)
    n = F.normalize(negative.detach().float() @ basis, dim=-1)
    losses = F.softplus(4 * ((q * n).sum(-1) - (q * p).sum(-1)))
    return (losses * valid.float()).mean()


def peak_rss_bytes():
    try:
        import resource
        import sys
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(value * (1 if sys.platform == "darwin" else 1024))
    except ImportError:
        return None


def training(args):
    from research_gym.interp.models import build_model
    from research_gym.interp.tasks import FAMILIES
    if args.out.exists():
        raise FileExistsError(args.out)
    if args.steps < 1 or args.batch < 2 or args.refresh < 1 or args.samples < 1:
        raise ValueError("positive steps, refresh, samples; batch >= 2")
    if args.mib < 0 or args.weight < 0 or args.rank < 1:
        raise ValueError("invalid memory or loss parameters")
    if (args.mode in ("off", "batch") and args.mib != 0
            or args.mode in ("memory", "shuffled_memory", "random_basis", "native_memory") and args.mib == 0):
        raise ValueError("zero RAM for off/batch, positive RAM for memory arms")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = torch.Generator().manual_seed(args.seed)
    map_rng = torch.Generator().manual_seed(args.seed + 7919)
    basis_rng = torch.Generator().manual_seed(args.seed + 104729)
    device = torch.device(args.device)
    manifest = json.loads((args.data / "manifest.json").read_text())
    def checked(split):
        name = f"{split}/{args.family}.jsonl"
        path = args.data / name
        if digest_file(path) != manifest["files"][name]["sha256"]:
            raise ValueError("shard hash mismatch")
        return rows(path)
    train, val = checked("train"), checked("validation")
    model = build_model("LOOP-T", visits=8).to(device)
    initial = None
    native_basis = None
    if args.init_run is not None:
        initial = json.loads((args.init_run / "receipt.json").read_text())
        if (initial["arm"] != "LOOP-T" or initial["family"] != args.family
                or initial["manifest_sha256"] != digest_file(args.data / "manifest.json")
                or initial["checkpoint_sha256"] != digest_file(args.init_run / "model.pt")):
            raise ValueError("warmstart provenance mismatch")
        model.load_state_dict(torch.load(args.init_run / "model.pt", map_location="cpu", weights_only=True))
    if args.mode == "native_memory":
        if initial is None or args.factor_dir is None:
            raise ValueError("native_memory requires --init-run and --factor-dir")
        from research_gym.interp.aux_gym_v2_native_bpd import load_native_basis
        native_basis = load_native_basis(args.factor_dir, initial["checkpoint_sha256"],
                                         args.rank, model.blocks[0].mlp[-1].weight)
    elif args.factor_dir is not None:
        raise ValueError("factor directory only for native_memory")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    site = model.blocks[0].mlp[-1]
    memory = FactorMemory(args.mib * 2**20, site.out_features, FAMILIES[args.family].answer_tokens)
    tokens = torch.tensor([r["tokens"] for r in train])
    targets = torch.tensor([r["target_token"] for r in train])
    depths = torch.tensor([r["difficulty"] for r in train])
    trace_targets = torch.tensor([
        [r["symbolic_trace_tokens"][min(i, r["difficulty"])] for i in range(1, 9)] for r in train])
    buckets = [torch.where(depths == d)[0] for d in sorted(set(depths.tolist()))]
    captures = []
    def hook(_module, _inputs, output):
        captures.append(output[:, -1])
    handle = site.register_forward_hook(hook)
    basis_time, refreshes, coverage, energy = 0.0, 0, 0, []
    loss_sum, factor_sum = 0.0, 0.0
    start = time.perf_counter()
    try:
        for step in range(args.steps):
            if step == 0 or (not args.freeze_basis and step % args.refresh == 0):
                before = time.perf_counter()
                if native_basis is not None:
                    basis = native_basis
                    w = site.weight.detach().float().cpu()
                    retained = float((basis.T @ w).square().sum() / w.square().sum().clamp_min(1e-12))
                else:
                    basis, retained = physical_basis(site.weight, args.rank,
                        "random" if args.mode == "random_basis" else "svd", basis_rng)
                basis = basis.to(device)
                basis_time += time.perf_counter() - before
                refreshes += 1
                energy.append(retained)
            choices = torch.randint(len(buckets), (args.batch,), generator=rng)
            idx = torch.tensor([int(buckets[c][torch.randint(len(buckets[c]), (1,), generator=rng)])
                                for c in choices.tolist()])
            captures.clear()
            opt.zero_grad()
            output = model(tokens[idx].to(device))
            if len(captures) != 8:
                raise RuntimeError("physical tied site must run eight times")
            base = trace_loss(model, output, targets[idx].to(device),
                              trace_targets[idx].to(device), "symbolic_trace", args.trace_weight)
            aux, pending = [], []
            for visit, write in enumerate(captures):
                keys, negative_keys = memory.keys(depths[idx], visit, trace_targets[idx, visit])
                current = write.float()
                if args.mode == "batch":
                    p, pm = batch_donors(keys.to(device), current)
                    # Next class is the matched negative; never use the same row.
                    equal = negative_keys.to(device)[:, None] == keys.to(device)[None, :]
                    nm = equal.any(1)
                    n = equal.to(current.dtype) @ current.detach()
                    n = n / equal.sum(1).clamp_min(1)[:, None]
                else:
                    p, pm = memory.retrieve(keys, map_rng, args.samples)
                    n, nm = memory.retrieve(negative_keys, map_rng, args.samples)
                    p, n, pm, nm = (x.to(device) for x in (p, n, pm, nm))
                valid = pm & nm
                coverage += int(valid.sum())
                aux.append(factor_loss(current, p, n, valid, basis))
                insert_keys = keys
                if args.mode == "shuffled_memory":
                    insert_keys, _ = memory.keys(depths[idx], visit, trace_targets[idx, visit].roll(1))
                pending.append((insert_keys, current.detach()))
            extra = torch.stack(aux).mean()
            loss = base + (0 if args.mode == "off" else args.weight) * extra
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            for keys, writes in pending:
                memory.insert(keys, writes)
            loss_sum += float(base.detach())
            factor_sum += float(extra.detach())
    finally:
        handle.remove()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    model.eval()
    pred = []
    with torch.no_grad():
        for offset in range(0, len(val), args.batch):
            x = torch.tensor([r["tokens"] for r in val[offset:offset + args.batch]], device=device)
            pred.extend(model(x).logits[:, -1].argmax(-1).cpu().tolist())
    args.out.mkdir(parents=True, exist_ok=False)
    torch.save(model.state_dict(), args.out / "model.pt")
    write_rows(args.out / "validation_predictions.jsonl", [
        {"example_id": r["example_id"], "depth": r["difficulty"],
         "target": r["target_token"], "prediction": p} for r, p in zip(val, pred)])
    dump(args.out / "receipt.json", {
        "arm": "LOOP-T", "mode": args.mode, "family": args.family, "seed": args.seed,
        "steps": args.steps, "batch": args.batch, "visits": 8, "lr": args.lr,
        "aux_weight": args.trace_weight, "factor_weight": args.weight, "rank": args.rank,
        "device": str(device), "parameters": sum(p.numel() for p in model.parameters()),
        "manifest_sha256": digest_file(args.data / "manifest.json"),
        "checkpoint_sha256": digest_file(args.out / "model.pt"),
        "predictions_sha256": digest_file(args.out / "validation_predictions.jsonl"),
        "site": "blocks.0.mlp.2",
        "method": ("native_bpd_learned_output_span" if args.mode == "native_memory"
                   else "physical_weight_svd_proxy"),
        "init_checkpoint_sha256": initial["checkpoint_sha256"] if initial else None,
        "native_factor_receipt_sha256": digest_file(args.factor_dir / "receipt.json") if args.factor_dir else None,
        "freeze_basis": args.freeze_basis or args.mode == "native_memory",
        "memory": memory.receipt(), "basis_refreshes": refreshes,
        "basis_retained_energy": energy, "basis_seconds": basis_time,
        "paired_donor_coverage": coverage / (args.steps * args.batch * 8),
        "training_seconds": elapsed, "peak_rss_bytes": peak_rss_bytes(),
        "base_loss_mean": loss_sum / args.steps, "factor_loss_mean": factor_sum / args.steps,
        "validation": metrics(val, pred, majority(train)),
    })


def patch_eval(args):
    from research_gym.interp.models import build_model
    if args.split.startswith("locked") and not args.unlock:
        raise ValueError("locked evaluation requires explicit --unlock")
    if args.out.exists() or args.visits < 1:
        raise ValueError("fresh output and positive visits required")
    torch.set_num_threads(args.threads)
    receipt = json.loads((args.run / "receipt.json").read_text())
    manifest = json.loads((args.pairs / "manifest.json").read_text())
    if receipt["manifest_sha256"] != manifest["source_manifest_sha256"]:
        raise ValueError("pair manifest/checkpoint mismatch")
    if receipt["checkpoint_sha256"] != digest_file(args.run / "model.pt"):
        raise ValueError("checkpoint hash mismatch")
    name = f"{args.split}/{receipt['family']}.jsonl"
    path = args.pairs / name
    if digest_file(path) != manifest["files"][name]["sha256"]:
        raise ValueError("pair shard hash mismatch")
    data = rows(path)
    model = build_model(receipt["arm"], visits=receipt["visits"])
    model.load_state_dict(torch.load(args.run / "model.pt", weights_only=True, map_location="cpu"))
    model.eval()
    if receipt["arm"] != "LOOP-T":
        raise ValueError("this physical tied-site evaluator requires LOOP-T")
    site = model.blocks[0].mlp[-1]
    rng = torch.Generator().manual_seed(args.seed)
    basis, energy = physical_basis(site.weight, args.rank, "svd", rng)
    records = []
    @torch.no_grad()
    def run(row, visit, delta=None):
        writes = []
        def hook(_m, _i, out):
            writes.append(out[:, -1].clone())
            if delta is not None and len(writes) == visit:
                out = out.clone()
                out[:, -1] += delta
                return out
            return None
        handle = site.register_forward_hook(hook)
        try:
            output = model(torch.tensor([row["tokens"]]), depth_visits=args.visits)
            return output.logits[0, -1].float(), writes
        finally:
            handle.remove()
    for pair in data:
        a, b = pair["anchor"], pair["donor"]
        visit = pair["first_divergence"] or min(a["difficulty"], args.visits)
        if visit > args.visits:
            records.append({"anchor": a["example_id"], "donor": b["example_id"],
                            "kind": pair["kind"], "status": "outside_visit_budget", "visit": visit})
            continue
        la, wa = run(a, visit)
        lb, wb = run(b, visit)
        delta = (wb[visit - 1] - wa[visit - 1]) @ basis @ basis.T
        # Random orthogonal-direction sham with identical physical write dose.
        noise = torch.randn(delta.shape, generator=rng) @ basis @ basis.T
        noise = noise * (delta.norm() / noise.norm().clamp_min(1e-12))
        lp, _ = run(a, visit, delta)
        ls, _ = run(a, visit, noise)
        donor, anchor = b["target_token"], a["target_token"]
        records.append({"anchor": a["example_id"], "donor": b["example_id"],
            "kind": pair["kind"], "status": "evaluated", "visit": visit,
            "anchor_target": anchor, "donor_target": donor,
            "anchor_prediction": int(la.argmax()), "donor_prediction": int(lb.argmax()),
            "patch_prediction": int(lp.argmax()), "sham_prediction": int(ls.argmax()),
            "dose": float(delta.norm()), "sham_dose": float(noise.norm()),
            "patch_donor_margin_shift": float((lp[donor] - lp[anchor]) - (la[donor] - la[anchor])),
            "sham_donor_margin_shift": float((ls[donor] - ls[anchor]) - (la[donor] - la[anchor]))})
    args.out.mkdir(parents=True, exist_ok=False)
    write_rows(args.out / "pairs.jsonl", records)
    counts = Counter(r["status"] for r in records)
    admitted = [r for r in records if r["status"] == "evaluated"
                and r["anchor_prediction"] == r["anchor_target"]
                and r["donor_prediction"] == r["donor_target"]]
    changed = [r for r in admitted if r["kind"] == "changed_target"]
    same = [r for r in admitted if r["kind"] == "same_target"]
    dump(args.out / "receipt.json", {
        "checkpoint_sha256": receipt["checkpoint_sha256"], "pairs_sha256": digest_file(path),
        "predictions_sha256": digest_file(args.out / "pairs.jsonl"), "rank": args.rank,
        "visits": args.visits, "seed": args.seed, "retained_energy": energy,
        "counts": dict(counts), "competent_changed_pairs": len(changed),
        "competent_same_pairs": len(same),
        "patch_target_hits": sum(r["patch_prediction"] == r["donor_target"] for r in changed),
        "sham_target_hits": sum(r["sham_prediction"] == r["donor_target"] for r in changed),
        "patch_same_target_preserved": sum(r["patch_prediction"] == r["anchor_target"] for r in same),
        "sham_same_target_preserved": sum(r["sham_prediction"] == r["anchor_target"] for r in same),
    })


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("pairs")
    a.add_argument("--data", type=Path, required=True)
    a.add_argument("--out", type=Path, required=True)
    a.add_argument("--config", type=Path, default=Path("configs/rmp_task_bundle_v0.json"))
    a.add_argument("--seed", type=int, default=2026112603)
    a.add_argument("--quota", type=int, default=4)
    t = sub.add_parser("train")
    t.add_argument("--data", type=Path, required=True)
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--family", choices=FAMILIES_V2, required=True)
    t.add_argument("--mode", choices=("off", "batch", "memory", "shuffled_memory", "random_basis", "native_memory"), required=True)
    t.add_argument("--mib", type=int, default=0)
    t.add_argument("--init-run", type=Path)
    t.add_argument("--factor-dir", type=Path)
    t.add_argument("--freeze-basis", action="store_true")
    t.add_argument("--seed", type=int, required=True)
    t.add_argument("--steps", type=int, default=1000)
    t.add_argument("--batch", type=int, default=64)
    t.add_argument("--rank", type=int, default=16)
    t.add_argument("--refresh", type=int, default=100)
    t.add_argument("--samples", type=int, default=4)
    t.add_argument("--weight", type=float, default=0.1)
    t.add_argument("--trace-weight", type=float, default=0.5)
    t.add_argument("--lr", type=float, default=0.001)
    t.add_argument("--threads", type=int, default=2)
    t.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    e = sub.add_parser("patch")
    e.add_argument("--run", type=Path, required=True)
    e.add_argument("--pairs", type=Path, required=True)
    e.add_argument("--out", type=Path, required=True)
    e.add_argument("--split", choices=("validation", "locked_familiar", "locked_novel"), required=True)
    e.add_argument("--unlock", action="store_true")
    e.add_argument("--rank", type=int, default=16)
    e.add_argument("--visits", type=int, default=8)
    e.add_argument("--seed", type=int, default=2026112604)
    e.add_argument("--threads", type=int, default=2)
    args = p.parse_args()
    if args.cmd == "pairs":
        make_pairs(args.data, args.out, args.config, seed=args.seed, quota=args.quota)
    elif args.cmd == "train":
        training(args)
    else:
        patch_eval(args)


if __name__ == "__main__":
    main()
