"""NEC-01 PEF-07: frozen recurrent-visit native-neuron discovery and interventions.

Run discovery before opening heldout evaluation. The 12 visits share one
MLP parameter tensor; visit:neuron is an *occurrence*, not a unique weight.
The hook targets the GELU output passed to mlp_down for the memory token.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from research.pef07.pef07 import (L, D, STEPS, Looper, SEEDS, TASKS,
                                   as_tensor, load_checkpoint, model_digest,
                                   source_split, target)
from mechinterp.neuron_communities.core import (build_graph, decode, discover,
                                                 matched_sham, node, summary,
                                                 interval)

WIDTH = 2 * D
SCHEMA = "NEC-01/pef07-context-paired-native-neurons/v1"


def canonical(payload: dict) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":"))+"\n").encode()


def source_model(args):
    catalog, source = source_split()
    if args.smoke:
        torch.manual_seed(args.seed + (10000 if args.task == "threshold" else 0))
        model = Looper().eval()
        return model, catalog, source, {"provenance": "random-initialized-smoke",
                                         "model_sha256": model_digest(model)}
    if not args.checkpoint_dir:
        raise ValueError("frozen checkpoint directory required")
    root = Path(args.checkpoint_dir)
    ledger = json.loads((root/"sealed_training_ledger.json").read_text())
    if (ledger.get("campaign") != "PEF-07"
        or ledger.get("source_sha256") != source["sha256"]
        or not ledger.get("all_six_trunks_sealed_before_scoring")
        or len(ledger.get("checkpoints", [])) != 6):
        raise ValueError("incomplete or mismatched PEF-07 frozen ledger")
    rec = next((x for x in ledger["checkpoints"]
                if x["seed"] == args.seed and x["task"] == args.task), None)
    if rec is None or rec["steps"] != STEPS or rec["source_sha256"] != source["sha256"]:
        raise ValueError("registered frozen PEF-07 seed/task/steps mismatch")
    model = load_checkpoint(root, rec)
    return model, catalog, source, {"provenance": "pef07-frozen-trained",
                                     "model_sha256": model_digest(model),
                                     "checkpoint_sha256": rec["checkpoint_sha256"],
                                     "training_steps": rec["steps"]}


def _chunks(rows, contexts):
    if contexts < 1 or contexts > len(rows)//16:
        raise ValueError("context count out of bounds")
    for j in range(contexts):
        quartet = rows[16*j:16*(j+1)]
        if len(quartet) != 16 or len({x[4:] for x in quartet}) != 1:
            raise AssertionError("source context quartet was split")
        yield quartet


def context_attributions(model: Looper, bits: torch.Tensor, top_k: int) -> list[str]:
    if top_k < 1 or top_k >= WIDTH or bits.shape != (16, 12):
        raise ValueError("expected complete 16-state quartet and valid top_k")
    preactivations = []

    def capture(module, args):
        preactivations.append(args[0])

    handle = model.block.mlp_down.register_forward_pre_hook(capture)
    try:
        # Do not disable autograd: attribution is local to each occurrence.
        logits = model.logits(bits)
        chosen = logits.argmax(dim=1)
        margin = (logits.gather(1, chosen[:, None]).squeeze(1)
                  - logits.gather(1, (1-chosen)[:, None]).squeeze(1))
        if len(preactivations) != L:
            raise AssertionError("wrong number of shared-MLP visits")
        grads = torch.autograd.grad(margin.sum(), preactivations)
        selected = []
        for visit, (a, g) in enumerate(zip(preactivations, grads)):
            if a.shape != (16, 2, WIDTH):
                raise AssertionError("unexpected GELU tensor shape")
            # Memory token is the only state carried to the next visit.
            magnitudes = (a[:, 1, :] * g[:, 1, :]).abs().mean(dim=0)
            indices = magnitudes.topk(top_k).indices.tolist()
            selected.extend(node(visit, int(j)) for j in indices)
        return selected
    finally:
        handle.remove()


def visit_masks(mode: str, candidate: list[str], hubs: list[str],
                sham: list[str]) -> dict[int, torch.Tensor]:
    if mode not in ("ablate", "retain", "hubs", "blank", "sham"):
        raise ValueError("unknown intervention mode")
    chosen = defaultdict(set)
    for key in candidate:
        visit, j = decode(key)
        if visit >= L or j >= WIDTH:
            raise ValueError("unit outside registered model")
        chosen[visit].add(j)
    hub_lookup = defaultdict(set)
    for key in hubs:
        visit, j = decode(key)
        if visit < L and j < WIDTH:
            hub_lookup[visit].add(j)
    sham_lookup = defaultdict(set)
    for key in sham:
        visit, j = decode(key)
        if visit < L and j < WIDTH:
            sham_lookup[visit].add(j)
    if {k: len(v) for k, v in chosen.items()} != {
            k: len(v) for k, v in sham_lookup.items()}:
        raise ValueError("sham does not match candidate visit counts")
    masks = {}
    for visit, selected in chosen.items():
        gate = torch.ones(WIDTH)
        if mode == "ablate":
            gate[list(selected)] = 0.
        elif mode == "sham":
            gate[list(sham_lookup[visit])] = 0.
        else:
            gate.zero_()
            if mode == "retain":
                gate[list(selected | hub_lookup[visit])] = 1.
            elif mode == "hubs" and hub_lookup[visit]:
                gate[list(hub_lookup[visit])] = 1.
        masks[visit] = gate
    return masks


@torch.no_grad()
def intervened_logits(model: Looper, bits: torch.Tensor,
                      masks: dict[int, torch.Tensor]) -> torch.Tensor:
    call = 0

    def change(module, args):
        nonlocal call
        visit = call
        call += 1
        if visit in masks:
            gate = masks[visit].to(args[0])
            gated = args[0].clone()
            gated[:, 1, :] = gated[:, 1, :] * gate
            return (gated,)
        return None

    hook = model.block.mlp_down.register_forward_pre_hook(change)
    try:
        z = model.logits(bits)
        if call != L:
            raise AssertionError("intervention missed a visit")
        return z
    finally:
        hook.remove()


def paired_effect(clean: torch.Tensor, modified: torch.Tensor,
                  labels: torch.Tensor) -> dict[str, float]:
    if clean.shape != modified.shape or clean.shape != (16, 2):
        raise ValueError("expected complete quartet logits")
    lp = F.log_softmax(clean.double(), dim=1)
    lq = F.log_softmax(modified.double(), dim=1)
    p = lp.exp()
    top = clean.argmax(dim=1)
    kl = (p * (lp - lq)).sum(dim=1).clamp_min(0)
    before = (top == labels).double()
    after = (modified.argmax(dim=1) == labels).double()
    top_delta = p.gather(1, top[:, None]).squeeze(1) - (
        lq.exp().gather(1, top[:, None]).squeeze(1))
    return {"kl": float(kl.mean()),
            "correctness_drop": float((before-after).mean()),
            "top_probability_drop": float(top_delta.mean())}


def discovery(args):
    model, catalog, source, identity = source_model(args)
    torch.set_num_threads(1)
    model_hash = model_digest(model)
    rows = []
    for group in _chunks(catalog["train"], args.train_contexts):
        rows.append(context_attributions(model, as_tensor(group), args.top_k))
    graph = build_graph(rows, min_phi=args.min_phi,
                        min_pair_count=args.min_joint)
    candidates = discover(graph, max_candidates=args.max_candidates,
                          max_size=args.max_size)
    for item in candidates:
        item["frequency_matched_sham"] = matched_sham(item, graph,
                                                       mlp_width=WIDTH)
    if model_digest(model) != model_hash:
        raise AssertionError("native model modified in discovery")
    payload = {"schema": SCHEMA, "status": "exploratory-discovery",
               "source_sha256": source["sha256"], "identity": identity,
               "seed": args.seed, "task": args.task,
               "source_split": "train_only",
               "context_groups_used": args.train_contexts,
               "counterfactuals_per_context": 16,
               "top_k_per_visit": args.top_k,
               "frozen_candidate_selection_before_heldout": True,
               "graph": graph, "candidates": candidates,
               "intervention_scope": "native-GELU-memory-token-by-recurrent-visit",
               "discovery_uses_predicted_margin_without_task_labels": True}
    args.out.mkdir(parents=True, exist_ok=True)
    raw = canonical(payload)
    (args.out/"discovery.json").write_bytes(raw)
    (args.out/"discovery.sha256").write_text(hashlib.sha256(raw).hexdigest()+"\n")
    print(json.dumps({"status": payload["status"],
                      "candidates": len(candidates),
                      "graph_edges": len(graph["edges"]),
                      "discovery_sha256": hashlib.sha256(raw).hexdigest()}))
    return payload


def evaluation(args):
    if not args.smoke and args.heldout_contexts != 48:
        raise ValueError("frozen-model heldout evaluation requires all 48 contexts")
    raw = (args.out/"discovery.json").read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    expected = (args.out/"discovery.sha256").read_text().strip()
    if actual != expected:
        raise ValueError("discovery file changed after candidate freeze")
    frozen = json.loads(raw)
    model, catalog, source, identity = source_model(args)
    if (frozen["schema"] != SCHEMA or frozen["identity"] != identity
        or frozen["source_sha256"] != source["sha256"]
        or frozen["seed"] != args.seed or frozen["task"] != args.task):
        raise ValueError("model/source differs from the frozen discovery")
    model_hash = model_digest(model)
    torch.set_num_threads(1)
    rows = []
    raw_lines = []  # complete per-context paired logits, kept for independent inference
    groups = list(_chunks(catalog["heldout"], args.heldout_contexts))
    for item in frozen["candidates"]:
        conditions = ("ablate", "sham", "retain", "hubs", "blank")
        per_condition = {name: [] for name in conditions}
        paired_necessity, paired_sufficiency, layer_effect = [], [], []
        masks = {name: visit_masks(name, item["nodes"],
                                   frozen["graph"]["hubs"],
                                   item["frequency_matched_sham"])
                 for name in conditions}
        for group in groups:
            bits = as_tensor(group)
            y = target(bits, args.task)
            with torch.no_grad():
                clean = model.logits(bits)
            metrics = {}
            altered_logits = {}
            for name in conditions:
                changed = intervened_logits(model, bits, masks[name])
                metrics[name] = paired_effect(clean, changed, y)
                altered_logits[name] = changed.double().tolist()
                per_condition[name].append(metrics[name])
            raw_lines.append({
                "candidate_id": item["id"],
                "context_bits": list(group[0][4:]),
                "counterfactual_mask_order": list(range(16)),
                "labels": y.tolist(),
                "clean_logits": clean.double().tolist(),
                "intervention_logits": altered_logits,
                "per_condition_metrics": metrics,
            })
            paired_necessity.append(metrics["ablate"]["kl"]-metrics["sham"]["kl"])
            paired_sufficiency.append(metrics["hubs"]["kl"]-metrics["retain"]["kl"])
            layer_effect.append(metrics["blank"]["kl"])
        rows.append({
            "id": item["id"], "nodes": item["nodes"], "visits": item["visits"],
            "condition_summaries": {
                name: summary(per_condition[name], seed=1907+idx)
                for idx, name in enumerate(conditions)},
            "necessity_excess_kl_over_frequency_matched_sham":
                interval(paired_necessity, seed=1908),
            "sufficiency_kl_gain_over_hubs_only":
                interval(paired_sufficiency, seed=1909),
            "ablated_visit_layers_total_kl":
                interval(layer_effect, seed=1910),
            "interpretation": "exploratory; sufficiency is scoped to visited MLP occurrences"
        })
    if model_digest(model) != model_hash:
        raise AssertionError("native weights changed during intervention")
    trace_bytes = b"".join(canonical(entry) for entry in raw_lines)
    trace_name = "raw_paired_context_traces.jsonl"
    (args.out/trace_name).write_bytes(trace_bytes)
    trace_digest = hashlib.sha256(trace_bytes).hexdigest()
    report = {"schema": SCHEMA, "status": "exploratory-heldout-causal-tests",
              "registered_pef07_result_unchanged": True,
              "source_sha256": source["sha256"], "identity": identity,
              "candidate_roster_sha256": actual,
              "context_count": len(groups), "counterfactuals_per_context": 16,
              "independent_unit": "eight-bit context, 16 focal histories each",
              "causal_scope": "12 tied-weight visits; GELU neuron occurrences",
              "random_smoke": args.smoke, "tests": rows,
              "raw_paired_trace_filename": trace_name,
              "raw_paired_trace_sha256": trace_digest,
              "raw_paired_trace_records": len(raw_lines),
              "raw_paired_trace_contract": "one record per candidate per independent context; 16 clean and condition logits per record",
              "note": "Bootstraps are descriptive. No predeclared discovery-power or multiplicity gate; no mechanistic success claim."}
    (args.out/"heldout_causal.json").write_bytes(canonical(report))
    print(json.dumps({"status": report["status"], "tested": len(rows),
                      "context_count": len(groups)}))
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=("discover", "evaluate"))
    ap.add_argument("--checkpoint-dir", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, choices=SEEDS, default=103)
    ap.add_argument("--task", choices=TASKS, default="coalition")
    ap.add_argument("--smoke", action="store_true",
                    help="explicit random-initialized test, no scientific inference")
    ap.add_argument("--train-contexts", type=int, default=160)
    ap.add_argument("--heldout-contexts", type=int, default=48)
    ap.add_argument("--top-k", type=int, default=2)
    ap.add_argument("--min-phi", type=float, default=0.18)
    ap.add_argument("--min-joint", type=int, default=3)
    ap.add_argument("--max-size", type=int, default=24)
    ap.add_argument("--max-candidates", type=int, default=8)
    args = ap.parse_args(argv)
    if args.stage == "discover":
        return discovery(args)
    return evaluation(args)


if __name__ == "__main__":
    main()
