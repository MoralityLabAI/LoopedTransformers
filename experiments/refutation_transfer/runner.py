"""RefutationBench pilot using the *actual LR-01 tied TypedLoop cell*.

Only source-family-disjoint six synthetic training cases update model weights.
The public development set is evaluated once per prespecified run. This is a
structured-verdict and evidence-localization pilot, not an independent proof
checker or a natural-language proof generator. No LR-01 pretrained weights are
available in the repository; the shared cell is initialized anew.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import random
import re
import resource
import sys
import time

import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
LR01_DIR = HERE.parent / "lean_reasoner"
sys.path.insert(0, str(LR01_DIR))
from train import TypedLoop  # original LoopedTransformers LR-01 implementation

VOCAB = 1024
WIDTH = 24
MAX_EVIDENCE = 10
MAX_ATOMS = 1 + MAX_EVIDENCE + 4
MAX_RULES = MAX_EVIDENCE + 4
MAX_WORDS = 80
VERDICTS = ("established", "conditional", "refuted", "insufficient")
FLAWS = (
    "assumed_key_claim", "missing_semantic_bridge", "circular_case_split",
    "invalid_soundness_inference", "unsupported_model_construction",
    "incompatible_requirements", "inference_gap", "missing_premise",
)
STOP = set(("the a an is are of to for in with and or that as at by on it be "
            "this these those then from there into its their said also can "
            "would have has had will which while under both same other").split())
WORDS = re.compile(r"[A-Za-z0-9_]{2,}")


def load_bench(bench_root: Path):
    root = bench_root.resolve()
    sys.path.insert(0, str(root))
    from research.refutation_bench import bench
    from research.refutation_bench.expansion import suite
    cases, gold, _ = suite.load_combined()
    assert len(cases) == 25, "dataset version changed; repin and review"
    train = [c for c in cases if c["split"] == "train"]
    dev = [c for c in cases if c["split"] == "dev"]
    assert len(train) == 6 and len(dev) == 19
    families_train = {c["family_id"] for c in train}
    assert not (families_train & {c["family_id"] for c in dev})
    assert all(c["kind"] != "source_audit" for c in train)
    answers = {g["case_id"]: g for g in gold}
    return bench, train, dev, answers


def stable_tokens(value: str) -> list[int]:
    """Hash words, preserving token identity across runs without a tokenizer download."""
    ids = []
    for word in WORDS.findall(value.lower()):
        if word in STOP:
            continue
        digest = hashlib.blake2b(word.encode(), digest_size=4, person=b"LR01RB").digest()
        ids.append(1 + (int.from_bytes(digest, "big") % (VOCAB - 1)))
        if len(ids) >= MAX_WORDS:
            break
    return ids or [1]


def shuffled_case(case: dict, rng: random.Random) -> dict:
    out = deepcopy(case)
    rng.shuffle(out["evidence"])
    return out


def batch_encode(cases: list[dict], answers: dict | None = None) -> dict:
    """Build a bipartite claim/evidence/verdict graph independent of gold labels."""
    B = len(cases)
    ids = torch.zeros((B, MAX_ATOMS, MAX_WORDS), dtype=torch.long)
    roles = torch.full((B, MAX_ATOMS), 3, dtype=torch.long)  # 3 = padding
    rule_types = torch.full((B, MAX_RULES), 5, dtype=torch.long)  # 5 = padding
    prem = torch.zeros(B, MAX_RULES, MAX_ATOMS)
    head = torch.zeros_like(prem)
    n_ev = torch.zeros(B, dtype=torch.long)
    ev_mask = torch.zeros(B, MAX_EVIDENCE, dtype=torch.bool)
    verdict_y = torch.full((B,), -1, dtype=torch.long)
    flaw_y = torch.full((B,), -1, dtype=torch.long)
    evidence_y = torch.zeros(B, MAX_EVIDENCE)

    for b, case in enumerate(cases):
        evidence = case["evidence"]
        n = len(evidence)
        if not 1 <= n <= MAX_EVIDENCE:
            raise ValueError("unsupported evidence count: " + case["case_id"])
        n_ev[b] = n
        text_nodes = [case["claim"]] + [e["text"] for e in evidence]
        for a, text in enumerate(text_nodes):
            word_ids = stable_tokens(text)
            ids[b, a, :len(word_ids)] = torch.tensor(word_ids)
            roles[b, a] = 0 if a == 0 else 1

        for j in range(n):
            ev_mask[b, j] = True
            # Evidence rule links the central claim and a cited source sentence.
            prem[b, j, 0] = 1.0
            head[b, j, 1 + j] = 1.0
            rule_types[b, j] = 0

        for c in range(4):
            a = 1 + n + c
            r = n + c
            roles[b, a] = 2
            # Every verdict has an identical structural edge. Its identity
            # enters only through the trainable class-specific rule embedding.
            prem[b, r, 0] = 1.0
            head[b, r, a] = 1.0
            rule_types[b, r] = c + 1

        if answers is not None:
            answer = answers[case["case_id"]]
            verdict_y[b] = VERDICTS.index(answer["verdict"])
            if answer["verdict"] == "refuted":
                first = next(f for f in answer["findings"] if f["decisive"])
                flaw_y[b] = FLAWS.index(first["flaw_type"])
                accepted_set = set(first["evidence_sets"][0])
                for j, e in enumerate(evidence):
                    evidence_y[b, j] = float(e["evidence_id"] in accepted_set)

    return {"ids": ids, "roles": roles, "rule_types": rule_types,
            "prem": prem, "head": head, "n_ev": n_ev, "ev_mask": ev_mask,
            "verdict_y": verdict_y, "flaw_y": flaw_y, "evidence_y": evidence_y}


class RefutationPolicy(nn.Module):
    """LR-01 TypedLoop with a compact hashed-text encoder and evidence readouts."""

    def __init__(self, loops: int):
        super().__init__()
        self.loops = loops
        self.token = nn.Embedding(VOCAB, WIDTH, padding_idx=0)
        self.role = nn.Embedding(4, WIDTH, padding_idx=3)
        self.rule = nn.Embedding(6, WIDTH, padding_idx=5)
        self.cell = TypedLoop(WIDTH)
        self.candidate = nn.Linear(3 * WIDTH, 1)
        self.flaw = nn.Linear(3 * WIDTH, len(FLAWS))
        self.evidence_score = nn.Linear(WIDTH, 1)

    def forward(self, b: dict):
        x = self.token(b["ids"])
        lengths = (b["ids"] != 0).sum(-1).clamp_min(1).unsqueeze(-1)
        a = x.sum(-2) / lengths + self.role(b["roles"])
        r = self.rule(b["rule_types"])
        prem = b["prem"]
        head = b["head"]
        for _ in range(self.loops):
            a, r = self.cell(a, r, prem, head)
        parents = prem @ a / prem.sum(-1, keepdim=True).clamp_min(1.0)
        feats = torch.cat([r, head @ a, parents], dim=-1)
        idx = b["n_ev"][:, None] + torch.arange(4)[None, :]
        select = feats.gather(1, idx[:, :, None].expand(-1, -1, feats.shape[-1]))
        logits = self.candidate(select).squeeze(-1)
        # Flaw readout is conditioned on the dedicated "refuted" hypothesis.
        flaw_logits = self.flaw(select[:, VERDICTS.index("refuted"), :])
        ev_logits = self.evidence_score(a[:, 1:1 + MAX_EVIDENCE]).squeeze(-1)
        return logits, flaw_logits, ev_logits


def loss_function(outputs, b):
    logits, flaw_logits, ev_logits = outputs
    main = F.cross_entropy(logits, b["verdict_y"])
    active = b["flaw_y"] >= 0
    flaws = F.cross_entropy(flaw_logits[active], b["flaw_y"][active]) if active.any() else main * 0
    ev = F.binary_cross_entropy_with_logits(ev_logits, b["evidence_y"], reduction="none")
    ev = (ev * b["ev_mask"]).sum() / b["ev_mask"].sum().clamp_min(1)
    return main + 0.25 * flaws + 0.20 * ev


@torch.no_grad()
def respond(model: RefutationPolicy, cases: list[dict]) -> list[dict]:
    model.eval()
    b = batch_encode(cases)
    logits, flaws, evs = model(b)
    probs = F.softmax(logits, -1)
    result = []
    for i, case in enumerate(cases):
        choice = int(probs[i].argmax())
        verdict = VERDICTS[choice]
        findings = []
        if verdict == "refuted":
            flaw = FLAWS[int(flaws[i].argmax())]
            n = int(b["n_ev"][i])
            evidence_scores = evs[i, :n].sigmoid()
            picked = [j for j in range(n) if evidence_scores[j].item() >= 0.5]
            if not picked:
                picked = [int(evidence_scores.argmax())]
            picked = sorted(picked, key=lambda j: -float(evidence_scores[j]))[:3]
            ids = [case["evidence"][j]["evidence_id"] for j in picked]
            findings = [{"flaw_type": flaw, "evidence_ids": ids,
                         "explanation": "LR-01 verdict/evidence head selects cited passages; independent semantic verification is required."}]
        result.append({"case_id": case["case_id"], "verdict": verdict,
                       "confidence": float(probs[i, choice]), "findings": findings,
                       "assertions": []})
    return result


def train_model(train_cases: list[dict], answers: dict, *,
                loops: int, seed: int, steps: int, shuffle_labels: bool = False):
    torch.manual_seed(seed)
    random.seed(seed)
    torch.use_deterministic_algorithms(True)
    model = RefutationPolicy(loops)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=0.01)
    training_answers = {c["case_id"]: deepcopy(answers[c["case_id"]]) for c in train_cases}
    if shuffle_labels:
        # Fixed permutation of *only* the six training labels. The gold
        # development labels remain unseen to the optimizer.
        labels = [training_answers[c["case_id"]]["verdict"] for c in train_cases]
        random.Random(seed + 9917).shuffle(labels)
        for c, label in zip(train_cases, labels):
            training_answers[c["case_id"]]["verdict"] = label
            # Keep auxiliary supervision coherent with shuffled verdict.
            if label != "refuted":
                training_answers[c["case_id"]]["findings"] = []
            elif not training_answers[c["case_id"]]["findings"]:
                # A refuted label without a finding cannot train the
                # auxiliary heads; class-only ablation instead.
                training_answers[c["case_id"]]["findings"] = [{
                    "decisive": True, "flaw_type": "inference_gap",
                    "evidence_sets": [[c["evidence"][0]["evidence_id"]]]}]
    rng = random.Random(seed + 431)
    variants = [batch_encode([shuffled_case(c, rng) for c in train_cases],
                             training_answers) for _ in range(4)]
    started = time.perf_counter()
    losses = []
    for step in range(steps):
        b = variants[step % len(variants)]
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = loss_function(model(b), b)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 25 == 0 or step + 1 == steps:
            losses.append([step + 1, round(float(loss.detach()), 6)])
    return model, {"seed": seed, "loops": loops, "steps": steps,
                   "train_examples": len(train_cases), "losses": losses,
                   "train_seconds": round(time.perf_counter() - started, 4),
                   "param_count": sum(p.numel() for p in model.parameters()),
                   "label_shuffle_control": shuffle_labels}


def source_hashes(root: Path) -> dict[str, str]:
    rels = ["research/refutation_bench/data/cases.jsonl",
            "research/refutation_bench/data/gold.jsonl",
            "research/refutation_bench/expansion/cases.jsonl",
            "research/refutation_bench/expansion/gold.jsonl"]
    return {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in rels}


def model_run(bench, train, dev, gold, out_dir, loops, seed, steps, label_shuffle=False):
    model, info = train_model(train, gold, loops=loops, seed=seed, steps=steps,
                              shuffle_labels=label_shuffle)
    id_to_gold = {g["case_id"]: g for g in gold.values()}
    train_pred = respond(model, train)
    dev_pred = respond(model, dev)
    train_metrics = bench.evaluate(train, [id_to_gold[c["case_id"]] for c in train], train_pred)
    dev_metrics = bench.evaluate(dev, [id_to_gold[c["case_id"]] for c in dev], dev_pred)
    tag = "lr01_l" + str(loops) + "_s" + str(seed) + ("_labelshuffle" if label_shuffle else "")
    torch.save(model.state_dict(), out_dir / (tag + ".pt"))
    checkpoint_hash = hashlib.sha256((out_dir / (tag + ".pt")).read_bytes()).hexdigest()
    with (out_dir / (tag + "_dev_predictions.jsonl")).open("w", encoding="utf8") as f:
        for p in dev_pred:
            f.write(json.dumps(p, ensure_ascii=False, sort_keys=True) + "\n")
    info.update(tag=tag, checkpoint_sha256=checkpoint_hash,
                train_verdict_accuracy=train_metrics["metrics"]["verdict_accuracy"],
                dev_verdict_accuracy=dev_metrics["metrics"]["verdict_accuracy"],
                dev_macro_f1=dev_metrics["metrics"]["verdict_macro_f1"],
                dev_supported_refutation_rate=dev_metrics["metrics"]["supported_refutation_rate"],
                dev_finding_precision=dev_metrics["metrics"]["finding_precision"],
                dev_false_refutation_rate=dev_metrics["metrics"]["nonrefuted_false_refutation_rate"],
                dev_mean_response_bytes=dev_metrics["metrics"]["mean_response_utf8_bytes"],
                dev_confusion=dev_metrics["confusion"],
                dev_rows=[{"case_id": r["case_id"], "gold": r["gold_verdict"],
                           "pred": r.get("predicted_verdict", "invalid"),
                           "supported_refutation": r["supported_refutation"]}
                          for r in dev_metrics["cases"]])
    return info


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmark-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--loops", type=int, nargs="+", default=[1, 4])
    p.add_argument("--seeds", type=int, nargs="+", default=[17, 29])
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--label-shuffle", action="store_true")
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    root = args.benchmark_root.resolve()
    bench, train, dev, gold = load_bench(root)
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    runs = []
    for loops in args.loops:
        for seed in args.seeds:
            run = model_run(bench, train, dev, gold, args.out, loops, seed,
                            args.steps, label_shuffle=False)
            print(json.dumps({k: v for k, v in run.items()
                              if k in ("tag", "train_verdict_accuracy", "dev_verdict_accuracy",
                                       "dev_macro_f1", "dev_supported_refutation_rate",
                                       "dev_false_refutation_rate", "param_count")}), flush=True)
            runs.append(run)
    if args.label_shuffle:
        runs.append(model_run(bench, train, dev, gold, args.out,
                              max(args.loops), args.seeds[0], args.steps,
                              label_shuffle=True))
    majority = Counter(gold[c["case_id"]]["verdict"] for c in train).most_common(1)[0][0]
    dev_labels = [gold[c["case_id"]]["verdict"] for c in dev]
    naive = {label: sum(g == label for g in dev_labels) / len(dev_labels)
             for label in ("refuted", majority, "insufficient")}
    report = {
        "status": "real_cpu_training_and_inference",
        "pilot_scope": "verdict_classification_plus_learned_evidence_pointer",
        "reasoner": "LoopedTransformers/experiments/lean_reasoner/train.py:TypedLoop",
        "pretrained_lr01_weights_loaded": False,
        "training_family_ids": sorted({c["family_id"] for c in train}),
        "development_family_ids": sorted({c["family_id"] for c in dev}),
        "train_case_count": len(train), "dev_case_count": len(dev),
        "heldout_test_count": 0,
        "benchmark_sources_sha256": source_hashes(root),
        "torch": torch.__version__, "cpu_threads": args.threads,
        "total_wall_seconds": round(time.perf_counter() - started, 4),
        "process_peak_rss_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 3),
        "naive_constant_verdict_accuracy": naive, "runs": runs,
        "notes": ["Development answer key is public; no blind test is claimed.",
                  "Gold labels and findings from dev are used for reporting only.",
                  "Flaw/evidence predictions are pointer-head proxies, not independently checked mathematical refutations.",
                  "Any inference beyond the curated evidence paraphrases needs raw-source evaluation."]
    }
    report_path = args.out / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                           encoding="utf8")
    print("RECEIPT", report_path)
    print("TOTAL_SECONDS", report["total_wall_seconds"])


if __name__ == "__main__":
    main()
