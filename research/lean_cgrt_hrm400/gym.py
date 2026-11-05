"""LR-CGRT local gym: synthetic proof-frontier training with Lean export.

Kernel certification is a separate mandatory step. A Python graph reachability
result is never represented as a Lean proof. No Mathlib data is consumed here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

FEATURES = 9
PRESETS = {"smoke": 64, "5m": 448, "15m": 800, "30m": 1120}


@dataclass
class Problem:
    facts: list[int]
    rules: list[tuple[list[int], int]]
    goal: int
    teacher: list[int]
    depth: int
    family: str
    name: str = ""

    @staticmethod
    def from_json(row):
        return Problem(**row)


def make_problem(seed: int, depth: int, family: str = "mix") -> Problem:
    """Create a unique Horn DAG with a planted sufficient proof and distractors."""
    rng = random.Random(seed)
    facts = list(range(4))
    rules: list[tuple[list[int], int]] = []
    needed: list[int] = []
    atom = 4
    cursor = rng.choice(facts)

    def insert(pre, relevant):
        nonlocal atom
        out = atom
        atom += 1
        rules.append((list(pre), out))
        if relevant:
            needed.append(len(rules) - 1)
        return out

    for j in range(depth):
        if family in ("fork", "mix") and j > 0 and (family == "fork" or rng.random() < 0.4):
            side = insert([rng.choice(facts)], True)
            cursor = insert([cursor, side], True)
        else:
            cursor = insert([cursor], True)
    goal = cursor
    for _ in range(depth + rng.randrange(2, 5)):
        available = list(range(4, atom))
        choices = facts + available
        pre = [rng.choice(choices)]
        if rng.random() < 0.25:
            pre.append(rng.choice(choices))
        insert(pre, False)
    rule_perm = list(range(len(rules)))
    rng.shuffle(rule_perm)
    inv = {old: new for new, old in enumerate(rule_perm)}
    rules = [rules[i] for i in rule_perm]
    # Alpha renaming is part of the curriculum, and atom identifiers have no meaning.
    alpha = list(range(atom))
    rng.shuffle(alpha)
    return Problem([alpha[a] for a in facts],
                   [([alpha[a] for a in pre], alpha[out]) for pre, out in rules],
                   alpha[goal], [inv[i] for i in needed], depth, family,
                   f"thm_{seed}")


def applicable(p: Problem, known: set[int]) -> list[int]:
    return [i for i, (pre, out) in enumerate(p.rules)
            if out not in known and all(x in known for x in pre)]


def features(p: Problem, known: set[int], max_rules: int, device="cpu"):
    n = len(p.rules)
    if n > max_rules:
        raise ValueError(f"{n} rules exceed --max-rules={max_rules}")
    x = torch.zeros(max_rules, FEATURES, dtype=torch.float32)
    att = torch.eye(max_rules, dtype=torch.bool)
    legal = torch.zeros(max_rules, dtype=torch.bool)
    for i, (pre, out) in enumerate(p.rules):
        vals = [int(a in known) for a in pre]
        x[i] = torch.tensor([sum(vals)/len(vals), float(all(vals)),
                             float(out in known), float(out == p.goal),
                             len(pre)/2, vals[0], vals[1] if len(vals) == 2 else 0,
                             float(len(pre) == 1), 1.0])
        legal[i] = all(vals) and out not in known
        for j, (other, _) in enumerate(p.rules):
            if out in other:
                att[i, j] = True
    return x.to(device), att.to(device), legal.to(device)


def teacher_states(p: Problem):
    known = set(p.facts)
    out = []
    for action in p.teacher:
        legal = applicable(p, known)
        correct = sorted(set(legal).intersection(p.teacher))
        if action not in correct:
            raise AssertionError("Generated proof is ill-ordered")
        out.append((known.copy(), correct))
        known.add(p.rules[action][1])
    assert p.goal in known
    return out


def batch_problems(ps: list[Problem], max_rules: int, rng: random.Random, device):
    xs, masks, legals, positives = [], [], [], []
    for p in ps:
        state, correct = rng.choice(teacher_states(p))
        x, mask, legal = features(p, state, max_rules, device)
        pos = torch.zeros(max_rules, dtype=torch.bool, device=device)
        pos[correct] = True
        xs.append(x); masks.append(mask); legals.append(legal); positives.append(pos)
    return tuple(torch.stack(a) for a in (xs, masks, legals, positives))


class GraphBlock(nn.Module):
    def __init__(self, width: int, heads: int = 8):
        super().__init__()
        assert width % heads == 0
        self.width = width
        self.heads = heads
        self.n1, self.n2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3*width)
        self.proj = nn.Linear(width, width)
        self.up, self.down = nn.Linear(width, 4*width), nn.Linear(4*width, width)

    def forward(self, z, allowed):
        b, n, d = z.shape
        q, k, v = self.qkv(self.n1(z)).chunk(3, -1)
        def split(y):
            return y.view(b, n, self.heads, d // self.heads).transpose(1, 2)
        q, k, v = map(split, (q, k, v))
        sim = (q @ k.transpose(-1, -2)) * ((d // self.heads) ** -0.5)
        sim = sim.masked_fill(~allowed[:, None], -1e4)
        z = z + self.proj((sim.softmax(-1) @ v).transpose(1, 2).reshape(b,n,d))
        return z + self.down(F.gelu(self.up(self.n2(z))))


class TwoClockHRM(nn.Module):
    def __init__(self, width=64, fast_steps=2, slow_steps=2):
        super().__init__()
        self.width, self.fast_steps, self.slow_steps = width, fast_steps, slow_steps
        self.embed = nn.Linear(FEATURES, width)
        self.fast, self.slow = GraphBlock(width), GraphBlock(width)
        self.gate = nn.Parameter(torch.tensor(-2.0))
        self.readout = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))
        self.halt = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))

    def forward(self, x, edges, legal):
        h = self.embed(x)
        z = torch.zeros_like(h)
        for _ in range(self.slow_steps):
            for _ in range(self.fast_steps):
                h = self.fast(h + torch.sigmoid(self.gate) * z, edges)
            z = self.slow(z + h, edges)
        logits = self.readout(h + z).squeeze(-1)
        return logits.masked_fill(~legal, -1e4)


def set_loss(logits, positives):
    if not bool(positives.any(-1).all()):
        raise ValueError("Missing teacher action")
    return (torch.logsumexp(logits, -1) -
            torch.logsumexp(logits.masked_fill(~positives, -1e4), -1)).mean()


def choose_device(name):
    return ("cuda" if torch.cuda.is_available() else "cpu") if name == "auto" else name


def memory_info():
    try:
        import psutil
        return {"available_ram_gib": round(psutil.virtual_memory().available / 2**30, 2)}
    except ImportError:
        return {}


def save_checkpoint(path: Path, model, optimizer, config, step, best):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".part")
    torch.save({"weights": model.state_dict(), "optimizer": optimizer.state_dict(),
                "config": config, "step": step, "best_validation": best,
                "torch_rng": torch.get_rng_state()}, temp)
    os.replace(temp, path)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sample_batch(args, step, rng, device, opt_step=None):
    lo = 2
    # Audit fix: the curriculum clock is the optimizer step. v1 passed the micro-batch
    # index (step*accumulate+accum), unlocking depth `accumulate` times too early.
    clock = step if opt_step is None else opt_step
    hi = min(args.max_train_depth, 2 + clock // max(1, args.depth_step_interval))
    ps = [make_problem(args.seed + step*1_000_003 + j*101, rng.randint(lo,hi),
                       rng.choice(("chain", "fork", "mix"))) for j in range(args.batch)]
    return batch_problems(ps,args.max_rules,rng,device)


@torch.no_grad()
def evaluate(model, *, seed, n, max_rules, depth_min, depth_max, device,
             mode="greedy", branches=3, max_steps=40, budget_slack=1):
    model.eval()
    solved = 0
    calls = 0
    bad = 0
    for i in range(n):
        p = make_problem(seed + 104729*i, depth_min + i%(depth_max-depth_min+1),
                         ("chain", "fork", "mix")[i%3])
        known = set(p.facts)
        # Fair proof-search budget: the known witness length plus one extra step.
        # Unbounded forward closure makes every reachable graph look solved.
        budget = min(max_steps, len(p.teacher) + budget_slack)
        for turn in range(budget):
            if p.goal in known:
                break
            x, mask, legal = features(p,known,max_rules,device)
            if not legal.any():
                break
            logits = model(x[None], mask[None], legal[None])[0]
            calls += 1
            if mode == "cgrt":
                # Prefix is fixed. Generate cross-context continuations by renaming
                # atom labels and permuting rule order; average mapped predictions.
                # This tests context stability, not full CGRT tree search.
                scores = logits.clone()
                for j in range(1,branches):
                    rnd = random.Random(seed + i*1009 + turn*37 + j)
                    perm = list(range(len(p.rules))); rnd.shuffle(perm)
                    pp = Problem(p.facts, [p.rules[k] for k in perm],p.goal,
                                 [],p.depth,p.family,p.name)
                    xx,mm,ll = features(pp,known,max_rules,device)
                    values = model(xx[None],mm[None],ll[None])[0]
                    calls += 1
                    for new, old in enumerate(perm):
                        scores[old] += values[new]
                action = int(scores.argmax().item())
            else:
                action = int(logits.argmax().item())
            if action not in applicable(p,known):
                bad += 1
                break
            known.add(p.rules[action][1])
        solved += p.goal in known
    return {"solved": solved,"total": n,"rate": solved/n,
            "model_calls": calls,"calls_per_solved": calls/max(1,solved),
            "invalid_actions": bad,"mode": mode,"kernel_checked": False,
            "cgrt_note": ("rule-order permutations are a symmetry of TwoClockHRM; branch scores are "
                          "identical to greedy (audit/v1_probes.py). Superseded by lcgrt.search.") if mode == "cgrt" else None,
            "budget_policy": "teacher_length_plus_slack", "budget_slack": budget_slack}


def train(args):
    device = choose_device(args.device)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    cfg = {"width": PRESETS[args.preset], "fast_steps": args.fast_steps,
           "slow_steps": args.slow_steps, "seed": args.seed,
           "max_rules": args.max_rules, "max_train_depth":args.max_train_depth,
           "preset": args.preset, "scheme": "synthetic proof-state graph; kernel audit separate",
           # Audit fix: v1 resumed silently with a different lr/batch/schedule.
           "lr": args.lr, "batch": args.batch, "accumulate": args.accumulate,
           "depth_step_interval": args.depth_step_interval, "steps": args.steps}
    model = TwoClockHRM(cfg["width"],args.fast_steps,args.slow_steps).to(device)
    count = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    output = Path(args.out)
    output.mkdir(parents=True,exist_ok=True)
    ckpt = output / "latest.pt"
    first, best = 1, float("inf")
    if args.resume and ckpt.exists():
        loaded = torch.load(ckpt,map_location=device,weights_only=False)
        if loaded["config"] != cfg:
            raise ValueError("Checkpoint configuration changed; choose a new output directory")
        model.load_state_dict(loaded["weights"])
        optimizer.load_state_dict(loaded["optimizer"])
        first, best = loaded["step"]+1,loaded["best_validation"]
        torch.set_rng_state(loaded["torch_rng"].cpu())
    rng = random.Random(args.seed + 397)
    # The per-step seed scheme gives stable problem identities across restarts.
    history = output / "metrics.jsonl"
    (output / "config.json").write_text(json.dumps({**cfg,"parameters":count,"device":device,
                                                     "memory":memory_info()},indent=2)+"\n")
    print(json.dumps({"event":"start","parameters":count,"step":first,
                      "target":args.steps,"device":device,"memory":memory_info()}),flush=True)
    for step in range(first,args.steps+1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total = 0.
        for accum in range(args.accumulate):
            batch_rng = random.Random(args.seed + step*1019 + accum*117)
            x,m,l,p = sample_batch(args,step*args.accumulate+accum,batch_rng,device,opt_step=step)
            loss = set_loss(model(x,m,l),p) / args.accumulate
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            loss.backward()
            total += float(loss.detach().item())
        nn.utils.clip_grad_norm_(model.parameters(),1.0)
        optimizer.step()
        if step % args.save_every == 0 or step == args.steps:
            save_checkpoint(ckpt,model,optimizer,cfg,step,best)
        if step % args.eval_every == 0 or step == args.steps:
            val = evaluate(model,seed=73_000_000,n=args.eval_count,max_rules=args.max_rules,
                           depth_min=2,depth_max=args.max_train_depth,device=device)
            ood = evaluate(model,seed=83_000_000,n=args.eval_count,max_rules=args.max_rules,
                           depth_min=args.max_train_depth+1,depth_max=args.max_train_depth+2,device=device)
            row = {"step":step,"loss":total,"validation":val,"depth_ood":ood,
                   "unix":int(time.time()),"checkpoint":str(ckpt)}
            with history.open("a",encoding="utf-8") as f:
                f.write(json.dumps(row)+"\n")
            print(json.dumps(row),flush=True)
    print(json.dumps({"event":"complete","checkpoint":str(ckpt),
                      "sha256":sha256(ckpt)}))


def lean_source(problems):
    lines = ["set_option autoImplicit false", "namespace LocalGym"]
    for p in problems:
        bank = {a: f"f{i}" for i,a in enumerate(p.facts)}
        for action in p.teacher:
            pre, out = p.rules[action]
            if any(a not in bank for a in pre):
                raise AssertionError("Invalid teacher trace")
            bank[out] = "(" + " ".join([f"r{action}"] + [bank[a] for a in pre]) + ")"
        atoms = sorted(set(p.facts + [p.goal] + [a for pre,out in p.rules for a in pre+[out]]))
        fargs = " ".join(f"(f{i} : P{a})" for i,a in enumerate(p.facts))
        def rt(pre,out):
            return " → ".join([f"P{a}" for a in pre] + [f"P{out}"])
        rargs = " ".join(f"(r{i} : {rt(pre,out)})" for i,(pre,out) in enumerate(p.rules))
        lines += [f"theorem {p.name} ({' '.join('P'+str(a) for a in atoms)} : Prop) {fargs} {rargs} : P{p.goal} :=",
                  f"  {bank[p.goal]}"]
    return "\n".join(lines+["end LocalGym", ""])


def generate(args):
    path = Path(args.out);path.mkdir(parents=True,exist_ok=True)
    problems = [make_problem(args.seed+i*104729,2+i%max(1,args.max_depth-1),
                             ("chain","fork","mix")[i%3]) for i in range(args.count)]
    # A source file is a candidate for Lean checking until the kernel audit passes.
    source=path/"Generated.lean";source.write_text(lean_source(problems),encoding="utf-8")
    dataset=path/"problems.jsonl"
    dataset.write_text("".join(json.dumps(asdict(p))+"\n" for p in problems),encoding="utf-8")
    manifest={"format":"lean-cgrt-local-v1", "count":len(problems),
              "source_sha256":sha256(source),"dataset_sha256":sha256(dataset),
              "lean_kernel_checked":False,"source":"generated propositional Horn fragment"}
    (path/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps(manifest))


def audit(args):
    path=Path(args.source)
    receipt=Path(args.receipt);receipt.parent.mkdir(parents=True,exist_ok=True)
    binary=shutil.which(args.lean)
    result={"lean_kernel_checked":False,"source":str(path),"status":"blocked",
            "source_sha256":sha256(path),"reason":"Lean executable unavailable"}
    if not binary:
        receipt.write_text(json.dumps(result,indent=2)+"\n")
        print(json.dumps(result));return 2
    source=path.read_text(encoding="utf-8")
    # Audit fix: v1 matched the literal "axiom " (a tab evaded it), allowed set_option
    # (debug.skipKernelTC disables the kernel) and #eval (which could print a fake
    # "does not depend on any axioms" line and inflate the count).
    forbidden=re.compile(r"\b(sorry|admit|axiom|unsafe|opaque|implemented_by|extern|native_decide|"
                         r"set_option|macro|macro_rules|elab|syntax|notation|run_cmd)\b|#eval|#exit")
    hit=forbidden.search(source.replace("set_option autoImplicit false", ""))
    if hit:
        result.update(status="failed",reason=f"Forbidden proof construct: {hit.group(0)}")
        receipt.write_text(json.dumps(result,indent=2)+"\n");return 1
    names=[s.split()[1] for s in source.splitlines() if s.startswith("theorem ")]
    with tempfile.TemporaryDirectory() as d:
        runfile=Path(d)/"Check.lean"
        runfile.write_text(source + "\n".join(f"#print axioms LocalGym.{n}" for n in names)+"\n",encoding="utf-8")
        try:
            p=subprocess.run([binary,str(runfile)],text=True,capture_output=True,timeout=args.timeout)
            clean={m.group(1) for m in re.finditer(r"^'LocalGym\.([A-Za-z0-9_]+)' does not depend on any axioms$",
                                                   p.stdout, re.M)}
            zeros=len(clean & set(names))
            passed=p.returncode==0 and zeros==len(names) and "sorryAx" not in p.stdout
            result.update(lean_kernel_checked=passed,status="passed" if passed else "failed",
                          reason=None if passed else "Kernel or axiom audit failed",
                          declarations=len(names),zero_axiom_reports=zeros,returncode=p.returncode,
                          stderr=p.stderr[-2000:])
        except subprocess.TimeoutExpired:
            result.update(status="failed",reason="Lean timed out")
    receipt.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result));return 0 if result["lean_kernel_checked"] else 1


def build_parser():
    p=argparse.ArgumentParser(description="Local Lean/CGRT proof-routing gym")
    sub=p.add_subparsers(dest="command",required=True)
    g=sub.add_parser("generate")
    g.add_argument("--out",default="runs/corpus");g.add_argument("--count",type=int,default=32)
    g.add_argument("--max-depth",type=int,default=5);g.add_argument("--seed",type=int,default=41)
    a=sub.add_parser("audit")
    a.add_argument("--source",default="runs/corpus/Generated.lean")
    a.add_argument("--receipt",default="runs/corpus/lean_audit.json")
    a.add_argument("--lean",default="lean");a.add_argument("--timeout",type=int,default=180)
    t=sub.add_parser("train")
    t.add_argument("--preset",choices=PRESETS,default="smoke")
    t.add_argument("--device",choices=("auto","cpu","cuda"),default="auto")
    t.add_argument("--out",default="runs/local")
    t.add_argument("--seed",type=int,default=17);t.add_argument("--steps",type=int,default=200)
    t.add_argument("--batch",type=int,default=4);t.add_argument("--accumulate",type=int,default=1)
    t.add_argument("--lr",type=float,default=0.0003)
    t.add_argument("--fast-steps",type=int,default=2);t.add_argument("--slow-steps",type=int,default=2)
    t.add_argument("--max-rules",type=int,default=48)
    t.add_argument("--max-train-depth",type=int,default=5)
    t.add_argument("--depth-step-interval",type=int,default=500)
    t.add_argument("--eval-every",type=int,default=50);t.add_argument("--save-every",type=int,default=25)
    t.add_argument("--eval-count",type=int,default=24)
    t.add_argument("--threads",type=int,default=2)
    t.add_argument("--resume",action="store_true")
    e=sub.add_parser("eval")
    e.add_argument("--checkpoint",required=True);e.add_argument("--seed",type=int,default=83_000_000)
    e.add_argument("--count",type=int,default=60);e.add_argument("--depth-min",type=int,default=6)
    e.add_argument("--depth-max",type=int,default=7);e.add_argument("--max-rules",type=int,default=48)
    e.add_argument("--mode",choices=("greedy","cgrt"),default="greedy")
    e.add_argument("--branches",type=int,default=3)
    e.add_argument("--budget-slack",type=int,default=1)
    e.add_argument("--device",default="auto")
    return p


def main(argv=None):
    args=build_parser().parse_args(argv)
    if args.command=="generate": generate(args)
    elif args.command=="audit": return audit(args)
    elif args.command=="train": train(args)
    elif args.command=="eval":
        d=choose_device(args.device)
        ck=torch.load(args.checkpoint,map_location=d,weights_only=False)
        cfg=ck["config"]
        model=TwoClockHRM(cfg["width"],cfg["fast_steps"],cfg["slow_steps"]).to(d)
        model.load_state_dict(ck["weights"])
        print(json.dumps(evaluate(model,seed=args.seed,n=args.count,max_rules=args.max_rules,
                                  depth_min=args.depth_min,depth_max=args.depth_max,
                                  mode=args.mode,branches=args.branches,
                                  budget_slack=args.budget_slack,device=d)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
