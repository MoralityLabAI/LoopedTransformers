"""Verifier-backed task families for the INK-FP-AUDIT-02 curriculum.

Reused generators (provenance recorded per task):
  pointer_chase   research_gym.ink_fp_audit_01.core.roots
  storyworld      research_gym.envs.coupled_storyworld.CoupledStoryworldEnv (exact reachability;
                  answers are plans verified by simulation against the modeled rival)
  mcp_retrieval   HybridTRMLDT/MCP/environment.py:make_cases, run in a subprocess because
                  HybridTRMLDT ships its own top-level ``research_gym`` package
New exact generators (no stdlib generator found in DataGoblin/BPD/EvalMath/HybridTRMLDT):
  modular_arith, boolean_parity, dyck, register_overwrite, multislot_copy,
  recurrent_stopping (text proxy for TRM/LDT halting), bpd_nullspace, experiment_design
Unavailable on this CPU-lean path (reason recorded in UNAVAILABLE):
  lean_tactic     DataGoblin rlt_lean_rank needs torch and a Lean-checked pool
"""
from __future__ import annotations

import functools
import json
import os
import random
import subprocess
import sys
from pathlib import Path

from .curriculum import Task, VERIFIERS, register, semantic_id, final_answer  # noqa: F401

GITHUB = Path(__file__).resolve().parents[3]
SUFFIX = "\nFinish with a separate final line of the form ANSWER: {fmt}"
UNAVAILABLE = {"lean_tactic": "DataGoblin rlt_lean_rank/rlt_lean_labels require torch and a "
                              "Lean-checked tactic pool; neither is loaded on the CPU-lean path"}


def _task(family, root, partition, prompt, answer, content, *, verifier="exact_token.v1",
          generator, version="v1", source_repo="LoopedTransformers/ink_fp_audit_02 (new)",
          difficulty=0.5, cost_ms=0.05, certificate=None):
    return Task(task_id=f"{family}-{semantic_id(family, content)}", family=family, prompt=prompt,
                answer=str(answer), source_root=root, source_world=f"{family}:{root}",
                source_repo=source_repo, generator=generator, generator_version=version,
                difficulty=difficulty, verifier=verifier, verify_cost_ms=cost_ms,
                partition=partition, certificate=certificate or {})


@register("modular_arith")
def modular_arith(root, partition):
    r = random.Random(root)
    m = r.choice([7, 11, 13, 17, 19, 23, 29])
    terms = [(r.choice("+-*"), r.randrange(2, 60)) for _ in range(r.randint(3, 6))]
    x = r.randrange(1, 60)
    expr, val = str(x), x % m
    for op, v in terms:
        expr = f"({expr} {op} {v})"
        val = (val + v) % m if op == "+" else (val - v) % m if op == "-" else (val * v) % m
    prompt = f"Compute {expr} mod {m}. Give the least nonnegative residue." + SUFFIX.format(fmt="<integer>")
    return _task("modular_arith", root, partition, prompt, val, [expr, m], verifier="exact_int.v1",
                 generator="ink02.gyms.modular_arith", difficulty=len(terms) / 6)


@register("boolean_parity")
def boolean_parity(root, partition):
    r = random.Random(root)
    n = r.randint(4, 8)
    names = r.sample("ABCDEFGHJKLMNPQR", n)
    vals = {v: r.randint(0, 1) for v in names}
    rules, derived = [], dict(vals)
    for i in range(r.randint(2, 4)):
        a, b = r.sample(sorted(derived), 2)
        op = r.choice(["XOR", "AND", "OR"])
        name = f"Z{i}"
        derived[name] = derived[a] ^ derived[b] if op == "XOR" else derived[a] & derived[b] if op == "AND" else derived[a] | derived[b]
        rules.append(f"{name} = {a} {op} {b}")
    target = r.sample(sorted(derived), r.randint(3, min(6, len(derived))))
    parity = sum(derived[t] for t in target) % 2
    facts = ", ".join(f"{k}={v}" for k, v in vals.items())
    prompt = (f"Boolean world: {facts}. Rules: {'; '.join(rules)}. What is the parity (XOR) of "
              f"{', '.join(target)}? Answer 0 or 1." + SUFFIX.format(fmt="<0 or 1>"))
    return _task("boolean_parity", root, partition, prompt, parity, [facts, rules, target],
                 verifier="exact_int.v1", generator="ink02.gyms.boolean_parity", difficulty=len(target) / 6)


PAIRS = {"(": ")", "[": "]", "{": "}", "<": ">"}


@register("dyck")
def dyck(root, partition):
    r = random.Random(root)
    depth_cap = r.randint(3, 6)
    s, stack = [], []
    for _ in range(r.randint(6, 18)):
        if stack and (len(stack) >= depth_cap or r.random() < 0.45):
            s.append(PAIRS[stack.pop()])
        else:
            o = r.choice(list(PAIRS))
            stack.append(o)
            s.append(o)
    if r.random() < 0.3 and s:  # corrupt: insert a mismatched closer
        i = r.randrange(len(s) + 1)
        s.insert(i, r.choice(list(PAIRS.values())))
    text = "".join(s)
    st, valid = [], True
    for ch in text:
        if ch in PAIRS:
            st.append(ch)
        elif not st or PAIRS[st.pop()] != ch:
            valid = False
            break
    answer = "".join(PAIRS[c] for c in reversed(st)) if valid else "INVALID"
    answer = answer or "EMPTY"
    prompt = (f"Bracket prefix: {text}\nIf the prefix can be completed into a balanced string, give the "
              "shortest closing suffix; if it is already balanced answer EMPTY; if it can never be "
              "balanced answer INVALID." + SUFFIX.format(fmt="<suffix | EMPTY | INVALID>"))
    return _task("dyck", root, partition, prompt, answer, text, generator="ink02.gyms.dyck",
                 difficulty=len(text) / 19)


@register("pointer_chase")
def pointer_chase(root, partition):
    from research_gym.ink_fp_audit_01.core import roots
    row = roots(seed=root, n=1)[0]
    return _task("pointer_chase", root, partition, row["prompt"], row["answer"], row["prompt"],
                 verifier="exact_int.v1", generator="research_gym.ink_fp_audit_01.core.roots",
                 source_repo="LoopedTransformers/ink_fp_audit_01")


@register("register_overwrite")
def register_overwrite(root, partition):
    r = random.Random(root)
    regs = r.sample(["r0", "r1", "r2", "r3", "r4", "r5"], r.randint(3, 5))
    state = {k: r.randrange(0, 20) for k in regs}
    lines = [f"{k} = {v}" for k, v in state.items()]
    for _ in range(r.randint(4, 10)):
        dst, src = r.choice(regs), r.choice(regs)
        kind = r.choice(["copy", "add", "set"])
        if kind == "copy":
            state[dst] = state[src]
            lines.append(f"{dst} = {src}")
        elif kind == "add":
            c = r.randrange(1, 9)
            state[dst] = state[src] + c
            lines.append(f"{dst} = {src} + {c}")
        else:
            c = r.randrange(0, 20)
            state[dst] = c
            lines.append(f"{dst} = {c}")
    q = r.choice(regs)
    prompt = ("Execute in order:\n" + "\n".join(lines) + f"\nWhat is the final value of {q}?"
              + SUFFIX.format(fmt="<integer>"))
    return _task("register_overwrite", root, partition, prompt, state[q], [lines, q],
                 verifier="exact_int.v1", generator="ink02.gyms.register_overwrite", difficulty=len(lines) / 15)


@register("multislot_copy")
def multislot_copy(root, partition):
    r = random.Random(root)
    words = ["amber", "basil", "cedar", "delta", "ember", "fjord", "garnet", "harbor", "indigo",
             "juniper", "kelp", "lumen", "mosaic", "nectar", "onyx", "pylon"]
    slots = r.sample(range(1, 20), r.randint(4, 8))
    mem = {s: r.choice(words) for s in slots}
    ops = [f"store {w} in slot {s}" for s, w in mem.items()]
    for _ in range(r.randint(1, 4)):
        a, b = r.sample(slots, 2)
        mem[b] = mem[a]
        ops.append(f"copy slot {a} to slot {b}")
    qs = r.sample(slots, r.randint(2, 4))
    prompt = ("Memory operations:\n" + "\n".join(ops) + f"\nList the contents of slots {', '.join(map(str, qs))} "
              "in that order, comma separated." + SUFFIX.format(fmt="<word>,<word>,..."))
    return _task("multislot_copy", root, partition, prompt, ",".join(mem[q] for q in qs), [ops, qs],
                 generator="ink02.gyms.multislot_copy", difficulty=len(ops) / 12)


@register("recurrent_stopping")
def recurrent_stopping(root, partition):
    """Text proxy for TRM/LDT halting: iterate an affine map mod m from x0, stop at first repeat."""
    r = random.Random(root)
    m = r.choice([11, 13, 17, 19, 23])
    a, b, x = r.randrange(1, m), r.randrange(0, m), r.randrange(0, m)
    seen, steps = {}, 0
    while x not in seen:
        seen[x] = steps
        x = (a * x + b) % m
        steps += 1
    prompt = (f"Start with x = {list(seen)[0]}. Repeatedly replace x by ({a}*x + {b}) mod {m}. Stop as soon "
              "as x takes a value it has taken before. How many replacements were applied when you stop?"
              + SUFFIX.format(fmt="<integer>"))
    return _task("recurrent_stopping", root, partition, prompt, steps, [a, b, m, list(seen)[0]],
                 verifier="exact_int.v1", generator="ink02.gyms.recurrent_stopping", difficulty=min(1, steps / 20),
                 certificate={"cycle_entry_step": seen[x], "halt_step": steps})


@register("bpd_nullspace")
def bpd_nullspace(root, partition):
    """Which candidate update directions keep the protected rows unchanged (X_keep d = 0)?"""
    r = random.Random(root)
    rows = [[r.randint(-3, 3) for _ in range(4)] for _ in range(2)]
    opts, letters = {}, "ABCDE"
    for L in letters:
        if r.random() < 0.5:  # build an exact null vector: d = u x v style combination for 2x4
            a, b = r.randint(-2, 2), r.randint(-2, 2)
            # Generic null vector via Cramer-style construction on coordinates (2,3) free.
            x1, x2 = rows[0], rows[1]
            det = x1[0] * x2[1] - x1[1] * x2[0]
            if det == 0:
                d = [r.randint(-3, 3) for _ in range(4)]
            else:
                c0 = -(a * x1[2] + b * x1[3])
                c1 = -(a * x2[2] + b * x2[3])
                d0n = c0 * x2[1] - x1[1] * c1
                d1n = x1[0] * c1 - c0 * x2[0]
                d = [d0n, d1n, a * det, b * det]
        else:
            d = [r.randint(-3, 3) for _ in range(4)]
        opts[L] = d
    keep = sorted(L for L, d in opts.items() if any(d) and all(sum(x * y for x, y in zip(row, d)) == 0 for row in rows))
    answer = ",".join(keep) if keep else "NONE"
    desc = "\n".join(f"{L}: {d}" for L, d in opts.items())
    prompt = (f"Protected input rows X_keep = {rows}. An update direction d preserves them iff X_keep d = 0 "
              f"and d is nonzero. Candidates:\n{desc}\nWhich candidates preserve the protected rows?"
              + SUFFIX.format(fmt="<letters comma separated, or NONE>"))
    return _task("bpd_nullspace", root, partition, prompt, answer, [rows, opts], verifier="exact_set.v1",
                 generator="ink02.gyms.bpd_nullspace", certificate={"products": {L: [sum(x * y for x, y in zip(row, d)) for row in rows] for L, d in opts.items()}})


@register("experiment_design")
def experiment_design(root, partition):
    """Identify the arm that breaks the matched-budget comparison."""
    r = random.Random(root)
    letters = "ABCDE"[: r.randint(3, 5)]
    base = {"steps": r.choice([100, 200, 400]), "rank": r.choice([8, 16, 32]), "tokens": r.choice([2, 4, 8]) * 10000,
            "roots": r.choice([24, 48, 64])}
    bad = r.choice(letters)
    field = r.choice(sorted(base))
    arms = []
    for L in letters:
        a = dict(base)
        if L == bad:
            a[field] = a[field] * r.choice([2, 4]) if r.random() < 0.5 else max(1, a[field] // 2)
        arms.append(f"arm {L}: optimizer steps {a['steps']}, adapter rank {a['rank']}, training tokens "
                    f"{a['tokens']}, independent roots {a['roots']}")
    prompt = ("A study compares training methods and claims budgets are matched.\n" + "\n".join(arms)
              + "\nWhich arm violates the matched budget? If none does, answer NONE." + SUFFIX.format(fmt="<letter | NONE>"))
    return _task("experiment_design", root, partition, prompt, bad, [arms], generator="ink02.gyms.experiment_design",
                 certificate={"deviating_field": field})


# ---- storyworld (reused env) -------------------------------------------------------------
STORY_RULES = (
    "State (trust in [-3,3], evidence in [0,4], heat in [0,4], scene in [0,5]). Each turn applies your action, "
    "then the rival's, then the scene update, and clamps to the bounds only at the end of the turn. "
    "Your actions: befriend (trust+1, heat-1 floored at 0), investigate (evidence+1, heat+1, then trust-1 if "
    "heat>=2), defuse (heat-2), rush (scene+1, heat+1), wait (heat-1 floored at 0). The rival then acts on the "
    "state before your action: rival_sabotage (trust-1, heat+1) if evidence>=2 and heat<=3; else rival_flatter "
    "(trust+1, evidence-1 floored at 0) if trust<=-1 and evidence>=1; else rival_wait. Then scene+1 unless you "
    "waited. The game ends when scene reaches 5. Secret ending: scene>=5, trust>=1, "
    "evidence>=3, heat<=2.")


def storyworld_check(task: Task, answer: str) -> bool:
    from research_gym.envs.coupled_storyworld import CoupledStoryworldEnv, StoryState
    cert = task.certificate
    if not cert["reachable"]:
        return answer.strip().upper() == "NONE"
    env = CoupledStoryworldEnv()
    s = StoryState(**cert["start"])
    plan = [a.strip().lower() for a in answer.split(",") if a.strip()]
    if not plan or len(plan) > cert["horizon"] or any(a not in env.self_actions for a in plan):
        return False
    for a in plan:
        if env.terminal(s):
            return False
        s = env.step(s, a, env.other_policy(s))
        if env.target(s):
            return True
    return False


VERIFIERS["storyworld_plan.v1"] = storyworld_check


@register("storyworld")
def storyworld(root, partition):
    from research_gym.envs.coupled_storyworld import CoupledStoryworldEnv, StoryState
    r = random.Random(root)
    env = CoupledStoryworldEnv()
    want = r.random() < 0.5  # balance reachable / unreachable starts
    for _ in range(64):
        s = StoryState(trust=r.randint(-2, 2), evidence=r.randint(0, 3), heat=r.randint(0, 3), scene=r.randint(0, 2))
        horizon = r.randint(3, 6)
        ok, path = env.reachable(s, horizon, other_mode="model")
        if ok == want:
            break
    prompt = (STORY_RULES + f"\nStart: {s.to_dict()}. Within at most {horizon} of your actions, give a plan "
              "that reaches the secret ending, as comma-separated action names, or NONE if impossible."
              + SUFFIX.format(fmt="<action,action,... | NONE>"))
    witness = ",".join(p.split("/")[0] for p in path) if ok else "NONE"
    return _task("storyworld", root, partition, prompt, witness, [s.to_dict(), horizon],
                 verifier="storyworld_plan.v1", generator="research_gym.envs.coupled_storyworld.CoupledStoryworldEnv.reachable",
                 source_repo="LoopedTransformers/research_gym/envs/coupled_storyworld.py",
                 certificate={"reachable": ok, "start": s.to_dict(), "horizon": horizon, "witness": witness},
                 cost_ms=1.0)


# ---- MCP retrieval (HybridTRMLDT, subprocess) -----------------------------------------------
_MCP_CACHE: dict[tuple, dict] = {}
_MCP_CODE = r"""
import json, sys
sys.path.insert(0, sys.argv[1])
from MCP.environment import make_cases
split, roots = sys.argv[2], json.loads(sys.argv[3])
out = {}
for root in roots:
    case = make_cases(split=split, seed=root, n=1)[0]
    out[str(root)] = {k: case[k] for k in ("id", "family", "group_id", "task", "corpus", "oracle", "provenance")}
print(json.dumps(out))
"""


def mcp_prefetch(roots: list[int], split: str) -> None:
    need = [x for x in roots if (split, x) not in _MCP_CACHE]
    if not need:
        return
    repo = GITHUB / "HybridTRMLDT"
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "-1", "PYTHONPATH": str(repo)}
    out = subprocess.run([sys.executable, "-c", _MCP_CODE, str(repo), split, json.dumps(need)],
                         capture_output=True, text=True, cwd=str(repo), env=env, timeout=300, check=True)
    for k, v in json.loads(out.stdout).items():
        _MCP_CACHE[(split, int(k))] = v


def _mcp_split(partition):
    return {"train": "train", "dev": "dev", "accept": "dev", "sealed": "test"}[partition]


@functools.lru_cache(maxsize=1)
def mcp_commit() -> str:
    """HybridTRMLDT checkout commit, read from .git without running git."""
    git = GITHUB / "HybridTRMLDT" / ".git"
    try:
        head = (git / "HEAD").read_text(encoding="utf-8").strip()
        if head.startswith("ref: "):
            ref = head[5:]
            p = git / ref
            if p.exists():
                return p.read_text(encoding="utf-8").strip()
            for line in (git / "packed-refs").read_text(encoding="utf-8").splitlines():
                if line.endswith(" " + ref):
                    return line.split()[0]
        return head
    except OSError:
        return "unknown"


@register("mcp_retrieval")
def mcp_retrieval(root, partition):
    split = _mcp_split(partition)
    mcp_prefetch([root], split)
    case = _MCP_CACHE[(split, root)]
    rows = [{k: row[k] for k in ("ref", "entity", "field", "revision", "scope", "epoch", "unit", "value")}
            for row in case["corpus"]]
    oracle = case["oracle"]
    answer = "ABSTAIN" if oracle["abstain"] else str(oracle["answer"])
    prompt = ("Records (JSON lines):\n" + "\n".join(json.dumps(x, separators=(",", ":")) for x in rows)
              + "\n\nRequest: " + case["task"]["request"]
              + "\nGive the requested number, or ABSTAIN if a required reading is missing."
              + SUFFIX.format(fmt="<integer | ABSTAIN>"))
    return _task("mcp_retrieval", root, partition, prompt, answer, [case["group_id"]],
                 generator="HybridTRMLDT/MCP/environment.py:make_cases", version="mcp-fixture-v1",
                 source_repo="HybridTRMLDT@" + mcp_commit()[:12], difficulty=0.6, cost_ms=0.1,
                 certificate={"required_refs": oracle["required_refs"], "mcp_family": case["family"]})



# ---- hill-climb research judgment (rule-verified) ----------------------------------------------
JUDGE_LABELS = ("ACCEPT", "REJECT primary_gain", "REJECT family_regression", "REJECT repetition",
                "REJECT completion")


def judgment_label(decision: dict) -> str:
    if decision["accepted"]:
        return "ACCEPT"
    for check, name in (("primary_gain_ok", "primary_gain"), ("family_regression_ok", "family_regression"),
                        ("repetition_ok", "repetition"), ("completion_ok", "completion")):
        if not decision["checks"][check]:
            return "REJECT " + name


@register("hillclimb_judgment")
def hillclimb_judgment(root, partition):
    """Decide a generation-acceptance step under the frozen INK-FP-AUDIT-02 criteria."""
    from . import acceptance as acc
    r = random.Random(root)
    want = r.choice(JUDGE_LABELS[:1] * 2 + JUDGE_LABELS[1:])
    crit = acc.Criteria()
    names = r.sample(["dyck", "parity", "pointer", "storyworld", "mcp", "registers", "modular"], r.randint(2, 4))
    for _ in range(400):
        fams = {f: r.choice([6, 8, 10, 12]) for f in names}
        pred, cand = [], []
        for f, n in fams.items():
            for i in range(n):
                tid = f"{f}-{i}"
                pc, cc = r.random() < 0.5, r.random() < 0.5 + r.uniform(-0.2, 0.3)
                comp_p, comp_c = r.random() < 0.95, r.random() < 0.93
                rep_p, rep_c = r.random() < 0.02, r.random() < 0.04
                pred.append({"task_id": tid, "family": f, "correct": pc and comp_p, "completed": comp_p, "repetition": rep_p})
                cand.append({"task_id": tid, "family": f, "correct": cc and comp_c, "completed": comp_c, "repetition": rep_c})
        d = acc.decide(crit, pred, cand)
        if judgment_label(d) == want:
            break
    label = judgment_label(d)
    P, C = d["predecessor_summary"], d["candidate_summary"]
    rows = "\n".join(f"  {f}: n={P['families'][f]['n']}, incumbent correct {P['families'][f]['correct']}, "
                      f"candidate correct {C['families'][f]['correct']}" for f in fams)
    prompt = (
        "A recursive self-improvement run evaluates a candidate adapter against the incumbent on the same paired tasks.\n"
        f"Per family:\n{rows}\n"
        f"Totals: n={P['n']}; correct {P['correct']} -> {C['correct']}; repetition rate {P['repetition_rate']:.4f} -> "
        f"{C['repetition_rate']:.4f}; completion rate {P['completion_rate']:.4f} -> {C['completion_rate']:.4f}.\n"
        "Frozen rule, checked in this order: (1) primary_gain: candidate total correct must exceed the incumbent's by at "
        "least 1. (2) family_regression: in each family the candidate may lose at most max(1, ceil(0.10 * n_family)) "
        "correct answers relative to the incumbent. (3) repetition: repetition rate may rise by at most 0.02. "
        "(4) completion: completion rate may drop by at most 0.02. Accept only if all four pass; otherwise report the "
        "first failing check." + SUFFIX.format(fmt="ACCEPT | REJECT <primary_gain|family_regression|repetition|completion>"))
    return _task("hillclimb_judgment", root, partition, prompt, label,
                 [P["families"], C["families"], P["repetition_rate"], C["repetition_rate"], P["completion_rate"], C["completion_rate"]],
                 generator="ink02.gyms.hillclimb_judgment", certificate={"checks": d["checks"], "family_losses": d["family_losses"],
                 "family_limits": d["family_limits"], "primary_gain": d["primary_gain"]}, difficulty=0.6)

FAMILIES = ("modular_arith", "boolean_parity", "dyck", "pointer_chase", "register_overwrite",
            "multislot_copy", "mcp_retrieval", "storyworld", "recurrent_stopping", "bpd_nullspace",
            "experiment_design", "hillclimb_judgment")
