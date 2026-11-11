"""Rapid gym: short training runs, one Lean-certified dev pass per round, a ledger, the next round.

    python -m lcgrt.rapid                       # runs rounds until D:/lean_cgrt/runs/rapid/STOP exists
    python -m lcgrt.rapid --ledger-only         # rebuild ledger.jsonl / ledger.md from finished rounds

Exploratory, not confirmatory. Models train on mathlib-c3 `train` and are scored on a fixed
sample of `val` states, never a test split, so registered campaigns (c4) stay clean.
- A round trains its specs one after another on the GPU (one lease, `cpugym --lanes train`).
- It then scores all of them in one `mathlib_eval` job on the CPU, so each Mathlib file is
  elaborated once per round. Lean's limits are heartbeats, so the verdicts do not depend on load.
- The ledger keeps each spec's config, budget (steps, fresh samples), best val bits/byte and dev
  close@k (model alone and interleaved with automation).

The next round is chosen from the ledger:
- The better half of the last round survives, ranked by dev close@8 (ties: lower val bits/byte).
  Survivors move to twice the steps, up to --max-steps, and to the next seed.
- The rest of the round is one-knob mutations of the best spec at the same budget and seed, not
  tried at that budget before. So every mutant is paired with its parent's rerun in the same round.
- `knob_effects` in ledger.md is the mean paired difference (mutant - parent) per knob and value.
- An arm (preset, cfg, lr) never scored, because its training or evaluation failed, comes back in
  the next round at that round's budget and seed, after the survivors.
Failed training is retried once per round; what still fails is listed in ledger.md.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

from . import config

ROOT = config.RUN_ROOT / "rapid"
MANIFEST = "D:/lean_cgrt/data/mathlib-full/manifest.json"
SEEDS = (17, 29, 43, 59, 71, 83, 97, 101)
DEFAULTS = {"lr": 5e-4, "steps": 1000}
# Round 1 asks what c4 left open at the XS budget: does TRM's deep supervision cost more in fresh
# samples than it buys, and does recursion depth matter, next to flat and single-loop encoders.
ROUND1 = [{"preset": "tlm-flat-XS"}, {"preset": "tlm-loop-XS"}, {"preset": "tlm-trm-XS"},
          {"preset": "tlm-trm-XS", "cfg": {"trm_sup": 4}}, {"preset": "tlm-trm-XS", "cfg": {"trm_sup": 1}},
          {"preset": "tlm-trm-XS", "cfg": {"trm_n": 2, "trm_T": 1, "trm_sup": 4}}]
KNOBS = {"trm": {"trm_sup": (1, 2, 4, 8, 16), "trm_n": (2, 4, 6), "trm_T": (1, 2, 3), "trm_ema": (0.0, 0.999),
                 "lr": (2.5e-4, 5e-4, 1e-3)},
         "flat": {"lr": (2.5e-4, 5e-4, 1e-3)}, "loop": {"lr": (2.5e-4, 5e-4, 1e-3)}}


def full(spec):
    return {"preset": spec["preset"], "cfg": dict(sorted((spec.get("cfg") or {}).items())),
            "lr": spec.get("lr", DEFAULTS["lr"]), "steps": spec.get("steps", DEFAULTS["steps"]),
            "seed": spec.get("seed", SEEDS[0]), **({"parent": spec["parent"], "knob": spec["knob"]} if "knob" in spec else {})}


def arm(spec):
    """A spec without budget or seed: an arm that should be scored at least once."""
    s = full(spec)
    return json.dumps([s["preset"], s["cfg"], s["lr"]])


def key(spec):
    """A spec without its seed: what replication across rounds is grouped by."""
    s = full(spec)
    return json.dumps([s["preset"], s["cfg"], s["lr"], s["steps"]])


def arch(spec):
    from .tactic_lm import PRESETS
    return PRESETS[spec["preset"]].arch


def value(spec, knob):
    from .tactic_lm import PRESETS
    s = full(spec)
    return s["lr"] if knob == "lr" else s["cfg"].get(knob, getattr(PRESETS[s["preset"]], knob))


def name(rnd, spec):
    s = full(spec)
    short = s["preset"].replace("tlm-", "").replace("-", "")
    knobs = "".join(f"_{k.replace('trm_', '')}{v:g}" for k, v in s["cfg"].items())
    return f"r{rnd:02d}_{short}{knobs}_lr{s['lr']:g}_k{s['steps'] // 1000 if s['steps'] % 1000 == 0 else s['steps']}_s{s['seed']}"


def next_specs(last, history, rnd, k, max_steps, retry=()):
    """Round `rnd`'s specs from the last round's ledger rows and every row so far. `retry`: specs
    of arms never scored; they come back at this round's budget and seed, after the survivors."""
    ranked = sorted(last, key=lambda r: (-r["dev_close@8"], r["val_bpb"]))
    seed = SEEDS[(rnd - 1) % len(SEEDS)]
    out, seen = [], {key(r["spec"]) for r in history}
    for r in ranked[:math.ceil(k / 2)]:
        base = {f: v for f, v in r["spec"].items() if f not in ("parent", "knob")}    # a survivor is no one's mutant
        out.append(full(base | {"steps": min(2 * base["steps"], max_steps), "seed": seed}))
    scored = {arm(h["spec"]) for h in history}
    for sp in retry:
        r = full({f: v for f, v in sp.items() if f not in ("parent", "knob")} | {"steps": out[0]["steps"], "seed": seed})
        if arm(r) not in scored and arm(r) not in {arm(x) for x in out}:
            out.append(r)
    best = out[0]
    for knob, values in KNOBS[arch(best)].items():
        for v in values:
            if len(out) >= k:
                return out
            if v == value(best, knob):
                continue
            m = dict(best, cfg=dict(best["cfg"]))
            if knob == "lr":
                m["lr"] = v
            else:
                m["cfg"][knob] = v
            m = full(m | {"parent": key(best), "knob": f"{knob}={v:g}"})
            if key(m) not in seen and key(m) not in {key(x) for x in out}:
                out.append(m)
    return out


def train_line(rnd, spec):
    s = full(spec)
    args = ["--data", "mathlib-c3", "--source", "mathlib-full", "--preset", s["preset"], "--seed", str(s["seed"]),
            "--steps", str(s["steps"]), "--batch", "128", "--eval-every", str(s["steps"]), "--warmup", "100",
            "--lr", f"{s['lr']:g}", "--grad-ckpt"]           # as c3/c4 on the GPU
    if arch(s) == "loop":
        args += ["--random-depth", "4", "12"]                       # c3's loop arm
    if s["cfg"]:
        args += ["--cfg", *[f"{k}={v:g}" for k, v in s["cfg"].items()]]
    # Leases from GPU smoke runs at batch 128 with --grad-ckpt: TRM-XS fits 3G; loop-XS at depth up to 12
    # needs more than 3G and fits 6G. Round 1 ran without checkpointing on 2G and only flat-XS fitted.
    vram = "6G" if arch(s) == "loop" else "3G"
    return {"name": name(rnd, spec), "module": "lcgrt.tactic_lm", "vram": vram, "requires": [MANIFEST], "args": args}


def eval_line(rnd, runs, per_split):
    return {"name": f"r{rnd:02d}_eval", "module": "lcgrt.mathlib_eval",
            "args": ["--data", "mathlib-c3", "--ckpt", *[str(ROOT / r) for r in runs], "--splits", "val",
                     "--per-split", str(per_split), "--per-file", "4", "--beam", "8", "--workers", "3",
                     "--threads", "3", "--proposal-jobs", "3"]}


def read_rounds(root=ROOT):
    f = root / "rounds.jsonl"
    return [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines()] if f.exists() else []


def ledger(root=ROOT):
    """One row per trained spec of every round whose dev evaluation finished."""
    rows = []
    for rd in read_rounds(root):
        ev = root / f"r{rd['round']:02d}_eval" / "eval.json"
        if not ev.exists():
            continue
        summ = json.loads(ev.read_text())["summary"]["all"]
        for spec in rd["specs"]:
            n = name(rd["round"], spec)
            if n not in summ:
                continue
            best = json.loads((root / n / "best.json").read_text())
            done = json.loads((root / n / "complete.json").read_text())
            rows.append({"round": rd["round"], "name": n, "spec": full(spec), "val_bpb": best["bits_per_byte"],
                         "fresh": done.get("fresh") or full(spec)["steps"] * 128, "train_min": round(done["wall_s"] / 60, 1),
                         "certifiable": summ["certifiable"], **{f"dev_{k}": v for k, v in summ[n].items()},
                         **{f"dev+auto_{k}": v for k, v in summ[n + "+auto"].items()},
                         "auto_close@8": summ["auto"]["close@8"], "human_close@1": summ["human"]["close@1"]})
    return rows


def knob_effects(rows):
    by = {(r["round"], key(r["spec"])): r for r in rows}
    eff = {}
    for r in rows:
        if "knob" in r["spec"] and (r["round"], r["spec"]["parent"]) in by:
            p = by[(r["round"], r["spec"]["parent"])]
            eff.setdefault(r["spec"]["knob"], []).append(round(r["dev_close@8"] - p["dev_close@8"], 4))
    return {k: {"n": len(v), "mean_delta_close@8": round(sum(v) / len(v), 4), "deltas": v} for k, v in sorted(eff.items())}


def write_ledger(root=ROOT, copy_to=None):
    rows = ledger(root)
    (root / "ledger.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    groups = {}
    for r in rows:
        groups.setdefault(key(r["spec"]), []).append(r)
    L = ["# Rapid gym ledger", "", "Exploratory: dev = mathlib-c3 `val` states, Lean-certified. Not a registered test.", "",
         "## By spec (seeds pooled)", "", "| spec | runs | dev close@8 | dev+auto close@8 | val bits/byte | fresh |",
         "|---|---|---|---|---|---|"]
    for k, g in sorted(groups.items(), key=lambda kv: -sum(r["dev_close@8"] for r in kv[1]) / len(kv[1])):
        m = lambda f: sum(r[f] for r in g) / len(g)
        L.append(f"| `{k}` | {len(g)} | {m('dev_close@8'):.3f} | {m('dev+auto_close@8'):.3f} | {m('val_bpb'):.4f} | "
                 f"{int(m('fresh') or 0)} |")
    L += ["", "## Knob effects (mutant - parent, same round, budget and seed)", "", "| knob | n | mean delta close@8 |", "|---|---|---|"]
    L += [f"| {k} | {v['n']} | {v['mean_delta_close@8']:+.4f} |" for k, v in knob_effects(rows).items()]
    L += ["", "## Rows", "", "| round | run | dev close@1 | dev close@8 | +auto close@8 | auto close@8 | val bits/byte | train min |",
          "|---|---|---|---|---|---|---|---|"]
    L += [f"| {r['round']} | {r['name']} | {r['dev_close@1']:.3f} | {r['dev_close@8']:.3f} | {r['dev+auto_close@8']:.3f} | "
          f"{r['auto_close@8']:.3f} | {r['val_bpb']:.4f} | {r['train_min']} |" for r in rows]
    scored = {r["name"] for r in rows}
    missing = [(rd["round"], name(rd["round"], sp)) for rd in read_rounds(root) for sp in rd["specs"]
               if name(rd["round"], sp) not in scored and (root / f"r{rd['round']:02d}_eval" / "complete.json").exists()]
    if missing:
        L += ["", "## Not scored", ""]
        L += [f"- round {r}: {n}" + ("" if (root / n / "complete.json").exists() else " (training did not finish)")
              for r, n in missing]
    text = "\n".join(L) + "\n"
    (root / "ledger.md").write_text(text, encoding="utf-8")
    if copy_to:
        Path(copy_to).parent.mkdir(parents=True, exist_ok=True)
        Path(copy_to).write_text(text, encoding="utf-8")
    return rows


def gym(lanes, a):
    caps = ["--ram-cap", a.train_ram, "--gpu-vram", a.gpu_vram, "--gpu-duty", str(a.gpu_duty)] if lanes == "train" \
        else ["--ram-cap", a.eval_ram]
    cmd = [sys.executable, "-u", "-m", "lcgrt.cpugym", "--campaign", "rapid", "--queue", str(ROOT / "queue.jsonl"),
           "--lanes", lanes, *caps, "--status", f"cpugym_status_{lanes}.json"]
    return subprocess.run(cmd, cwd=str(config.PKG_ROOT)).returncode


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=6, help="specs per round")
    ap.add_argument("--max-steps", type=int, default=2000, help="survivors stop doubling here (rounds stay rapid)")
    ap.add_argument("--per-split", type=int, default=120, help="dev states per round")
    ap.add_argument("--rounds", type=int, default=0, help="0 = until STOP exists in the run root")
    ap.add_argument("--train-ram", default="3G")
    ap.add_argument("--gpu-vram", default="6G", help="the gym's VRAM allowance: the largest lease (loop-XS)")
    ap.add_argument("--gpu-duty", type=float, default=0.5)
    ap.add_argument("--eval-ram", default="10G")
    ap.add_argument("--ledger-only", action="store_true")
    ap.add_argument("--copy-ledger", default=str(config.PKG_ROOT / "results" / "rapid" / "ledger.md"))
    a = ap.parse_args(argv)
    ROOT.mkdir(parents=True, exist_ok=True)
    if a.ledger_only:
        write_ledger(copy_to=a.copy_ledger); return 0
    done_rounds = 0
    while not (ROOT / "STOP").exists() and (a.rounds == 0 or done_rounds < a.rounds):
        rounds = read_rounds()
        rows = ledger()
        finished = {r["round"] for r in rows}
        if rounds and rounds[-1]["round"] not in finished:
            rd = rounds[-1]                                            # resume an unfinished round
        else:
            rnd = rounds[-1]["round"] + 1 if rounds else 1
            scored = {arm(r["spec"]) for r in rows}
            retry = list({arm(sp): sp for rd0 in rounds for sp in rd0["specs"] if arm(sp) not in scored}.values())
            specs = [full(s) for s in ROUND1] if rnd == 1 else \
                next_specs([r for r in rows if r["round"] == rnd - 1], rows, rnd, a.k, a.max_steps, retry)
            rd = {"round": rnd, "specs": specs, "unix": int(time.time())}
            with (ROOT / "rounds.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(rd) + "\n")
        rnd, specs = rd["round"], rd["specs"]
        q = ROOT / "queue.jsonl"
        q.write_text("".join(json.dumps(train_line(rnd, s)) + "\n" for s in specs), encoding="utf-8")
        print(json.dumps({"event": "round_train", "round": rnd, "runs": [name(rnd, s) for s in specs]}), flush=True)
        gym("train", a)
        if any(not (ROOT / name(rnd, s) / "complete.json").exists() for s in specs):
            print(json.dumps({"event": "train_retry", "round": rnd}), flush=True)    # a new runner retries failures
            gym("train", a)
        trained =[name(rnd, s) for s in specs if (ROOT / name(rnd, s) / "complete.json").exists()]
        if not trained:
            print(json.dumps({"event": "round_failed", "round": rnd, "reason": "no spec finished training"}), flush=True)
            return 3
        q.write_text(json.dumps(eval_line(rnd, trained, a.per_split)) + "\n", encoding="utf-8")
        print(json.dumps({"event": "round_eval", "round": rnd, "runs": trained}), flush=True)
        gym("eval", a)
        if not (ROOT / f"r{rnd:02d}_eval" / "complete.json").exists():
            print(json.dumps({"event": "round_failed", "round": rnd, "reason": "dev evaluation did not finish"}), flush=True)
            return 3
        rows = write_ledger(copy_to=a.copy_ledger)
        top = max((r for r in rows if r["round"] == rnd), key=lambda r: (r["dev_close@8"], -r["val_bpb"]))
        print(json.dumps({"event": "round_done", "round": rnd, "best": top["name"], "dev_close@8": top["dev_close@8"],
                          "auto_close@8": top["auto_close@8"]}), flush=True)
        done_rounds += 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
