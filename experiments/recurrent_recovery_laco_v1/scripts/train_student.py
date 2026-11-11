"""Train + evaluate one condition.

Conditions (--cond)
  A  : teacher tuning.  Base Qwen3 + LoRA on every layer: CE on synthetic tasks, KL-to-base on natural text
       (keeps knowledge).  LoRA is merged and the tuned teacher is saved -> this is the *teacher transformer*
       that all pruned students are pruned from and distilled against.
  B  : LaCo-pruned student, no training (use --steps 0).
  C  : LaCo-pruned student + ordinary LoRA recovery (every remaining layer, n=1 adapter).
  D  : LaCo-pruned student whose collapsed layer is replaced by a shared recurrent block with loop-specific
       adapters:  h[t+1] = h[t] + g[t] * F_shared(h[t]; adapter[t], norm[t]).
       --steps 0 on D gives the "naive repeat" control (adapters are identity at init).
"""
import argparse
import json
import math
import os
import platform
import random
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

from looprec.adapters import LoRALinear, lora_param_count
from looprec.data import TextData
from looprec.evaluation import build_eval_sets, eval_suite, summarize
from looprec.mcbench import eval_mc, load_mc
from looprec.laco import (build_student, count_trainable, load_laco_manifest, sha256_file, trainable_params,
                          trainable_state_dict)
from looprec.losses import distill_ce, hidden_loss
from looprec.qwen3_core import Block, export_hf_state_dict, load_qwen3
from looprec.tasks import TaskTokenizer, train_example

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--cond", required=True, choices=["A", "B", "C", "D"])
ap.add_argument("--teacher", default="checkpoints/teacher_tuned", help="dir or HF id of the (tuned) teacher")
ap.add_argument("--base", default="Qwen/Qwen3-0.6B-Base", help="cond A only: model to tune")
ap.add_argument("--laco", help="LaCo manifest json")
ap.add_argument("--out-root", default="runs")
# adapters
ap.add_argument("--rank", type=int, default=16)
ap.add_argument("--alpha", type=float, default=None, help="default 2*rank")
ap.add_argument("--match-params", type=int, default=None, help="cond C/A: pick LoRA rank to match this many trainables")
ap.add_argument("--loops", type=int, default=8, help="D: number of loop-specific adapters (max trained loop count)")
ap.add_argument("--adapter", default="loop", choices=["loop", "shared", "none"])
ap.add_argument("--norm", default="loop", choices=["frozen", "shared", "loop"])
ap.add_argument("--no-gate", action="store_true")
ap.add_argument("--rec-init", default="laco", choices=["laco", "anchor", "mean"])
ap.add_argument("--extrap", default="clamp", choices=["clamp", "cycle"])
ap.add_argument("--t-sched", default="pow2", choices=["fixed", "pow2", "uniform"])
# objective
ap.add_argument("--w-kl", type=float, default=1.0)
ap.add_argument("--w-ce", type=float, default=0.5)
ap.add_argument("--w-kl-text", type=float, default=None)
ap.add_argument("--w-ce-text", type=float, default=None)
ap.add_argument("--w-kl-task", type=float, default=None)
ap.add_argument("--w-ce-task", type=float, default=None)
ap.add_argument("--hid", default="none", choices=["none", "final", "steps", "all"])
ap.add_argument("--w-hid", type=float, default=1.0)
ap.add_argument("--p-scratch", type=float, default=0.5, help="fraction of task examples with intermediate-state targets")
ap.add_argument("--task-weight", type=float, default=1.0)
# optimisation / memory
ap.add_argument("--steps", type=int, default=600)
ap.add_argument("--bs-text", type=int, default=4)
ap.add_argument("--bs-task", type=int, default=8)
ap.add_argument("--ctx", type=int, default=256)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--warmup", type=int, default=30)
ap.add_argument("--quant", default="none", choices=["none", "4bit", "8bit"])
ap.add_argument("--grad-ckpt", action="store_true")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--device", default="cuda", help="cuda (default) or cpu (tiny-model pipeline tests)")
# eval
ap.add_argument("--eval-loops", default=None, help="comma list, e.g. 1,2,4,8,16 (cond D)")
ap.add_argument("--n-text", type=int, default=64)
ap.add_argument("--n-long", type=int, default=16)
ap.add_argument("--n-task", type=int, default=64)
ap.add_argument("--n-mc", type=int, default=300, help="items per MC benchmark (ARC-E, ARC-C, PIQA); 0 disables")
ap.add_argument("--no-latency", action="store_true")
ap.add_argument("--eval-base-too", action="store_true", help="cond A: also evaluate the untuned base model")
a = ap.parse_args()

dev = a.device
CUDA = dev.startswith("cuda")
out = os.path.join(a.out_root, a.name)
os.makedirs(out, exist_ok=True)
random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
nrng = np.random.default_rng(a.seed)
prng = random.Random(a.seed)
log_f = open(os.path.join(out, "train_log.jsonl"), "a" if os.path.exists(os.path.join(out, "resume.pt")) else "w")


def log(*x):
    print(*x, flush=True)


# ------------------------------------------------------------------ build ----
t_load = time.time()
cpu_load = a.quant != "none"            # quantisation must happen at .cuda() time
if a.cond == "A":
    teacher, tdir = load_qwen3(a.base, device=dev)          # frozen reference (base model)
else:
    teacher, tdir = load_qwen3(a.teacher, device="cpu" if cpu_load else dev)
tok_file = os.path.join(tdir, "tokenizer.json")
text = TextData(tok_file)
tt = TaskTokenizer(tok_file)

merges, laco_man, deleted, modes = [], None, [], None
if a.laco:
    laco_man = load_laco_manifest(a.laco)
    merges, deleted, modes = laco_man["merges"], laco_man.get("deleted", []), laco_man.get("modes")
n_removed = sum(len(m["absorbed"]) for m in merges) + len(deleted)
alpha = a.alpha if a.alpha else 2.0 * a.rank

if a.cond == "A":
    per_rank = lora_param_count(teacher.entries[0], 1) * teacher.cfg.num_layers
    rank = max(1, round(a.match_params / per_rank)) if a.match_params else a.rank
    student = build_student(teacher, [], lora=dict(rank=rank, alpha=2.0 * rank))
elif a.cond == "B":
    student = build_student(teacher, merges, quant=a.quant, modes=modes, deleted=deleted)
elif a.cond == "C":
    n_blocks = teacher.cfg.num_layers - n_removed
    per_rank = lora_param_count(teacher.entries[0], 1) * n_blocks
    rank = max(1, round(a.match_params / per_rank)) if a.match_params else a.rank
    student = build_student(teacher, merges, lora=dict(rank=rank, alpha=a.alpha or 2.0 * rank), quant=a.quant,
                            modes=modes, deleted=deleted)
else:
    rec = dict(max_loops=a.loops, adapter=a.adapter, rank=a.rank, alpha=alpha, norm=a.norm,
               gate=not a.no_gate, init=a.rec_init, extrap=a.extrap)
    student = build_student(teacher, merges, rec=rec, quant=a.quant, modes=modes, deleted=deleted)
student = student.to(dev)
if a.quant != "none":
    teacher = teacher.to(dev)
student.grad_ckpt = a.grad_ckpt
n_train = count_trainable(student)
site = student.site()
log(f"[{a.name}] cond={a.cond} layer_evals(T=default)={student.n_layer_evals()} entries={len(student.entries)} "
    f"trainable={n_train:,} load={time.time()-t_load:.1f}s gpu={(torch.cuda.memory_allocated()/2**20 if CUDA else 0):.0f}MB")

# ---------------------------------------------------------------- weights ----
w = dict(kl_text=a.w_kl, ce_text=a.w_ce, kl_task=a.w_kl, ce_task=a.w_ce)
if a.cond == "A":
    w = dict(kl_text=1.0, ce_text=0.0, kl_task=0.0, ce_task=1.0)          # tune on tasks, stay near base on text
for k in w:
    v = getattr(a, "w_" + k)
    if v is not None:
        w[k] = v
need_teacher = (w["kl_text"] > 0 or w["kl_task"] > 0 or a.hid != "none")
W = student.head_weight()

# ---------------------------------------------------------------- training ----
params = [p for _, p in trainable_params(student)]
if a.steps > 0 and params:
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.99), weight_decay=0.0)
    sched = lambda s: min(1.0, (s + 1) / a.warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / a.steps))))
    Tmax = a.loops if a.cond == "D" else None
    pow2 = [t for t in (1, 2, 4, 8, 16, 32) if t <= (Tmax or 1)]

    def sample_T():
        if Tmax is None:
            return None
        if a.t_sched == "fixed":
            return Tmax
        if a.t_sched == "uniform":
            return prng.randint(1, Tmax)
        return prng.choice(pow2)

    if CUDA:
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    run_stats = {}
    ck_path = os.path.join(out, "resume.pt")
    start = 0
    if os.path.exists(ck_path):                      # resume after a crash (shared GPU can drop the process)
        ck = torch.load(ck_path, map_location=dev)
        for (n, p_), (n2, v) in zip(trainable_params(student), ck["params"].items()):
            assert n == n2
            p_.data.copy_(v)
        opt.load_state_dict(ck["opt"])
        start = ck["step"]
        log_f.close(); log_f = open(os.path.join(out, "train_log.jsonl"), "a")
        log(f"  resumed from step {start}")
    for step in range(start, a.steps):
        prng.seed(f"{a.seed}-{step}"); nrng = np.random.default_rng([a.seed, step])
        for g in opt.param_groups:
            g["lr"] = a.lr * sched(step)
        T = sample_T()
        text_ids = text.train_batch(nrng, a.bs_text, a.ctx).to(dev)
        exs = [train_example(prng, a.p_scratch) for _ in range(a.bs_task)]
        task_ids, tmask, _ = tt.collate(exs, dev)
        batches = [("text", text_ids, torch.ones_like(text_ids, dtype=torch.bool), 1.0),
                   ("task", task_ids, tmask, a.task_weight)]
        tot = 0.0
        for src, ids, pm, wt in batches:
            ht, t_states = None, None
            if need_teacher:
                with torch.no_grad():
                    ht, tst = teacher.forward_hidden(ids, collect=a.hid != "none")
                    t_states = {i: h for i, h, _ in tst} if tst else None
            hs, s_states = student.forward_hidden(ids, n_loops=T, collect=a.hid != "none")
            loss, st = distill_ce(hs, ht, ids, pm, W, w["kl_" + src], w["ce_" + src])
            if a.hid != "none":
                hl, npairs = hidden_loss(s_states, t_states, a.hid, T)
                if hl is not None:
                    loss = loss + a.w_hid * hl
                    st["hid"] = hl.detach()
            (loss * wt).backward()
            tot = tot + loss.detach() * wt
            for k, v in st.items():
                if k != "n":
                    run_stats[f"{src}_{k}"] = run_stats.get(f"{src}_{k}", 0.0) + v
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if (step + 1) % 100 == 0 and step + 1 < a.steps:
            torch.save({"params": {n: p_.detach() for n, p_ in trainable_params(student)}, "opt": opt.state_dict(),
                        "step": step + 1}, ck_path + ".tmp")
            os.replace(ck_path + ".tmp", ck_path)
        if (step + 1) % 25 == 0 or step == a.steps - 1:
            rec_ = {"step": step + 1, "loss": float(tot), "T": T, "gnorm": float(gn), "lr": a.lr * sched(step),
                    "sec": time.time() - t0, "peak_mb": torch.cuda.max_memory_allocated() / 2**20 if CUDA else 0.0}
            n_acc = 25 if (step + 1) % 25 == 0 else (step + 1) % 25
            rec_.update({k: float(v) / n_acc for k, v in run_stats.items()})
            run_stats = {}
            log_f.write(json.dumps(rec_) + "\n"); log_f.flush()
            log(f"  step {step+1}/{a.steps} loss={rec_["loss"]:.3f} T={T} gn={gn:.2f} {rec_['sec']:.0f}s peak={rec_['peak_mb']:.0f}MB "
                + " ".join(f"{k}={rec_[k]:.3f}" for k in ("text_kl", "text_ce", "task_kl", "task_ce", "task_hid") if k in rec_))
    train_sec = time.time() - t0
    train_peak_mb = torch.cuda.max_memory_allocated() / 2**20 if CUDA else 0.0
else:
    train_sec, train_peak_mb = 0.0, 0.0

# ------------------------------------------------------------------- save ----
files = []
if a.cond == "A" and a.steps > 0:
    tdir_out = os.path.join("checkpoints", a.name)
    os.makedirs(tdir_out, exist_ok=True)
    from safetensors.torch import save_file
    save_file(export_hf_state_dict(student), os.path.join(tdir_out, "model.safetensors"))
    shutil.copy(os.path.join(tdir, "config.json"), tdir_out)
    shutil.copy(tok_file, tdir_out)
    files.append(os.path.join(tdir_out, "model.safetensors"))
elif a.steps > 0 and params:
    p = os.path.join(out, "trainable.pt")
    torch.save(trainable_state_dict(student), p)
    files.append(p)

# --------------------------------------------------------------- evaluate ----
ref_teacher = teacher
base_model = None
if a.cond == "A":
    # evaluate the *saved* (merged, bf16-rounded) tuned teacher - that is what the students will be pruned from
    base_model = teacher
    del student, params
    if a.steps > 0:
        if CUDA:
            torch.cuda.empty_cache()
        student, _ = load_qwen3(os.path.dirname(files[0]), device=dev)
    else:
        student = teacher
    ref_teacher = student        # tuned teacher is its own reference (KL = 0); base-vs-tuned reported below
    n_train_report = n_train
    W = student.head_weight()
sets = build_eval_sets(text, tt, dev, n_text=a.n_text, n_long=a.n_long, n_task=a.n_task)
if a.cond == "D":
    loops = [int(x) for x in (a.eval_loops or "1,2,4,8,16").split(",")]
else:
    loops = [None]
t_ev = time.time()
res = eval_suite(student, ref_teacher, sets, loops, latency=not a.no_latency)
mc = load_mc(a.n_mc) if a.n_mc else None
mc_res = eval_mc(student, tt, mc, loops, dev) if mc else {}
results = {"per_loop": {}}
for T in loops:
    r = res[str(T)]
    if mc_res:
        r["mc"] = mc_res[str(T)]
    r["summary"] = summarize(r)
    r["supported"] = (T is None) or (T <= a.loops)
    results["per_loop"][str(T)] = r
if a.cond == "A" and a.eval_base_too:
    base = base_model
    rb = eval_suite(base, student, sets, [None], latency=False)["None"]
    if mc:
        rb["mc"] = eval_mc(base, tt, mc, [None], dev)["None"]
    rb["summary"] = summarize(rb)
    results["base_vs_tuned"] = rb
log(f"eval done in {time.time()-t_ev:.0f}s")
for T in loops:
    s = results["per_loop"][str(T)]["summary"]
    log(f"  T={T}: text_kl={s['text_kl']:.4f} text1024_kl={s['text1024_kl']:.4f} algo_id_direct={s['algo_id_direct']:.3f} "
        f"algo_ood_direct={s['algo_ood_direct']:.3f} algo_id_scratch={s['algo_id_scratch']:.3f} "
        f"mc_norm={s.get('mc_acc_norm', float('nan')):.3f}")

results["meta"] = dict(trainable_params=n_train if a.steps > 0 else 0, trainable_params_available=n_train, train_sec=train_sec, train_peak_mb=train_peak_mb,
                       weights_mb=sum(p.numel() * p.element_size() for p in student.parameters()) / 2**20,
                       layer_evals_default=student.n_layer_evals(),
                       n_entries=len(student.entries), site=None if site is None else dict(anchor=site.anchor, span=site.span))
man = dict(name=a.name, cond=a.cond, args=vars(a), weights=w, results_file="results.json",
           trainable_params=n_train, laco_manifest=a.laco,
           laco_manifest_sha256=sha256_file(a.laco) if a.laco else None,
           teacher_dir=tdir, teacher_sha256=sha256_file(os.path.join(tdir, "model.safetensors")),
           checkpoint_files=[dict(path=f, bytes=os.path.getsize(f), sha256=sha256_file(f)) for f in files],
           env=dict(torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0) if CUDA else "cpu",
                    python=platform.python_version()),
           timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"))
json.dump(results, open(os.path.join(out, "results.json"), "w"), indent=1)
json.dump(man, open(os.path.join(out, "manifest.json"), "w"), indent=1)
log("saved", out)
