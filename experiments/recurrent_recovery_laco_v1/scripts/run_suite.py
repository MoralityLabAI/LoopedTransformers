"""Experiment orchestrator.  Runs every condition / ablation sequentially as subprocesses, skipping runs whose
results.json already exists, waiting for free VRAM (the GPU may be shared) and retrying on CUDA OOM.

  python scripts/run_suite.py --phase teacher
  python scripts/run_suite.py --phase laco  [--laco-recipe path/to/laco_recipe.json]
  python scripts/run_suite.py --phase main
  python scripts/run_suite.py --phase ablate
  python scripts/run_suite.py --phase all
  python scripts/run_suite.py --list

Scale knobs (documented reductions are recorded in reports/REDUCTIONS.md):
  --steps / --steps-ablate, --min-free-mb, --quick
"""
import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.environ.get("LOOPREC_PYTHON", sys.executable)

ap = argparse.ArgumentParser()
ap.add_argument("--phase", default="all", choices=["teacher", "laco", "main", "ablate", "all"])
ap.add_argument("--base", default="Qwen/Qwen3-0.6B-Base")
ap.add_argument("--n-remove", type=int, default=7, help="layers removed by LaCo (ignored if --laco-recipe)")
ap.add_argument("--laco-recipe", default=None, help="recipe/manifest from the LaCo reproduction project")
ap.add_argument("--weight-mode", default="last")
ap.add_argument("--steps", type=int, default=600)
ap.add_argument("--steps-ablate", type=int, default=300)
ap.add_argument("--teacher-steps", type=int, default=1200)
ap.add_argument("--loops", type=int, default=8)
ap.add_argument("--rank", type=int, default=16)
ap.add_argument("--min-free-mb", type=int, default=4500)
ap.add_argument("--quick", action="store_true", help="tiny smoke configuration")
ap.add_argument("--list", action="store_true")
ap.add_argument("--only", default=None, help="comma list of run names")
ap.add_argument("--max-retries", type=int, default=40)
a = ap.parse_args()
os.chdir(ROOT)

TEACHER = "checkpoints/teacher_tuned"
LACO = a.laco_recipe or f"manifests/laco_r{a.n_remove}.json"
common_eval_main = ["--n-task", "64", "--n-mc", "400"]
common_eval_abl = ["--n-task", "32", "--n-mc", "150", "--no-latency", "--eval-loops", "1,2,4,8"]
if a.quick:
    a.steps = a.steps_ablate = a.teacher_steps = 20
    common_eval_main = common_eval_abl = ["--n-task", "8", "--n-mc", "20", "--n-text", "8", "--n-long", "2", "--no-latency", "--eval-loops", "1,2"]

# D(main) trainable budget, used to rank-match C
D_PARAMS = None


def d_params(loops, rank, norm="loop", gate=True, adapter="loop"):
    per_rank = 22528  # q,k,v,o,gate,up,down for hidden=1024 / inter=3072 / 16x128 heads / 8 kv heads
    n_ad = {"loop": loops, "shared": 1, "none": 0}[adapter]
    n_nm = {"loop": loops, "shared": 1, "frozen": 0}[norm]
    return n_ad * rank * per_rank + n_nm * (2 * 1024 + 2 * 128) + (loops if gate else 0)


def run(name, argv, need_gpu=True):
    if a.only and name not in a.only.split(","):
        return
    out = os.path.join("runs", name, "results.json")
    if os.path.exists(out):
        print(f"[skip] {name}")
        return
    cmd = [PY, "-u", "scripts/train_student.py", "--name", name] + argv
    if a.list:
        print(name, " ".join(argv))
        return
    for attempt in range(a.max_retries):
        while need_gpu and free_mb() < a.min_free_mb:
            print(f"  waiting for VRAM ({free_mb()} MB free < {a.min_free_mb}) ...", flush=True)
            time.sleep(60)
        print(f"[run] {name} (attempt {attempt+1})  {' '.join(argv)}", flush=True)
        os.makedirs("reports/logs", exist_ok=True)
        with open(f"reports/logs/{name}.log", "a") as lf:
            rc = subprocess.call(cmd, stdout=lf, stderr=subprocess.STDOUT)
        if rc == 0:
            return
        tail = open(f"reports/logs/{name}.log", errors="ignore").read()[-3000:]
        if "CUDA" in tail or "out of memory" in tail or "CUBLAS" in tail or rc in (3221225477, -1073741819) or not tail.strip():
            print(f"  CUDA failure for {name}; backing off 90 s", flush=True)
            time.sleep(90)
            continue
        print(f"  {name} FAILED (rc={rc}); see reports/logs/{name}.log", flush=True)
        return


def free_mb():
    try:
        o = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"], text=True)
        u, t = [int(x) for x in o.strip().split(",")]
        return t - u
    except Exception:
        return 10**9


def laco(name, n_remove, mode="single"):
    path = f"manifests/{name}.json"
    if os.path.exists(path):
        return path
    if a.list:
        print("laco", name, n_remove)
        return path
    while free_mb() < 3000:
        time.sleep(60)
    subprocess.check_call([PY, "scripts/run_laco.py", "--teacher", TEACHER, "--n-remove", str(n_remove), "--mode", mode,
                           "--weight-mode", a.weight_mode, "--out", path])
    return path


# ------------------------------------------------------------- phases ----
if a.phase in ("teacher", "all"):
    run("teacher_tuned", ["--cond", "A", "--base", a.base, "--rank", "64", "--steps", str(a.teacher_steps), "--lr", "3e-4",
                          "--p-scratch", "0.5", "--grad-ckpt", "--eval-base-too", "--n-task", "64", "--n-mc", "400"])

if a.phase in ("laco", "all", "main", "ablate"):
    if a.laco_recipe:
        L7 = a.laco_recipe
        from_recipe = True
    else:
        L7 = laco(f"laco_r{a.n_remove}", a.n_remove)
        from_recipe = False
    # FLOPs-matched baselines: keep T-1 extra real layers  (m' = m - T + 1)
    L_m = {}
    for T in (2, 4):
        mm = a.n_remove - T + 1
        if mm > 0 and not from_recipe:
            L_m[T] = laco(f"laco_r{mm}", mm)

if a.phase in ("main", "all"):
    P = d_params(a.loops, a.rank)
    base_args = ["--teacher", TEACHER, "--steps", str(a.steps), "--grad-ckpt"]
    run("B_pruned", ["--cond", "B", "--laco", L7, "--teacher", TEACHER, "--steps", "0", "--grad-ckpt"] + common_eval_main)
    run("E_naive_repeat", ["--cond", "D", "--laco", L7, "--teacher", TEACHER, "--steps", "0", "--loops", str(a.loops),
                           "--eval-loops", "1,2,4,8,16"] + common_eval_main)
    run("C_lora_matched", ["--cond", "C", "--laco", L7, "--match-params", str(P)] + base_args + common_eval_main)
    run("C_lora_r32", ["--cond", "C", "--laco", L7, "--rank", "32"] + base_args + common_eval_main)
    run("D_main", ["--cond", "D", "--laco", L7, "--rank", str(a.rank), "--loops", str(a.loops),
                   "--eval-loops", "1,2,4,8,16"] + base_args + common_eval_main)
    # FLOPs-matched: LaCo-prune fewer layers (T-1 extra real layers instead of T-1 extra loops) + LoRA recovery
    for T, path in L_m.items():
        run(f"B_pruned_f{T}", ["--cond", "B", "--laco", path, "--teacher", TEACHER, "--steps", "0"] + common_eval_main)
        run(f"C_lora_f{T}", ["--cond", "C", "--laco", path, "--match-params", str(P)] + base_args + common_eval_main)

if a.phase in ("ablate", "all"):
    ab = ["--teacher", TEACHER, "--steps", str(a.steps_ablate), "--grad-ckpt"] + common_eval_abl
    D = lambda extra: ["--cond", "D", "--laco", L7, "--rank", str(a.rank), "--loops", str(a.loops)] + ab + extra
    # reference run at ablation scale
    run("abl_ref", D([]))
    # adapter rank
    for r in (2, 4, 8, 32):
        run(f"abl_rank{r}", D(["--rank", str(r)]))
    # loop count (adapters / max trained loops)
    for L in (1, 2, 4):
        run(f"abl_loops{L}", D(["--loops", str(L)]))
    run("abl_tfixed", D(["--t-sched", "fixed"]))
    # normalisation sharing
    run("abl_norm_frozen", D(["--norm", "frozen"]))
    run("abl_norm_shared", D(["--norm", "shared"]))
    run("abl_adapter_shared", D(["--adapter", "shared"]))
    run("abl_adapter_none", D(["--adapter", "none"]))
    run("abl_nogate", D(["--no-gate"]))
    run("abl_init_anchor", D(["--rec-init", "anchor"]))
    # training objectives
    run("abl_obj_ce", D(["--w-kl", "0", "--w-ce", "1"]))
    run("abl_obj_kl", D(["--w-kl", "1", "--w-ce", "0"]))
    run("abl_obj_kl_hidfinal", D(["--hid", "final", "--w-hid", "1.0"]))
    run("abl_obj_kl_hidsteps", D(["--hid", "steps", "--w-hid", "1.0"]))
    run("abl_scratch0", D(["--p-scratch", "0"]))
    run("abl_scratch1", D(["--p-scratch", "1.0"]))
    # matched-params C at ablation scale
    run("abl_C_ref", ["--cond", "C", "--laco", L7, "--match-params", str(d_params(a.loops, a.rank))] + ab)
print("suite done")
