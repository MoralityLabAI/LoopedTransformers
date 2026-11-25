import json
import shutil
import subprocess
import sys
from pathlib import Path

PUSH = Path(r"C:\Users\patri\lt-main-push-20261126")
AUX = Path(r"C:\Users\patri\Documents\Codex\rmp_v0_aux")
LBF = Path(r"C:\Users\patri\lbf20m-run-20261125")
BLU = r"C:\Users\patri\.venvs\bluebeam\Scripts\python.exe"
WT = r"C:\Users\patri\rmp-aux-20261125"

lbf_dst = PUSH / "research" / "lite_braille_fresco" / "local_reproduction_20261125"
aux_dst = PUSH / "experiments" / "recurrence_monitorability_v0" / "aux_gym_20261126"
for d in (lbf_dst, aux_dst):
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    (d / ".gitattributes").write_text("* -text\n")  # keep bytes exact: receipt hashes are over the files as written

# ---- LBF-20M local reproduction ----
skip = {"src", "report"}
for p in LBF.iterdir():
    if p.name in skip or p.suffix == ".zip" or p.name in ("launcher.log", "launcher.err", "theory.stdout", "theory.stderr", "launcher.done"):
        continue
    if p.is_dir():
        shutil.copytree(p, lbf_dst / p.name, ignore=shutil.ignore_patterns("__pycache__"))
    else:
        shutil.copyfile(p, lbf_dst / p.name)
for n in ("launcher.log",):
    if (LBF / n).exists():
        shutil.copyfile(LBF / n, lbf_dst / n)

# ---- RMP aux gym ----
env = {"PYTHONPATH": WT, "PATH": __import__("os").environ["PATH"], "SYSTEMROOT": __import__("os").environ.get("SYSTEMROOT", "")}
status = subprocess.run([BLU, str(AUX / "collect_snapshot.py"), str(aux_dst)], capture_output=True, text=True, env=env)
(aux_dst / "cell_status.md").write_text(status.stdout, encoding="utf-8")
for name in ("cpu_profile.py", "modular_check.py", "modular_check2.py"):
    out = subprocess.run([BLU, "-u", str(AUX / name)] + (["fp32"] if name == "cpu_profile.py" else []), capture_output=True, text=True, env={**__import__("os").environ, "PYTHONPATH": WT}, cwd=WT)
    (aux_dst / (name.replace(".py", "") + ".out.txt")).write_text(out.stdout + ("\n--- stderr (tail) ---\n" + out.stderr[-1500:] if out.returncode else ""), encoding="utf-8")
out = subprocess.run([BLU, "-u", str(AUX / "cpu_profile.py"), "bf16"], capture_output=True, text=True, env={**__import__("os").environ, "PYTHONPATH": WT}, cwd=WT)
(aux_dst / "cpu_profile_bf16.out.txt").write_text(out.stdout, encoding="utf-8")
(aux_dst / "cpu_profile.out.txt").rename(aux_dst / "cpu_profile_fp32.out.txt")
for n in ("bf16_bench_gpu.json", "bf16_bench_cpu.json", "reeval_looptt_cpu.json", "bf16_autocast.patch"):
    shutil.copyfile(AUX / n, aux_dst / n)
sc = aux_dst / "scripts"; sc.mkdir()
for n in ("bf16_bench.py", "cpu_profile.py", "modular_check.py", "modular_check2.py", "collect_snapshot.py", "assemble_push.py",
          "run_aux_queue_seed.ps1", "run_cpu_cell.ps1", "run_cpu_modular.ps1", "lease_wait_s1.ps1"):
    shutil.copyfile(AUX / n, sc / n)
shutil.copyfile(LBF / "rmp_reeval.py", sc / "rmp_reeval.py")
shutil.copyfile(LBF / "cpu_throughput.py", sc / "cpu_throughput.py")
logs = aux_dst / "logs"; logs.mkdir()
for n in ("overnight.log", "overnight2_s1.log", "overnight2_s2.log", "overnight2_s3.log", "bf16_bench_gpu.log", "cpu_cell.log", "cpu_modular.log"):
    f = AUX / n
    if f.exists():
        raw = f.read_bytes().replace(b"\x00", b"").replace(b"\xff\xfe", b"").replace(b"\xef\xbb\xbf", b"")
        (logs / n).write_text(raw.decode("utf-8", "replace").replace("\r\n", "\n"), encoding="utf-8")
print(status.stdout)
