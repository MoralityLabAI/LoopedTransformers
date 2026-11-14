"""Hardware probes and guard thresholds for unattended runs (Windows and Linux).

`nvidia-smi` is read instead of torch.cuda queries so probing never creates a CUDA
context. On this laptop the GPU power cap can drop to 15 W; `enforced.power.limit`
is the reliable signal (the SM clock reading is not).
"""
from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import dataclass

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None


def gpu() -> dict:
    q = "name,temperature.gpu,memory.total,memory.used,memory.free,utilization.gpu,power.draw,enforced.power.limit"
    try:
        out = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=15).stdout.strip().splitlines()
    except (OSError, subprocess.TimeoutExpired):
        return {}
    if not out:
        return {}
    v = [s.strip() for s in out[0].split(",")]

    def num(s):
        try:
            return float(s)
        except ValueError:
            return None
    return {"name": v[0], "temp_c": num(v[1]), "mem_total_mib": num(v[2]), "mem_used_mib": num(v[3]),
            "mem_free_mib": num(v[4]), "util_pct": num(v[5]), "power_w": num(v[6]),
            "power_limit_w": num(v[7])}


def host(path=".") -> dict:
    d = {"disk_free_gib": round(shutil.disk_usage(path).free / 2 ** 30, 2)}
    if psutil:
        vm = psutil.virtual_memory()
        d.update(ram_total_gib=round(vm.total / 2 ** 30, 2), ram_avail_gib=round(vm.available / 2 ** 30, 2),
                 cpu_pct=psutil.cpu_percent(interval=None), cpu_count=psutil.cpu_count())
        try:
            b = psutil.sensors_battery()
            if b is not None:
                d.update(on_ac=bool(b.power_plugged), battery_pct=b.percent)
        except Exception:
            pass
        d["rss_gib"] = round(psutil.Process().memory_info().rss / 2 ** 30, 3)
    return d


@dataclass
class Limits:
    gpu_temp_c: float = 85.0
    min_power_limit_w: float = 60.0      # below this the 15 W cap is active
    min_disk_gib: float = 10.0
    min_ram_avail_gib: float = 0.4
    require_ac: bool = True
    use_gpu: bool = True                 # False for a CPU-only job: the GPU's state is not its concern


def check(limits: Limits, path=".") -> tuple[bool, str, dict]:
    """(ok, reason, snapshot). Missing sensors never block; they are logged as absent."""
    g, h = (gpu() if limits.use_gpu else {}), host(path)
    snap = {"gpu": g, "host": h}
    if g.get("temp_c") is not None and g["temp_c"] >= limits.gpu_temp_c:
        return False, f"gpu_temp {g['temp_c']}C >= {limits.gpu_temp_c}", snap
    if g.get("power_limit_w") is not None and g["power_limit_w"] < limits.min_power_limit_w:
        return False, f"gpu_power_cap {g['power_limit_w']}W", snap
    if h["disk_free_gib"] < limits.min_disk_gib:
        return False, f"disk_free {h['disk_free_gib']}GiB < {limits.min_disk_gib}", snap
    if h.get("ram_avail_gib") is not None and h["ram_avail_gib"] < limits.min_ram_avail_gib:
        return False, f"ram_avail {h['ram_avail_gib']}GiB", snap
    if limits.require_ac and h.get("on_ac") is False:
        return False, "on_battery", snap
    return True, "ok", snap


class ResourcePause(RuntimeError):
    """A bounded workload should exit 75 and resume after resources recover."""


class Guard:
    """Sample the existing hardware limits at most once per interval during inference."""
    def __init__(self, limits: Limits, path=".", interval=15):
        self.limits, self.path, self.interval = limits, path, interval
        self.last = float("-inf")

    def __call__(self):
        now = time.monotonic()
        if now - self.last < self.interval:
            return
        self.last = now
        ok, why, _ = check(self.limits, self.path)
        if not ok:
            raise ResourcePause(why)
