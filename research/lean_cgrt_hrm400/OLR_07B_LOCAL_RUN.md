# Run OLR-07B on a CPU/RAM machine

This is the registered local library-receipt gym, not a request to redo the finite rewrite series. See [protocol](OLR_07B_REGISTRATION.md) and [07A audit](OLR_07A_RESULTS.md).

## Setup (Windows PowerShell)

Use a fresh checkout of the OLR research branch containing this file, or a worktree of your existing clone. Python 3.11 and the Lean v4.19.0 toolchain are recommended.

```powershell
git clone -b research/olr-obligation-ledger-20261123 https://github.com/MoralityLabAI/LoopedTransformers.git
cd LoopedTransformers/research/lean_cgrt_hrm400
py -3.11 -m venv .venv07b
$py = ".\.venv07b\Scripts\python.exe"
& $py -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
& $py -m pip install numpy psutil
elan toolchain install leanprover/lean4:v4.19.0
Push-Location lean
lake build
Pop-Location
git clone --branch v4.19.0 --depth 1 https://github.com/leanprover/lean4.git D:\lean4-419-source
git -C D:\lean4-419-source rev-parse HEAD
```

Expected source HEAD: 6caaee842e9495688c1567e78c0e68dbb96942aa. Do not build Lean from the source clone. The existing lgym backend is built against the installed pinned toolchain. Launcher detects the active elan toolchain from lean/; alternatively set LCGRT_LEAN_HOME explicitly to the pinned toolchain directory.

Put the unmodified 07A Actions artifact ZIP from [14890402416](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45327039231/artifacts/14890402416) at D:\olr07a-native-library.zip. Do not unzip it. It supplies exact old-goal exclusions, not source training labels.

## Run

```powershell
& $py run_olr07b.py --lean-source D:\lean4-419-source --prior-07a D:\olr07a-native-library.zip --out D:\olr07b-smoke --profile smoke --threads 4 --workers 1 --ram 8G
& $py run_olr07b.py --lean-source D:\lean4-419-source --prior-07a D:\olr07a-native-library.zip --out D:\olr07b-full --profile full --threads 8 --workers 2 --ram 12G
```

Run smoke first. It exercises the real collector and classical policies on six files, but is not eligible for model fitting or a scientific claim. Full freezes 55 files across distinct roles, samples up to 48 positions per file (not a guaranteed final scored count), and uses four fresh select passes per file. The local workload is CPU/RAM, with no GPU use. If source validity/calibration headroom fails, fitting is deliberately skipped and full native receipts and controls are returned. If it passes, fit 15 checkpoints at 400 updates each. No 5M scale-up.

On a 32 GiB machine with sufficient idle memory, 12G is a conservative committed-memory limit for two native workers. You can use --threads 16 --workers 4 --ram 20G on a larger/idle machine. Resource settings are not an experimental intervention. Re-running the same command resumes verified per-file and per-checkpoint stages; output must remain on the same scientific contract (including torch version, training threads and binary hashes). To change the scientific settings, use a new directory. Unfinished stages are rebuilt. If nested Windows Job assignment fails, run in a normal PowerShell window outside the restrictive parent job; do not silently remove the limit.

Linux: install the same Python packages and use `systemd-run --user --scope -p MemoryMax=12G python run_olr07b.py ... --ram 12G`. The launcher checks an actual cgroup cap. Explicit --ram 0 opts out. Sampled summed process-tree RSS may double-count shared pages; Windows additionally reports job peak commit. Actual wall times and resource use are recorded.

## Return

Return **olr07b-full_AUDIT.zip** to ChatGPT. It includes the raw native traces, separate human audits, both candidate replays, frozen registry and policies, actual weights (if eligible), source/evaluation metrics, resource report, code and manifest. A failed/incomplete run preserves file-level checkpoints; attach resource_latest.json and the error log if it cannot complete.

The generated **olr07b-full_FABLE.zip** contains the protocol, code, concise summary, frozen policy/fit metadata and full-audit manifest. Wait for the independent ChatGPT audit before sending that smaller package to Fable. Its scientific status stays explicit even when a gate fails. No raw unrelated local files, tokens or credentials are included.
