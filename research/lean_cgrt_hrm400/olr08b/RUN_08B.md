# OLR-08B CPU Launch Package

CPU-first execution package for **OLR-08B: Adaptive Proof Frontier Allocation and Memory-Augmented Lean Reasoning** (8 threads, 2 workers, enforced 12 GiB RAM cap).

## Prerequisites

- Pinned Lean 4.19.0 source checkout at `6caaee842e9495688c1567e78c0e68dbb96942aa` (clean tracked tree).
- Built standalone `lgym.exe` from `research/lean_cgrt_hrm400/olr08/lean/` (`SHA-256 61923c657261feb458c60c7049f83cc0f2c121ce8dd3dcad09b039f8d3c5dede`).
- Lean 4.19.0 toolchain `bin` on `PATH`.
- Authenticated 07A and 07B exact-registry reissue manifests (`results/olr07_reissue/olr07-scope-reissue-07a.zip` and `07b.zip` unpacked).
- CPU Python 3.12+ with `torch==2.5.1+cpu`, `numpy`, `psutil`.

## 1. Regression Tests

```powershell
python -m unittest -v research/lean_cgrt_hrm400/olr08b/test_olr08b.py
python -m unittest -v research/lean_cgrt_hrm400/olr08/test_gym.py
```

## 2. Objective 1 Spent-Cohort Diagnostics (Exploratory, Spent Cohorts Only)

```powershell
$env:PATH = "$HOME\.elan\toolchains\leanprover--lean4---v4.19.0\bin;" + $env:PATH
python research/lean_cgrt_hrm400/olr08b/diag_obj1.py `
  --source D:\lean4-419-source `
  --exe C:\olr08\pkg\lean\.lake\build\bin\lgym.exe `
  --audit-zip research/lean_cgrt_hrm400/results/olr08a/olr08-full_AUDIT.zip `
  --referee-zip <path-to-authenticated-olr08_referee.zip> `
  --out C:\olr08b\diag_obj1 --n-train-sample 30 --workers 2
```

## 3. Pre-Outcome Contract & Registry Freeze (Zero Outcomes Collected)

```powershell
$env:PATH = "$HOME\.elan\toolchains\leanprover--lean4---v4.19.0\bin;" + $env:PATH
python research/lean_cgrt_hrm400/olr08b/launch08b.py `
  --source D:\lean4-419-source `
  --exe C:\olr08\pkg\lean\.lake\build\bin\lgym.exe `
  --out C:\olr08b\full `
  --profile full --threads 8 --workers 2 --ram-gib 12 `
  --reissue-a C:\olr08\reissues\scope-reissue-07a\manifest.json `
  --reissue-b C:\olr08\reissues\scope-reissue-07b\manifest.json `
  --stop-after-freeze
```

## 4. Full CPU-First Calibration Campaign (8 Threads, 2 Workers, 12 GiB Cap)

```powershell
$env:PATH = "$HOME\.elan\toolchains\leanprover--lean4---v4.19.0\bin;" + $env:PATH
python research/lean_cgrt_hrm400/olr08b/launch08b.py `
  --source D:\lean4-419-source `
  --exe C:\olr08\pkg\lean\.lake\build\bin\lgym.exe `
  --out C:\olr08b\full `
  --profile full --threads 8 --workers 2 --ram-gib 12 `
  --reissue-a C:\olr08\reissues\scope-reissue-07a\manifest.json `
  --reissue-b C:\olr08\reissues\scope-reissue-07b\manifest.json
```

On Linux, wrap with:
`systemd-run --user --scope -p MemoryMax=12G python research/lean_cgrt_hrm400/olr08b/launch08b.py ... --ram-gib 12`

## Invariants

- Original 275-root evaluation registry remains strictly protected (smoke-observed root `BitVec.shiftLeftZeroExtend`, id `45ce790b…`, explicitly excluded from pristine confirmation -> 274 pristine roots).
- `SInt` remains strictly sealed (`0` files touched).
- No 500K or 5M model promotion until all preregistered gates pass.
