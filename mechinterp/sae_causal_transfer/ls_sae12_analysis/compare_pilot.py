"""Compare two LS-SAE12 engineering-pilot receipts (e.g. CPU vs CUDA): screening digits,
directions, same-forward clean agreement, V and D_SI per state. Usage: compare_pilot.py A.json B.json"""
import json, sys
import numpy as np

a = json.load(open(sys.argv[1]))  # first receipt (CPU)
b = json.load(open(sys.argv[2]))  # second receipt (CUDA)
print("platform cpu :", {k: a["platform"].get(k) for k in ("torch", "numpy", "torch_threads")})
print("platform cuda:", {k: b["platform"].get(k) for k in ("torch", "numpy", "gpu", "cuda", "float32_matmul_precision")})
for ca, cb in zip(a["cases"], b["cases"]):
    sa, sb = ca["screen"], cb["screen"]
    pa, pb = sa["clean_predicted_digits_by_step"], sb["clean_predicted_digits_by_step"]
    same_steps = sum(x == y for x, y in zip(pa, pb))
    print(f'{sa["source_file"]}: screen digits equal at {same_steps}/16 steps, eligible {sa["eligible"]}/{sb["eligible"]}, '
          f'site {sa["selected_site"]}/{sb["selected_site"]}, ACT16 correct {sa["clean_blank_correct_by_step"][15]}/{sb["clean_blank_correct_by_step"][15]}')
    for key in ("selected", "control"):
        if key not in ca or key not in cb:
            continue
        x, y = ca[key], cb[key]
        da, db = x["directions"], y["directions"]
        cs = float(np.dot(da["structural"]["vector"], db["structural"]["vector"]))
        cl = float(np.dot(da["logit"]["vector"], db["logit"]["vector"]))
        rows_eq = sum(r1 == r2 for r1, r2 in zip(x["predicted_digits_by_step"], y["predicted_digits_by_step"]))
        vx, vy = x["summary"]["0.02"]["V"], y["summary"]["0.02"]["V"]
        print(f'  {key:8s} site {x["site"]}: cos(struct cpu,cuda) {cs:.6f} cos(logit) {cl:.6f} '
              f'sigma_s {da["structural"]["sigma_estimate"]:.4f}/{db["structural"]["sigma_estimate"]:.4f} '
              f'rows with identical digits {rows_eq}/{len(x["predicted_digits_by_step"])} '
              f'batch clean==screen cpu {sum(x["same_forward_clean_matches_screen_steps"])} cuda {sum(y["same_forward_clean_matches_screen_steps"])}')
        print(f'           V@0.02 cpu {({k: round(v, 3) for k, v in vx.items()})}  cuda {({k: round(v, 3) for k, v in vy.items()})}')
        print(f'           D_SI@0.02 cpu {x["summary"]["0.02"]["D_structural_minus_isotropic"]:.3f} cuda {y["summary"]["0.02"]["D_structural_minus_isotropic"]:.3f}')
