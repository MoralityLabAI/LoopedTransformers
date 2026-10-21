"""Run LaCo layer-collapse search on a teacher checkpoint and write a LaCo manifest (JSON).

  python scripts/run_laco.py --teacher checkpoints/teacher_tuned --n-remove 7 --out manifests/laco_r7.json

If you already have a manifest from a LaCo reproduction, skip this script and pass it to train_student.py via
--laco (any JSON with `merges: [{anchor, absorbed:[...]}]` or `removed_layers: [...]` is accepted).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

from looprec.data import TextData
from looprec.laco import (laco_search_paper, laco_search_single, plan_similarity, sha256_file,
                          write_laco_manifest)
from looprec.qwen3_core import load_qwen3

ap = argparse.ArgumentParser()
ap.add_argument("--teacher", default="Qwen/Qwen3-0.6B-Base")
ap.add_argument("--weight-mode", default="last", help="last|first|mean|rdsc|rdsc_mean (laco-lab semantics)")
ap.add_argument("--norm-mode", default="base")
ap.add_argument("--qknorm-mode", default="follow")
ap.add_argument("--n-remove", type=int, required=True)
ap.add_argument("--mode", default="single", choices=["single", "paper"])
ap.add_argument("--max-span", type=int, default=8, help="paper mode: C")
ap.add_argument("--gap", type=int, default=1, help="paper mode: interval I")
ap.add_argument("--calib-seqs", type=int, default=32)
ap.add_argument("--calib-ctx", type=int, default=256)
ap.add_argument("--out", required=True)
ap.add_argument("--device", default="cuda")
a = ap.parse_args()

modes = {"weight": a.weight_mode, "norm": a.norm_mode, "qknorm": a.qknorm_mode}
dev = a.device
teacher, d = load_qwen3(a.teacher, device=dev)
teacher.eval()
text = TextData(os.path.join(d, "tokenizer.json"))
rng = torch.Generator().manual_seed(0)
# calibration text = training-split pile chunks (never the held-out docs used for evaluation)
import numpy as np
calib = text.train_batch(np.random.default_rng(0), a.calib_seqs, a.calib_ctx).to(dev)

if a.n_remove == 0:
    merges, extra = [], {}
elif a.mode == "single":
    merges, table = laco_search_single(teacher, calib, a.n_remove, modes=modes)
    extra = {"search_table": table}
else:
    merges, info = laco_search_paper(teacher, calib, a.n_remove, a.max_span, a.gap, modes=modes)
    extra = {"search_info": info}
sim = plan_similarity(teacher, calib, merges, modes=modes) if merges else 1.0
tsha = sha256_file(os.path.join(d, "model.safetensors")) if os.path.exists(os.path.join(d, "model.safetensors")) else None
man = write_laco_manifest(a.out, model_dir=d, num_layers=teacher.cfg.num_layers, merges=merges,
                          modes=modes, extra={"mode": a.mode, "n_remove": a.n_remove, "final_cos_sim": sim, "teacher_sha256": tsha,
                                 "calib": {"seqs": a.calib_seqs, "ctx": a.calib_ctx, "source": "pile-10k train split"},
                                 **extra})
print(json.dumps({k: v for k, v in man.items() if k != "search_table"}, indent=1))
if "search_table" in extra:
    print("top candidates:", sorted(extra["search_table"], key=lambda r: -r["cos_sim"])[:5])
