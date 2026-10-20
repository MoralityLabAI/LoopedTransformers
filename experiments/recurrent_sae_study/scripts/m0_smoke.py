"""Smoke test for the mini-LaCo replica: train a small teacher, LaCo-collapse, build recurrent student, 1 recovery run."""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np
import torch

import recsae  # noqa: F401
from recsae import mini

ap = argparse.ArgumentParser()
ap.add_argument("--task", default="dfa")
ap.add_argument("--steps", type=int, default=400)
ap.add_argument("--rsteps", type=int, default=150)
ap.add_argument("--layers", type=int, default=12)
ap.add_argument("--threads", type=int, default=6)
a = ap.parse_args()
torch.set_num_threads(a.threads)
task = mini.PrefixTask(a.task)
t0 = time.time()
teacher = mini.train_teacher(task, layers=a.layers, d=128, seed=0, steps=a.steps)
print("teacher time", time.time() - t0)
rng = np.random.default_rng(5)
print("teacher last-acc by length", mini.eval_last_acc(teacher, task, rng, [4, 8, 16, 24, 32]))
merges, table = mini.laco_plan(teacher, task, n_remove=4)
print("merges", merges, "best sim", max(t["cos_sim"] for t in table))
lac = mini.build_laco_student(teacher, merges, "laco")
print("laco-only acc", mini.eval_last_acc(lac, task, rng, [4, 8, 16, 24, 32]), mini.model_summary(lac))
T = 4
rec = mini.build_laco_student(teacher, merges, "rec", T=T)
print("rec(untrained) summary", mini.model_summary(rec))
for t in (1, 2, 4, 6):
    print(f" untrained loops={t}", mini.eval_last_acc(rec, task, rng, [8, 16], n_loops=t))
mini.train_recovery(rec, teacher, task, T, steps=a.rsteps)
for t in (1, 2, 3, 4, 6, 8):
    print(f" recovered loops={t}", mini.eval_last_acc(rec, task, rng, [8, 16, 24], n_loops=t))
print("total", time.time() - t0)
