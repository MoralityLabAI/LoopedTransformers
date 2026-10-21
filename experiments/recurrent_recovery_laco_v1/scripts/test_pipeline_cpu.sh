#!/bin/bash
# End-to-end CPU smoke test of every code path on a tiny random Qwen3 (no GPU needed).
set -e
PY="${LOOPREC_PYTHON:-python}"
cd "$(dirname "$0")/.."
OUT=runs_test
COMMON="--device cpu --bs-text 2 --bs-task 4 --n-text 4 --n-long 2 --n-task 4 --n-mc 6 --no-latency --out-root $OUT"
$PY scripts/make_tiny_model.py --out checkpoints/tiny_base --layers 8
$PY -u scripts/train_student.py --name tiny_A --cond A --base checkpoints/tiny_base --steps 6 --rank 4 --eval-base-too $COMMON
$PY scripts/run_laco.py --teacher checkpoints/tiny_A --n-remove 3 --device cpu --calib-seqs 8 --out manifests/tiny_laco.json > /dev/null
$PY -u scripts/train_student.py --name tiny_B --cond B --teacher checkpoints/tiny_A --laco manifests/tiny_laco.json --steps 0 $COMMON
$PY -u scripts/train_student.py --name tiny_C --cond C --teacher checkpoints/tiny_A --laco manifests/tiny_laco.json --steps 6 --rank 4 $COMMON
$PY -u scripts/train_student.py --name tiny_D --cond D --teacher checkpoints/tiny_A --laco manifests/tiny_laco.json --steps 6 --rank 4 --loops 4 --hid steps --eval-loops 1,2,4,8 --grad-ckpt $COMMON
$PY -u scripts/train_student.py --name tiny_E --cond D --teacher checkpoints/tiny_A --laco manifests/tiny_laco.json --steps 0 --loops 4 --eval-loops 1,2,4 $COMMON
echo PIPELINE_OK
