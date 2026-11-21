#!/bin/sh
set -eu
backend="$1"
gateway="$2"
limit="$3"
mode="${4:-model}"
case "$backend" in nf4|bf16) ;; *) exit 2;; esac
case "$mode" in model) software=False ;; mock) software=True ;; *) exit 2;; esac
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 HF_HUB_OFFLINE=1
export NAN_STORY23_TOKEN="$(cat runs/nan_story23/local_auth.txt)"
exec nice -n 19 /home/duganist/.venvs/nanbeige-prime-ed70053/bin/vf-run nanbeige_storyworld_suite \
  --env.taskset.backend "$backend" --env.taskset.software-only "$software" \
  --env.agent.harness.id null --env.agent.runtime.type subprocess \
  --env.agent.max-turns 1 --env.agent.timeout.setup 120 --env.agent.timeout.rollout 3600 \
  --client.base-url "http://$gateway:18764/v1" --client.api-key-var NAN_STORY23_TOKEN --client.timeout.read 3700 \
  --sampling.temperature 0 --sampling.top-p 1 --sampling.max-tokens 64 \
  --model "nanbeige-451ed48c-$backend" --max-concurrent 1 \
  --select.limit "$limit" --select.shuffle False --num-rollouts 1 \
  --run.name "nan-story23-$backend-$mode" --output-dir runs/nan_story23/prime
