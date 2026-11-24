#!/bin/sh
set -eu
backend="$1"
gateway="$2"
case "$backend" in nf4|bf16) ;; *) exit 2;; esac
export PYTHONPATH="$PWD"
export NAN_PRIME_LOCAL_TOKEN="$(cat runs/nan_prime06/local_auth.txt)"
exec /tmp/nanbeige-prime-ed70053/bin/vf-run nanbeige_bpd_suite \
  --env.taskset.scope local --env.taskset.include-copy-controls True \
  --env.agent.harness.id null --env.agent.runtime.type subprocess \
  --env.agent.max-turns 1 --env.agent.timeout.setup 120 --env.agent.timeout.rollout 600 \
  --client.base-url "http://$gateway:18761/v1" --client.api-key-var NAN_PRIME_LOCAL_TOKEN \
  --sampling.temperature 0 --sampling.top-p 1 --sampling.max-tokens 192 \
  --model "nanbeige-451ed48c-$backend" --max-concurrent 1 \
  --select.limit 162 --select.shuffle False --num-rollouts 1 \
  --run.name "nan-prime06-$backend" --output-dir runs/nan_prime06/prime
