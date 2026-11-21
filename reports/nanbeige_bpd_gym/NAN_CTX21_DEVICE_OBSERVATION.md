# CTX21: live 3050 device observation

During an observed 180.188-second interval, the device averaged **98.73% utilization**
(median 100%, minimum 82%, maximum 100%). Device memory use was 3,357 MiB of 4,096 MiB
in all 37 samples. The authenticated NF4 worker handle remained live throughout.
The device was busy while its clocks were thermally limited; an idle-device
explanation does not fit this particular interval.

| Measurement | Result |
|---|---:|
| Observation UTC | 2026-11-21T10:34:54.301069+00:00 to 2026-11-21T10:37:54.275104+00:00 |
| Samples | 37, scheduled every 5 seconds |
| Completed 4K development contexts | 236 to 249: 13 additional contexts |
| Observed closed-context throughput | 4.329 per minute |
| Temperature | 88–89 C |
| Graphics clock | 412–1237 MHz, median 682 MHz |
| Instantaneous unused device memory | 739 MiB |
| Software thermal clock limiter | Active at both additional limiter reads |
| Thermal counter increase in separate interval | 127.458 seconds over 127.252 wall seconds |

NVIDIA's detailed snapshot reports P0, software thermal slowdown Active,
and idle, software power cap, hardware thermal slowdown and hardware power
brake Not Active. It reports a GPU target temperature of 87 C; the observed
current temperature is 88–89 C. These are reported driver fields, not a new
thermal threshold or hardware configuration. The limiter-counter interval
overlaps only part of the sampling window and extends beyond it; do not
attribute its entire increase to the 180-second window or solely to Nanbeige.
The counter increase exceeds the host timestamp span by 0.206 seconds; treat
this as approximate driver telemetry, without inferring an exact duty cycle.

This observation concerns the registered CTX21 NF4 two-loop, one-token
condition, with all 22 decoder layers and the unchanged BF16 head on CUDA.
It adds no neural calls, grades, inference changes, fan/clock/power settings
or causal comparison. Its 13 closed contexts are ungraded development
outputs, not 13 successes. No 8K capacity or score is inferred from this 4K
window. FULL26's longer traces and genuine batched A/B execution remain a
separate frozen continuation.

NVML utilization and device memory cover all GPU activity, including system
work; they are not per-process kernel occupancy. Sampled memory is not the
full-run physical peak and unused driver memory is not a guaranteed future
allocation. The observer used the actual Windows worker handle, read-only
NVML queries and the append-only case journal. It did not read live atomic
progress files. Numerical workers and the separate math process were unchanged.

[Machine receipt](nan_ctx21_device_observation.json) ·
[All 37 timestamped samples](nan_ctx21_device_samples.jsonl) ·
[Detailed NVIDIA snapshot](nan_ctx21_device_limits.txt) ·
[Observation code](../../mechinterp/nanbeige_bpd_gym/observe_live_device.py).
Published sample line endings and snapshot trailing whitespace are normalized
for Git; the receipt retains original and published byte hashes. No telemetry
value changed. The receipt independently reconciles all sample counts/means, closed-context
counts and source hashes. All 584 FULL26 frozen bindings remain unchanged.
The continuous research goal stays active; the original 168-case test stays sealed.
