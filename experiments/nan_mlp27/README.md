# Conditional bounded-MLP full gym

NAN-MLP-27 runs only after verified FULL26 termination and a source-bound
CUDA/input-capacity failure. It retains every FULL26 source, parser, generation
budget and control: 1,549 baseline contexts and 216 A/B pairs per precision.
Both precisions run fresh if qualified. No original output or missing grade
is repaired. Otherwise the stage is explicitly unrun.

The new condition bounds the gate/up activation workspace while retaining one
full down-projection/BPD invocation per layer. Padded reached-state capacity
probes precede scored inference, including a hard required B1/short-B2 gate.
[Protocol](PROTOCOL.md) · [Configuration](config.json) ·
[Installed package pins](ENVIRONMENT.json) ·
[Progress report](../../reports/nanbeige_bpd_gym/NAN_MLP27_PROGRESS.md).

The supervisor waits on an authenticated actual FULL26 handle and any surviving
children. It stops no existing model, host or math process. Live supervision
uses actual handles and append-only journals; closed artifacts are read only
after their writers finish. One numerical CPU thread and Below Normal priority
apply throughout. The original 168-case test stays sealed. This finite stage
does not end the continuous goal.
