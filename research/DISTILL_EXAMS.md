# LoopedTransformers Distill adapters v0.1

Canonical compiler: [Compactification-Theory Distill v0.1.0](https://github.com/MoralityLabAI/Compactification-Theory/tree/ac6ec9243d16f73117a986307851cd031bca7fa8/Distill), pinned at ac6ec9243d16f73117a986307851cd031bca7fa8. These source-locked sidecars define independent public design exams. No training-loop, frozen data, verifier or existing source changes. Use reviewed public development material only. Narrative derivations need separate review.

- PPT-A-B: `research/lean_cgrt_hrm400/ppt_gym/distill_exam.json`
- LRM-REV-03: `mechinterp/revision_memory_gym/distill_exam.json`
- AUX-GYM: `experiments/recurrence_monitorability_v0/aux_gym_v2_20261126/distill_exam.json`

From the pinned Compactification-Theory checkout, run `python -m Distill compile --adapter /absolute/path/to/adapter.json --out /tmp/exam.json`, followed by `python -m Distill prompt --packet /tmp/exam.json`. Native scoring stays within the original gym; the sidecars do not execute scorers or model calls. No automatic training export.
