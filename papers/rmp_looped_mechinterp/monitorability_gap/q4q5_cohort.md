# RMP-MG-01 original Q4/Q5 trained-checkpoint evaluation

This runbook extends the monitorability-gap protocol. The original Q4/Q5 weights are gitignored and external to the repository. Their training receipts and expected SHA-256 hashes are committed. The cohort runner rejects missing or mismatched weights and never substitutes a randomly initialized model.

## Frozen cohort

Q4 comprises lambda 0, 0.01, 0.1, and 1, with seeds 1 to 3, totaling 12 models. Receipts live under experiments/recurrence_monitorability_v0/loopscope_toy/runs/ARM/receipt.json.

Q5 comprises lambda 0 and 0.01, with seeds 4 to 9, totaling 12 models. Receipts live under experiments/recurrence_monitorability_v0/loopscope_toy_q5/runs/ARM/receipt.json.

Weight files are expected in the corresponding loopscope_toy/ckpt and loopscope_toy_q5/ckpt directories. Alternate locations can be supplied with --q4-dir and --q5-dir. Previous local checkout notes mention C:\Users\patri\OneDrive\Documents\GitHub\LoopedTransformers, with possible additional storage on another drive.

These Q4/Q5 models have already been examined for extrapolation. The experiments are retrospective development studies. New trained seeds with a frozen model-level prediction rule are required for prospective claims.

## Preflight on the machine holding the weights

From the repository root:

    python -m research_gym.scripts.loopscope_mg_cohort --out experiments/recurrence_monitorability_v0/mg01_preflight

The inventory.json contains every expected model and the actual checkpoint checksum when the file exists. Possible states are READY, MISSING and HASH_MISMATCH. Exit code 2 indicates a missing or mismatched selected checkpoint. Inventory is written even when preflight is blocked.

## Exploratory pilot

The pilot chooses arms deterministically, alternating lambda settings before taking additional seeds. It does not inspect outcomes to select models.

    python -m research_gym.scripts.loopscope_mg_cohort --run --pilot-per-cohort 2 --assay both --targets 8 --observations cursor_logits --probe-steps 100 --causal-items 1 --device cuda --dtype float32 --out experiments/recurrence_monitorability_v0/mg01_pilot

This runs a fixed-target bounded-probe fit and a native cursor-readout Jacobian assay per selected model. The optional --observations cursor_logits all_token_logits broadens the observation channel and increases Jacobian cost.

## Full Q4/Q5 probe cohort

    python -m research_gym.scripts.loopscope_mg_cohort --run --assay probes --targets 8 16 --device cuda --dtype float32 --out experiments/recurrence_monitorability_v0/mg01_full_probes

Subprocesses run sequentially. An execution_receipts.json file is updated after each model. Use --resume with an unchanged command and output directory after interruption. Resume verifies the checkpoint hash, recorded complete argv and a successful prior process. Other nonempty outputs are left intact.

By default all selected checkpoints must be READY. The --allow-partial switch explicitly permits evaluation of existing hash-valid subsets, with missing-model coverage recorded in the summary.

## Scientific interpretation

The fixed-target probe uses the same label Z_K = P^K(s) at every visit, with prompt access available to the baseline. The sensitivity assay measures whether future logits change along state directions with a locally null current observation derivative. The observation channel here is the model's native task readout. No natural-language chain-of-thought monitor or deception label is evaluated.

Summary statistics are first reported per checkpoint, lambda and training seed. Visits and interventions are repeated observations within each seed and must not be counted as independently trained models.

## Validation boundary

The original 24 trained Q4/Q5 weights were unavailable in the ChatGPT runtime. A separate newly trained small CPU fixture verifies the end-to-end runner and hash gate. The CI fixture builds synthetic training receipts and weights in a temporary folder, explicitly labeled NEWLY_TRAINED_CPU_FIXTURE. These data establish integration correctness and do not measure the original cohort.

For local checks:

    python -m unittest discover -s tests -p test_loopscope_mg_cohort.py -v
    python -m unittest discover -s tests -p test_loopscope_mg_cohort_native.py -v
