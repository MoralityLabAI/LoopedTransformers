# Fresh NF4/BF16 competence pilot

Read [PROTOCOL.md](PROTOCOL.md). The freeze includes all new source, bounded
parser, host grader, queue, criteria and tokenizer receipts before pretrained
inference. Raw model artifacts remain local. Only public procedural source
fixtures and sanitized report aggregates enter Git. The original 168 cases
stay sealed. Earlier failed gates are preserved.

Development preparation (not commands to replay a completed neural run):

    python -S -m unittest discover -s experiments/nan_comp14/tests -v
    python -S -m experiments.nan_comp14.sources
    python -m experiments.nan_comp14.tokenize --snapshot <pinned-local-snapshot> --compactification-root ../Compactification-Theory-arg02
    python -S -m experiments.nan_comp14.freeze

Commit and push the freeze BEFORE using the finite queue, from this repository
root with all four numerical thread environment variables set to 1:

    C:\Python312\python.exe -X utf8 -B -u -S -m experiments.nan_comp14.queue --python D:\codex-artifacts\NanbeigeQ4Pilot-20261117\venv\Scripts\python.exe --snapshot D:\codex-artifacts\NanbeigeQ4Pilot-20261117\snapshot\451ed48c3273ecef7ea8faaa43c31ce529763bb1 --compactification-root ../Compactification-Theory-arg02 --freeze-commit <pushed-full-commit> --publish

The queue checks that the freeze is an ancestor of the pushed current branch,
refuses an existing run directory, checks for other known live model workers,
and runs NF4 then matched BF16 with independent audits. The broader continuous
research goal remains active after this finite pilot. No optional integration
choice or new approval is needed for this already authorized local workload.

[Streaming report](../../reports/nanbeige_bpd_gym/NAN_COMP14_PROGRESS.md)
