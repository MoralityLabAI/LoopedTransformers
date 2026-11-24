# QUEUE43 local DELAY39 execution handoff

This is local supervision for the already-published DELAY39 experiment.
The worker, experiment limits, arm ordering, stage qualification and
measurement rules come from its frozen manifest. The new supervisor
starts exactly one NF4 process tree, then one staged BF16 process tree,
then performs the registered host graph scoring for each precision.
It keeps the five-GiB startup floor and the shared GPU mutex in the
original worker. Every worker verifies the published DELAY39 source
again at its own admission.

The supervisor authenticates the four predecessor processes by Windows
PID and creation FILETIME and waits on read-only process handles. It
requires the existing CACHE32 model and host terminal records and the
frozen inventory check. It imposes a finite 30-day predecessor and RAM
wait. No poll opens a live atomically replaced progress file.

Each pilot starts suspended and is assigned to a private Windows Job
Object before it can create descendants. Wait for zero active members
and the launcher exit code before opening terminal files or starting a
successor. Verify the admission, target, post-run checkpoint and
quantization receipts before allowing the next stage. A recorded native
qualification failure is a valid pilot outcome and retains unknown
intervention records. An ambiguous failure closes the queue. An owned job
that stays active beyond nine hours ends automatic sequencing and remains
under the pilot's safeguards. The supervisor closes its observation
handles without applying termination to a running tree.

Below Normal priority and one numerical thread apply to model workers
and registered host scoring. The queue creates a local append-only event
log with exclusive stage outputs; completed model receipts and reports
remain local. Publication of this supervisor or its receipts is a
separate action. This work does not read the sealed ARG-03 dataset.

Run from the canonical checkout with the registered Windows base Python.
The attempt directory is exclusive. Use the preflight command to verify
local bindings, source publication and current predecessor status:

```powershell
C:/Python312/python.exe -X utf8 -B -u -m experiments.nan_queue43.supervisor --observe
```

Omit `--observe` to start the bounded queue. `--freeze` creates the local
supervision manifest during commissioning.
