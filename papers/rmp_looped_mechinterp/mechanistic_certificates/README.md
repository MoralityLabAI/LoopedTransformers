# RMP-MC-01: mechanistic circuit certificates

Date: 2026-11-11. Status: exact controls and capture-interface tests executed.
Trained Q4/Q5 evaluation requires their external checkpoint captures. This patch
uses the existing SEM-ECHO adapter at `80d2ce8d073c6b634eab56f94c3f24c1a3c3a9e1`.
It leaves its training, readouts, and interventions unchanged.

## What this adds

A source-linked audit consumes `full_trace.npz` and `metrics.json` from
`research_gym.scripts.loopscope_semantic_echo`. It checks the candidate that the
model's own head advances exactly one pointer hop per recurrent visit. The input
supplies q0. All later labels come from the captured own-head logits. The verifier
recomputes the symbolic oracle independently and reports every item, including
items with a correct final answer and incorrect intermediate states.

There are separate verdicts for the abstract trace relation and agreement with a
trusted capture. Causal fidelity remains `unmeasured` for neural captures. A second
API, `audit_finite`, checks exact commutation and readout consistency under every
listed operation, including interventions, on a declared finite state domain.

An explicit R1CS relation and a Circom emitter provide the backend seam. The
pointer relation follows the earlier Algernon-02 circuit mirror in
Compactification-Theory. Its repeated quadratic step is an existing baseline.
The additions here concern source linkage, the SEM-ECHO file interface, and the
finite intervention tests. The M31 arithmetic mirror supplies a field-portability
check. A STWO AIR integration remains future work.

## Local execution

From the repository root:

```bash
python -m unittest discover -s tests -p test_mechanistic_certificate.py -v
python -O -m unittest discover -s tests -p test_mechanistic_certificate.py -v
python -m research_gym.scripts.mechanistic_certificate controls --out /tmp/mc01
```

The exact controls use the standard library. Capture-interface tests use NumPy.
The controls check 712 valid trace/field combinations, enumerate 2,187 candidate
traces, and reject 108 single-wire mutations across BN254 and M31. Exactly 81 of
the enumerated traces satisfy their fixed program, start and end claims.

The shadow-register control agrees perfectly on its natural trajectory. An
intervention exposes that the chosen variable fails to track the output-driving
register. Another control submits a correct oracle witness against a different
captured trajectory with the same endpoint. It passes the abstract relation and
fails the trusted-capture comparison.

`controls.json` is the machine-readable receipt. Tests construct synthetic NPZ
fixtures to validate the import contract; those fixtures carry no neural result.

## Audit an existing SEM-ECHO capture

Obtain identity pins from an independently trusted execution record. The hashes
identify files and a checkpoint. They do not establish honest execution on their
own. Each command audits one capture, without selecting only successful items.

```bash
python -m research_gym.scripts.mechanistic_certificate audit \
  --trace /path/to/run/full_trace.npz \
  --metrics /path/to/run/metrics.json \
  --expected-trace-sha256 "$TRACE_SHA256" \
  --expected-metrics-sha256 "$METRICS_SHA256" \
  --expected-checkpoint-sha256 "$CHECKPOINT_SHA256" \
  --out /path/to/mc01-audit
```

The importer checks file pins and source dimensions, verifies the graph encoded
in the original tokens, compares predictions with the captured logits, requires
the native state/head parity receipt, and recomputes the oracle. Numerical model
execution remains trusted to the SEM-ECHO capture process. It is neither replayed
nor proved by this import. `--allow-untrained` explicitly permits a smoke capture;
its status remains `UNTRAINED_SMOKE`.

An alternative internal algorithm can fail this fixed hop-per-loop hypothesis
while solving the task. Every rejection therefore carries its scope. Q4/Q5
outcomes are already known; these models are a development set. A prospective
prediction claim requires fresh seeds after freezing the whole procedure.

## Compile the abstract checker

The CI job downloads official Circom 2.2.3 and verifies its published SHA-256:
`85342c7ff332d948df7c0c50ecf201e6129349aef550ce873f3c811b79fe53a3`.
It uses `--O0 --sanity_check 0`, generates WASM witnesses, and checks the compiler's
own exported R1CS constraints with an independent WTNS parser. Disabling
witness-time checks ensures that rejection is tested at the constraints.

```bash
python -m research_gym.scripts.mc01_compiler_parity \
  --circom /path/to/circom --out /tmp/mc01-compiled
```

The compiled test expects one valid witness and rejection of all 54 individual
nonconstant-wire mutations. It reads the emitted symbol map rather than assuming
that compiler wire indices match the Python reference. A completed run writes
`compiler_receipt.json` with tool and artifact hashes. The local environment lacks
a Circom binary; consult the CI receipt for the separate compilation gate.

There is no trusted setup or proof generation in this job. A garbled verifier or
recursive proof can wrap this relation after the backend is validated. Such a
wrapper retains the relation's semantic scope. A proof that the neural model
produced the witness would need execution attestation or proved inference with
explicit arithmetic and decoder semantics.

## Resources and scope

For N=16 and T=16, the unoptimized reference has 4,945 constraints, 288 public
wires and 4,368 private wires. These are arithmetization counts. They are not
STWO proving costs or garbled-gate counts. Every public graph entry and both
boundary vectors are constrained. Model hashes are excluded from the circuit
until a relation actually binds them to inference.

[The theory note](theory.md) states the soundness obligations and handwritten
proofs. The finite checks are executable; Lean verification has not been run.
