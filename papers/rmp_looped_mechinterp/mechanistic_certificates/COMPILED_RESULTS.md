# MC-01 compiled relation audit

2026-11-11. **Passed** on code commit
`0793e23d5333693435a5a5176ac0fe9db03f14a2`.

GitHub Actions [run 43009551301](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43009551301),
job `130485418335`, completed successfully. It ran 39 unit tests in normal Python
and again with `python -O`, regenerated the exact controls, compiled the predicate,
and checked its emitted witness and every single nonconstant-wire mutation.

## Measured results

| Check | Result |
|---|---:|
| Unit tests, normal Python | 39 passed |
| Unit tests, optimized Python | 39 passed |
| Valid trace/field cases, BN254 and M31 | 712 accepted |
| Exhaustively enumerated candidate traces | 2,187 checked |
| Genuine trajectories in that enumeration | 81 accepted |
| Reference single-wire mutations, two fields | 108 rejected |
| Circom-compiled constraints | 70 |
| Compiled witness wires including constant | 55 |
| Compiled valid WASM-generated witness | 1 accepted |
| Compiled single-wire mutated witnesses | 54 rejected |

Compiler output: 48 nonlinear constraints, 22 linear constraints, 15 public
inputs, 39 private inputs, and no public output signal. This is a predicate on
public graph/boundary inputs and a private trace. The compiled example has N=3,
T=3. The N=16,T=16 count of 4,945 constraints elsewhere is a Python-emitter count;
that larger circuit was not compiled in this run.

## Negative-test correction

The first harness compiled successfully and accepted the valid witness, then
failed while trying to generate an invalid witness through WASM. The calculator
raised an assertion on `public_values_0`, despite the requested `--sanity_check 0`.
That run supplied no compiled negative-test result.

The successful harness generates the valid witness with normal compiler checks.
It reads the compiler's symbol map, verifies a one-to-one mapping onto all 54
nonconstant witness wires, and mutates each wire in turn. Every mutation is
serialized as WTNS, parsed independently, and evaluated directly against the
compiler's exported R1CS. All 54 are rejected there. This tests the constraints
without relying on the witness generator's early assertions.

## Reproducibility identities

Circom version: `circom compiler 2.2.3`.

Compiler SHA-256:
`85342c7ff332d948df7c0c50ecf201e6129349aef550ce873f3c811b79fe53a3`.

| Artifact | SHA-256 |
|---|---|
| trace.circom | 2cff28a5bec367b675b0b2f1af062adf8f1cac2a5ac23333aad43665d97e3dbf |
| trace.r1cs | 82c0a1f14eb33c12cc3974a3c37fac1800b8dce281df9d05b9468b63e512229d |
| trace_constraints.json | a061aa1b754859df7f20609482ff4340eeb2c1c59903e29180e7db5f413fa710 |
| trace.sym | 8de0c6d26ba4de06408ee3143187d05fc7fbbfa9ce6b523181b376c51ceb4c66 |
| trace_js/trace.wasm | 02c76fcfee81a25bddf492a606246f7fd321872bb88a48d20688ecd23bf24b1d |

The downloaded evidence archive, artifact `13850330454`, contains
`controls.json`, `compile.log`, and the full per-case `compiler_receipt.json`.
Its SHA-256 was checked after download:
`badadbf14ec8d72fbc3c5fd22f4cf255190de9c8a06dfc20692261383286dbe0`.
GitHub lists its expiry as 2027-02-09, so the source and this permanent report
remain the reproduction route after artifact expiry.

## Scope

This run checks the abstract relation and its Circom interface. No ZK proof or
trusted setup was generated. STWO and garbled-verifier backends were not run.
The neural capture adapter was tested with synthetic files; trained Q4/Q5
capture audits and neural causal interventions remain unperformed in MC-01.
Independently pinned SEM-ECHO execution is trusted by the adapter. Cryptographic
binding of a model's inference to a supplied witness remains a separate relation.
