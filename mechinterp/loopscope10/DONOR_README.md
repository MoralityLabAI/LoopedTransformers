# LOOPSCOPE-10B reachable-donor source instrument

See the [prospective donor protocol](DONOR_PROTOCOL.md), committed before donor code at `94cba712cbbf4346716afaab674ccfd01604999d`, and the [EvalMath planning note](POWER_PREFLIGHT.md).

A legal source edit changes one digit at index (goal−1) mod 81, so exactly the goal's cyclic-copy label changes. The recipient and donor are both naturally reachable up to the intervention depth. The native hybrid edit is obtained by projecting their current-token state difference into the recipient's **energy/head-tangent nullspace**, then applying a frozen positive physical dose. Eight independently randomized equal-L2 controls occupy that same nullspace. An inverse-sign control, same-dose unprojected difference and full token donor (larger natural dose) are retained separately.

Source-only test uses excluded seed 61060 with a newly initialized and untrained 44K recurrent network, plus quadratic artificial potential:

```bash
python -m unittest discover -s mechinterp/loopscope10/tests -p 'test_*.py' -v
```

This stage proves input construction and native instrument execution. Reserved 31060 is ungenerated; original frozen LGA-02 and LGA-05 model/potential binaries are not accessed by the PR workflow. Scientific behavior is pending source-lock, one-time new-source execution and independent audit.
