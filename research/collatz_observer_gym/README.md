# COL-OBS-01: what recurrent observations retain about a basin

This gym computes exact readout limits for the split Collatz family. Every
positive odd state in each declared finite core receives its eventual cycle
minimum as a target. The first sweep covers all 172 admissible parameters
`1 <= d <= 1025`, with 44,204 states in total. Parameter 2021 is a separate,
previously inspected stress case.

The calculation measures the information present in an observation. Its
readout chooses the best basin after seeing every label in the finite domain.
This access makes the reported score a hindsight oracle ceiling. Neural
training and causal intervention belong to the prospective protocol below.

## Model and target

For `d > 0`, `d = 1 (mod 4)`, and positive odd `n`, form `3*n+d` when
`n = 1 (mod 4)` and `n+d` when `n = 3 (mod 4)`, then remove every factor of two.
Each odd step emits `(u,a)`, where `u` identifies the triple branch and `a`
counts halvings. Both branches have `a >= 2`; the odd interval `[1,d]` is closed.
The code uses `gcd(d,6)=1` when it computes multiplicative quotient labels.

`q(n)` is the least residue in the orbit of `n mod d` under multiplication by
2 and 3. Every executed transition preserves this label. A basin label names
the minimum member of the eventual odd cycle. Several basins can share one q.

For observations `z(n)` on the complete core `X_d`, the exact best correct count is

\[
C(z)=\sum_z\max_b |\{n\in X_d:z(n)=z,\ B(n)=b\}|.
\]

The denominator is `|X_d|`. Macro accuracy gives each parameter equal weight;
micro accuracy gives each state equal weight. The receipt includes both.

## Observer arms and controls

| Arm | Input retained after r steps |
| --- | --- |
| Repeated quotient | q repeated r+1 times |
| Branch prefix | q and the ordered u sequence |
| Branch/exponent prefix | q and the ordered (u,a) sequence |
| Branch/exponent bag | q and the multiset of (u,a) pairs |

Depths are 0, 1, 2, 4, 5, 8, and 16. Four fixed hash permutations, with seeds
11, 29, 47, and 71, shuffle each arm's observations among states sharing q.
This preserves every cell size within every quotient. All controls are kept.
Hash permutations serve as capacity diagnostics; they carry no sampled-world
p-value. The raw full-state readout and a gcd readout give additional bounds.

## Executed finite results

| Observation | Depth | Best correct / 44,204 states |
| --- | ---: | ---: |
| Quotient | any listed depth | 37,317 |
| Branch prefix | 4 | 37,993 |
| Branch/exponent prefix | 2 | 39,971 |
| Branch/exponent prefix | 4 | 44,120 |
| Branch/exponent prefix | 5 | 44,204 |
| Branch/exponent bag | 4 | 42,233 |
| Branch/exponent bag | 8 | 43,922 |
| Branch/exponent bag | 16 | 44,087 |

At depth 4, the four matched shuffled prefix ceilings are 44,109, 44,117,
44,120, and 44,117. At depth 5, the ordered prefix and all four shuffles reach
44,204. The saturation therefore fails to distinguish dynamical reasoning
from the capacity to identify an individual state. Repeated quotient labels
stay at 37,317 throughout.

The known `d=2021` anchor has 1,011 odd states, 39 cycles, and six quotient
classes. Its quotient ceiling is 190/1,011. Four observed branch/exponent
pairs raise the ceiling to 965/1,011; five pairs identify every seed.

### Why the five-step result follows from arithmetic

For a fixed ordered prefix, write the accelerated composition as

\[
x_r=\frac{3^u x_0+B d}{2^A}.
\]

Starting at `B=A=u=0`, each edge with multiplier `c in {1,3}` updates
`B <- c*B + 2^A`, followed by `A <- A+a` and the corresponding u update.
The final state is odd, so

\[
x_0\equiv (2^A-Bd)(3^u)^{-1}\pmod{2^{A+1}}.
\]

Two seeds with the same prefix differ by a multiple of `2^(A+1)`. Since
`A >= 2r`, the prefix is injective on `[1,d]` whenever `2^(2r+1) > d-1`.
Five steps suffice throughout the sweep and the 2021 anchor.
`recover_seed` implements the congruence and replays the prefix before
accepting a recovered integer. The receipt records its recovery count at
all depths. This argument explains why a long ordered prefix forms an exact
code for a finite seed.

## Evidence contract

| Field | Value |
| --- | --- |
| Unit | Complete parameter graph d |
| Units / cases | 172 / 44,204 in the sweep; one separate anchor / 1,011 |
| Primary contrast | Information retained by each observer versus its occupancy-matched permutation |
| Physical dose | r emitted odd transitions, with r recorded per row |
| Scientific variation floor | Exact integer differences in the stated finite population |
| Matched-sham floor | Four complete occupancy-matched permutations, descriptive |
| Comparisons | Four observer families, seven depths, four controls per cell; all retained |
| Resolution / power | Exhaustive counts; no population confidence claim |
| Readout exposure | All finite target labels used to compute oracle ceilings |
| Causal controls | Prospective neural intervention design in NEURAL_PROTOCOL.md |
| Verdict | Exact finite result; saturated-prefix mechanism test fails discrimination |
| Provenance | Vendored core hash and all per-parameter counts in the receipt |

The core was cross-checked against a separately written traversal and quotient
BFS for every state and residue in the 172-parameter sweep. Seven tests cover
full-step acceleration, brute-force decoder optimality, quotient preservation,
monotone refinement, sham occupancies, exact prefix decoding, and replay.

## Reproduce

Run from the repository root with Python 3.10 or newer. The standard library
supplies every dependency.

```sh
python -m unittest research.collatz_observer_gym.test_observer_gym -v
python research/collatz_observer_gym/observer_gym.py \
  --check research/collatz_observer_gym/results/exact_receipt.json.gz
```

To regenerate the compressed full receipt and readable summary:

```sh
python research/collatz_observer_gym/observer_gym.py \
  --out /tmp/col-observer.json.gz --summary /tmp/col-observer-summary.json
```

`--export /tmp/col-development.jsonl` writes public development cases with
input fields and an explicit `evaluation_only` target field. Those cases have
already been exposed to the finite audit. A model adapter must choose its
allowed input projection before loading target labels.

The complete receipt is gzip-compressed with a fixed timestamp. The
[readable summary](results/summary.json) contains the aggregates and every
anchor row. [NEURAL_PROTOCOL.md](NEURAL_PROTOCOL.md) sets the next research gate.

## Source trail

The canonical mathematical implementation belongs to
[MathCan's Collatz resonance program](https://github.com/patrickdugan/MathCan/tree/research/collatz-resonance/research/collatz_resonance).
Its vendored `collatz_core.py` has SHA-256
`24ff52171aedf890c33439896ab5e71de65d9b05d557ba61d162f3623aa5dbca`.
The finite law and proof above can be audited from this package alone.

The evaluation design follows the repository's
[EvalMath mechanistic-power skill](https://github.com/MoralityLabAI/EvalMath/blob/main/skills/evalmath-mechinterp-power/SKILL.md)
and [FSI-01 evidence gate](https://github.com/MoralityLabAI/EvalMath/blob/main/paper/finite_sample_inference.md).
Their inspected blob hashes were respectively
`c1c60a502e0e512b77387056da4e1036c68b09dc` and
`1e5d5c6ce961ed2cc54f6f65818db6e5e1715501`.
