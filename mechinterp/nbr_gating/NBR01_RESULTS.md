# NBR-01/01B registered result

The registered 144-board experiment completed in Actions run 45519229947
(PR #355, registration 81c32e236a49b5660ea2bff07cfb46acdc1acca2,
workflow head b5c68b9132b54850557aa878ac843c23c6a4b96f). Artifact
14979514171 (nbr-registered) contains 153 files and the complete outcome
receipts. Uploaded ZIP SHA256:
2521bc7b3635c865b25463d433e71e26071c7df6b6ba21727c4425784cb412b6.
Source roster SHA256:
763ea63c8565245928a7dd59f7b56b7916fad6e2f8d3126075a801c56a36f224.
Independent source admission, inference, fixed-basis replay and arithmetic
gate audit succeeded. The gate audit does not independently rerun model
inference or Jacobians.

## Registered verdicts (all 48 boards valid in each family)

| Gate | Eval mean [95% CI] | Rep mean [95% CI] | Verdict |
|---|---|---|---|
| NBR-01, raw same-ID minus shuffled | 0.74710 [0.66435, 0.82356] | 0.81479 [0.76506, 0.85482] | PASS |
| NBR-01B, residualized native minus fixed mixture | -0.00679 [-0.01026, -0.00362] | -0.01072 [-0.01438, -0.00727] | FAIL |

The fixed-basis contrast fails the preregistered 0.05 threshold in both
families. In both families its mean is slightly negative and its
descriptive bootstrap interval excludes zero. Stalled-only means are
-0.00477 [-0.01094, 0.00079] and -0.01243 [-0.01888, -0.00663].
Singles-only means are -0.00881 [-0.01225, -0.00567] and
-0.00902 [-0.01269, -0.00576]. These are prespecified descriptive strata.

Native residualized temporal cosine was 0.74034 / 0.80824 (eval/rep);
fixed mixture cosine 0.74713 / 0.81896. Covariance-matched mixtures
are fixed across ACTs. Direction stability is real, and the basis-specific
interpretation is unsupported by this test. The result does not reject
native neurons as useful activation-gated computational features, and
does not test language-model concept polysemanticity.

## Continuation

NBR-02A is an explicitly exploratory replay on these already observed
boards. Use exact orthogonal coefficient rotations preserving the full
physical and first-order transported writes to measure sparse
reconstruction at equal coordinate budgets. For a future registered
NBR-02, admit a new source roster, use the development replay only to
design one frozen contrast, compare SAE/PCA controls and run genuine
finite causal interventions. Keep NBR-01 and NBR-01B verdicts unchanged.
