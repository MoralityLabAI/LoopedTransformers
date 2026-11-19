# DT05 post-pilot diagnostic erratum: top-k set overlap

**19 November 2026.** The first excluded eight-root autonomous-routing study was archived at BPD commit [`c24c676bad2f577cba94960093795c7dead5d218`](https://github.com/MoralityLabAI/BPD/commit/c24c676bad2f577cba94960093795c7dead5d218) with original receipt `benchmarks/08_autonomous_routing_study.json` and its SHA-256 in `benchmarks/ARTIFACT_MANIFEST.json`.

## Fault isolated

The prior implementation described `online_vs_teacher_topk_set_overlap_by_step` as `|S_online ∩ S_teacher| / k` but computed:

```python
(F.one_hot(m_on, C) * F.one_hot(m_tc, C)).sum(dim=-1).mean() / k
```

Both `m_on` and `m_tc` contain **ranked** top-k component indices. Elementwise multiplication aligns corresponding **rank positions**. The code divides each position's equality indicator by `k` and then averages those `k` positions, yielding **rank-position agreement divided by `k`**, with a maximum of `1/k` even when the ordered lists match exactly. The recorded `0.019–0.024` values should be multiplied by `12` to recover the ordered rank-position match fraction; they cannot recover the unordered component-set intersection. Two selections containing precisely the same components in a different order can have a reported value of zero. For independent uniform k=12 subsets of C=24 components, the true set-intersection fraction has expectation 0.5.

The fix measures `(m_on.unsqueeze(-1) == m_tc.unsqueeze(-2)).any(dim=-1).float().mean()` and rejects malformed input shapes. Regression cases cover equal sets in reverse order, half-shared sets, disjoint sets, and malformed inputs.

## What remains valid and what needs replay

The archived heldout exact-board accuracy, KL values, fixed-versus-online contrasts, factor equality checks, and per-arm timing have independent calculation paths. On eight **excluded engineering** roots, online was 27.34375% exact versus fixed 1.5625% (difference +25.78125 percentage points) at depth eight. On the three *postselected competent* roots, online 58.85417% versus fixed 4.16667% (+54.6875 pp). Privileged teacher-scheduled reached 81.77083% on those three roots. These are descriptive engineering results, and their causal mechanism cannot be assigned to component-set overlap until corrected.

The original rank-position-agreement sequence remains a historical observation and is retained byte-for-byte; it must not be cited as genuine set overlap or evidence of monotonic *set* divergence. The corrected values need a **fresh engineering-only local replay** of the exposed pilot inputs on the pinned code, with new hashes and a distinct output path. No archived receipt or original protocol should be overwritten.

The follow-up should separately record true unordered top-k overlap, ordered rank agreement, per-depth teacher-vs-online logits/KL, and delivered operator norm or intervention dose. Quantify whether the online-routing fidelity gap comes from mask support, ordering, accumulating state mismatch, or factor fit. A matched oracle controller can guide diagnosis but remains privileged and excluded from deployable inference.

All 59 reserved DT05 development roots `2605000..2605058`, DT04 confirmation roots, and other sealed cohorts remain unopened by this source-only fix.
