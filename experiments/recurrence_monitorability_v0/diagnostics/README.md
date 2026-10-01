# Plateau diagnostics (2026-10-01)

These are scratch runs, kept as evidence for why the first pilot did not learn and
what changed. They are not registered cells: paths are hard-coded, nothing is
checkpointed, and each ran once with seed 0. All use `pointer_chase`, width 128,
4 heads, Adam with weight decay 0.01, 200 warm-up steps, and the first 1,000 to
1,600 validation examples at the depths trained.

Spec format: `arm:visits:max_depth:beta:lr:batch[:flags]`. `d0` adds zero-hop
examples (answer is the start node), `cur` unlocks depth `d+1` once depth `d`
reaches 0.9, `ds` supervises every visit.

## Depth 1 only, two untied layers (`diag_grid.py`)

| Batch | beta | lr | Zero-hop | Depth-1 accuracy | Log |
| ---: | ---: | ---: | --- | --- | --- |
| 64 | 0.25 | 0.001 | no | chance at 8,000 steps | `diag_grid1.log` |
| 64 | 1.0 | 0.001 | no | chance at 8,000 steps | `diag_grid1.log` |
| 64 | 0.25 | 0.003 | no | chance at 8,000 steps | `diag_grid1.log` |
| 64 | 1.0 | 0.0003 | no | chance at 8,000 steps | `diag_grid1.log` |
| 64 | 0.25 | 0.001 | yes | chance at 10,000 steps | `diag_grid2.log` |
| 64 | 1.0 | 0.001 | yes | chance at 8,000 steps | `diag_grid2.log` |
| 512 | 1.0 | 0.001 | yes | 1.000 at 1,000 steps | `diag_grid3.log` |
| 512 | 1.0 | 0.0003 | yes | 1.000 at 2,500 steps | `diag_grid3.log` |
| 512 | 1.0 | 0.003 | yes | chance at 2,000 steps (run cut short) | `diag_grid3.log` |
| 512 | 0.25 | 0.001 | no | 0.21 at 6,000 steps, rising | `diag_grid4.log` |
| 512 | 1.0 | 0.001 | no | 0.44 at 6,000 steps, rising | `diag_grid4.log` |
| 512 | 0.25 | 0.001 | yes | 0.17 at 6,000 steps | `diag_grid4.log` |

Chance is 0.0625. In every zero-hop run the zero-hop examples themselves were
learned within 1,000 steps.

## Tied loop, eight visits

| Spec | Result | Log |
| --- | --- | --- |
| depth 1, batch 256, beta 1.0, zero-hop | depth-1 accuracy 1.000 at 3,000 steps | `diag_grid5.log` |
| depth 1, batch 256, beta 1.0, no zero-hop | chance at 2,500 steps (run cut short) | `diag_grid5.log` |
| depths 0-8 uniform mixture, batch 256, beta 1.0 | chance on depths 1-8 at 7,000 steps | `diag_cur_uniform_mixture.log` |
| depths 0-8 with curriculum, batch 256, beta 1.0 | depths 1-3 at 1.00 by 5,500 steps; depth 4 flat near 0.45 from step 5,500 to 10,000 | `diag_cur_curriculum.log` |

## Reading

With a permutation as the mapping, the answer is uniformly distributed given any
single input token, so no part of the lookup circuit gets a gradient until the
other parts exist. The escape from that plateau is noise-limited: it does not
happen at batch 64 under any setting tried, and it happens within about 1,000
steps at batch 512 once two further conditions hold. Zero-hop examples first
teach the model to carry a node to the answer, and a residual branch scale of
1.0 rather than 0.25 lets the lookup form quickly. In a mixture over depths each
depth gets only a fraction of the batch, which puts the model back in the
noise-limited regime; an easy-to-hard curriculum avoids that.

The first pilot cell (`LOOP-T-pointer-s101`: batch 128, beta 0.25, uniform over
depths 1-8, no zero-hop examples) was at chance after 46,750 steps, about 6.0
million examples.
