# MOLT-15 measured amortized repair

[CPU campaign 45922636784](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45922636784)
passed at e179e9dc3546e7a0240d112d90fb2ffd96b133df. One competent addressed
teacher and its native 16-component engineering fit; 128 TRAIN roots,
256 paired training edits, 32 fresh development roots, 64 paired dev edits.
Four 10,384-parameter repair controllers, 128 updates, batch 32 each.

| Arm | Changed suffix recovered /32 | Changed cut winner /32 | Same suffix preserved /32 |
| --- | ---: | ---: | ---: |
| Intact | 0 | 0 | 32 |
| Raw native swap | 21 | 22 | 32 |
| Native warm L2 | 22 | 22 | 32 |
| Native zero CE | 10 | 11 | 32 |
| Native warm CE | 21 | 22 | 32 |
| Random warm CE | 0 | 0 | 32 |
| Observer warm CE | 26 | 26 | 32 |
| Learned native CE | 27 | 27 | 32 |
| Learned native L2 | 20 | 21 | 32 |
| Learned random CE | 0 | 0 | 32 |
| Learned observer CE | 30 | 29 | 32 |

Changed-target mean residual dose for raw/native CE was 8.7317523,
learned observer CE 8.7229733. The controller fit times were 0.175-0.182
seconds each. Whole development evaluation elapsed 5.01857 seconds and
worker peak RSS was 409,169,920 B. No teacher updates. The observer chart
was derived from TRAIN-only average legal-margin Gramian; the solver and
learned controller used the same fixed chart. Privileged donor cut logits
are available to all repair policies. Cut winner recovery and suffix
recovery are distinct endpoints; soft carry can yield a different future
decision than its immediate argmax.

The learned observer CE development result exceeds raw native swap by
9/32 changed suffixes; native CE exceeds raw by 6/32. Learned observer CE
also exceeds native CE by 3/32. These are descriptive, correlated paired
observations at one frozen teacher and one controller seed. No population
claim, native specificity, promotion, or autonomous teacher training gain.
The best observer controller was selected descriptively after this panel.
Any fresh follow-up must freeze its roster and distinguish optimization
failures from physically unreachable legal decisions. Do not reuse these
development roots for confirmation.
