# COE-09 frozen synthetic baseline, measured before native inference

## Executed controls, 2026-11-15

Two independent **already trained synthetic** tied-attention cohorts from COE-04 and COE-05 were replayed. All heads were selected in the original fit-only studies and their checkpoint hashes, 64-of-64 task correctness admission, 4 fit/4 heldout context split, and original edge scores were checked before this decoder ran. Each line has 32 held-out label observations, one for each of eight source symbols in each of four heldout distractor contexts. The model's own training had included all 64 input combinations. These are **probe transfer** outcomes, not fresh pretrained generalization.

| Prior source | Seed | Original temporal pair hits /112 | Ordered correct /32 | Cyclic correct /32 | Count correct /32 | Final correct /32 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| COE-04 | 2406 | 90 | 15 | 12 | 12 | 8 |
| COE-04 | 2407 | 72 | 11 | 11 | 11 | 5 |
| COE-04 | 2408 | 79 | 16 | 15 | 15 | 8 |
| COE-05 | 2411 | 82 | 11 | 8 | 8 | 8 |
| COE-05 | 2412 | 70 | 9 | 8 | 8 | 8 |
| COE-05 | 2413 | 91 | 14 | 11 | 11 | 8 |
| **Six frozen models** | | **484/672** | **76/192** | **65/192** | **65/192** | **45/192** |

All six cohorts passed the original model-task admission gate, meaning the base models correctly returned the source symbol on every one of their 64 already trained examples. The COE-09 prototype decoder preserves the original selected threshold and weight vector. Cyclic and dihedral decoding were identical to positive-count decoding on each seed. The ordered gain is 11/192 labels, spread over four of six models; it remains dependent on observation phase. The positive-control class-label chance rate for uniform eight-class labels is 12.5%, or 24/192, but our primary null is conditional *fit-label reassignment*, with deterministic tie handling and 10,000 permutations per seed.

For the three COE-04 seeds, conditional shuffle upper-tail probabilities for **ordered** are 0.0041, 0.0059 and 0.0008, respectively; cyclic values are 0.0223, 0.0059 and 0.0008. For COE-05 seeds 2411/2412/2413 the ordered probabilities are 0.0340, 0.0219 and 0.0002, with cyclic values 0.1654, 0.1468 and 0.0044. These are exploratory post hoc tests on already reported heads with correlated predictions, and are not an independently preregistered detection claim. **No paired label-accuracy p-value was obtained for a pretrained TRM.**

The complete replay receipts record every prediction, the null distribution summaries, source artifact SHA-256, and per-context correct counts. Both original compressed activation bundles and original fitted heads reside in `MoralityLabAI/Compactification-Theory/research/chromatic_observer_gym_v1`, merged via PR #143.

## Native experiment gate

`PROTOCOL.md` registers a new task for the actual pretrained `gaoxin492` TRM using 18 heldout boards and four original COE-07 selected heads in four subspaces. The native experiment requires SHA-verified activation arrays and the original head receipt. Current local runtime has no native 7M weights and GitHub Actions did not start its CPU inference job. Native alignment and causal effects are **pending measurements**. No numerical synthetic result above should be described as pretrained or causal. A genuine mechanistic result must pass the native label-alignment gate, survive the conditional fit-label shuffle, and show an intervention effect larger than same-radius random tangent controls with current-logit parity for dark perturbations.
