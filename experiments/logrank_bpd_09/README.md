# LOGRANK-BPD-09 and 09B: noisy traces and unseen compositions

Development, no confirmation. Six fresh generator worlds, 36 trained neural proxies plus 18 local teacher-only compiler controls. Protocol frozen before execution (200 updates and 0/15/35% noise). Each world supplies a public descriptor table for 128 distinct program addresses (64 train, 64 test). Models see descriptors for test programs at inference, not their labels. Seven-bit address plus one reply is a logical protocol cost, not a wire benchmark; shared table storage and synchronization are excluded. Operator/sign/end semantics are public. Learned token heads choose a component and stop; binary accumulator folds multiplication or minimum. All primitive tokens occur in training, including clean singleton final task labels. No unknown primitive or arbitrary-address generalization. Distinct training and evaluation memory IDs. All 384 held-out programs are semantically novel relative to training in their own worlds; 192 length four/five programs exceed training length. Dense vectorized computation still performs all slots, so semantic stop and active rank do not establish FLOP savings.

## Replay
```sh
cd experiments/logrank_bpd_09
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python pilot.py --k 16 --out receipt16.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python pilot.py --k 32 --out receipt32.json
OPENBLAS_NUM_THREADS=1 python audit.py receipt16.json receipt32.json > audit.json
OPENBLAS_NUM_THREADS=1 python repair.py receipt16.json receipt32.json > repair.json
python identifiability.py
```
Exact counts are in summary_receipt.json; raw receipts, full audit and repair artifacts are in the local bundle, not committed. Source hashes and remote blobs matched. Receipt hashes include wall times; independently reproduced numeric results need not have identical file hashes.

## Long-program accuracy (49,152 decisions per arm/noise)
| Noise | Staged | Joint | Teacher-only local compiler | Staged projected |
|---|---:|---:|---:|---:|
| 0 | 49152 | 49147 | 49152 | 49152 |
| .15 | 46922 | 46733 | 47182 | 47521 |
| .35 | 40569 | 36966 | 44087 | 42716 |

At 35% teacher noise staged gains 3603/49152 = 7.33 percentage points over joint on longer programs, yet teacher-only local compilation leads. Staged dense certificates across all depths at noise 0/.15/.35: 275/9/3 out of 384. Signed projection: 384/347/254. At .35 exact stop classification in 4/6 staged vs 2/6 joint; compiler uses known syntax. No statistical superiority claim: six worlds and strong structured priors. Independent replay checked 884,736 original and 884,736 projected decisions, exact rational dominance certificates (with conservative fp32 margin), support truth tables, label and split consistency, saved routes and counters. Dense certificate total 1501, projected 2886 across 3456 arm/query instances. Does not replay optimizer or certify equivalence of dense to projection globally. No native BPDTrainer run.

## 09B adaptive stronger controls
Both were added after viewing the main results; they are exposed-root diagnostic controls and do not count as a prospective comparison. Enforce known one-to-one mapping via maximum-weight assignment on route vote and absolute correlation matrices; at .35 this produces 47320/49152 (96.27%) longer-program correct, 327/384 exact overall functions (156/192 long). A clean-singleton compiler uses the same clean final task labels available to neural training. Each primitive has a unique exact signed-coordinate match on the training memories. All 384 unseen functions then have exact certificates and all 98,304 sampled decisions are correct. This bypasses teacher traces entirely; it is a necessary strong task-label control, not proof that general noisy mechanistic problems are solved. Uses known singleton semantics and signed-coordinate prior.

## Next
Remove singleton supervision, use controlled-rank parity incidence equations, compare exact GF(2) solver and a learned router, distinguish identifiable from unidentifiable tasks. See IDENTIFIABILITY.md. Native BPD smoke on previous 08 fixture and repository source regression succeeded at BPD commit 06180637ec465c1eefc3aa28a38acc826305367d (runs 45855915179 and 45855915243); this does not validate 09 or train native BPD.
