# LOGRANK-BPD-03: joint information diagnostic

Exploratory synthetic GRUs, no native BPD/TRM or confirmation. Run `python experiments/logrank_bpd_03/pilot.py --steps 500 --out receipt.json`, then `python experiments/logrank_bpd_03/audit.py receipt.json`. PyTorch 2.10 CPU. Four fresh initialization/split/permutation seeds, 500 updates each, 128 examples per minibatch, 16,644 allocated parameters per model, BBBA fixed four-bit schedule. Each root permutes query-to-coordinate mapping and holds out 128 of 256 memory vectors. Roots are isomorphic indexing matrices, not independent task families. No verifier at inference. No GPU or protected data. Compare full uniform evaluation (includes train vectors) to the 82.8125% one-way bound; heldout is separate.

Enumerate all eight stochastic query paths per local j using forced prefixes. Joint loss is task BCE minus .5 I(J;C) in nats. Marginal control sums three individual I(J;C_t) terms with identical coefficient and path enumeration. This is marginal mutual information, not pure entropy. Both hard task forward and soft auxiliary calculation use the same query head. The fixed and guided arms are structural and supervised controls, not matched supervision.

| Arm | heldout /4096 | full /8192 | hard query code counts | mean soft joint bits |
|---|---:|---:|---|---:|
| interactive | 2515 | 5084 | 2,4,2,2 | .076 |
| quadratic | 2530 | 5135 | 4,2,2,3 | .247 |
| marginal | 2801 | 5556 | 2,2,2,2 | .977 |
| joint | 3131 | 6246 | 4,4,4,4 | 1.780 |
| joint_quadratic | 3022 | 6084 | 4,5,3,4 | 1.796 |
| guided | 4007 | 8064 | 8,8,8,8 | 2.995 |
| fixed | 4062 | 8140 | 8,8,8,8 | 3 |

Marginal objective reaches mean sum 2.931 bits while producing only two distinct codes. Joint objective improves heldout by 616/4096 = 15.039 pp but does not achieve injective addressing. No unguided or information-regularized root beats the uniform-domain one-way ceiling. Guided and fixed do so on every root. No exact full matrix success. Quadratic head adds no consistent gain.

Exact query-collision bound: if the hard code partitions k coordinates into groups of sizes s_i, uniform accuracy is at most sum_i (s_i/2 + E|S_{s_i}|/2)/k for independent Rademacher walk S. For each group Bob has two reconstruction vectors; expected optimum increases with their Hamming distance by convexity and attains maximum at antipodal reconstructions. Thus the bound is attainable with unrestricted local computation. Independent audit checked group sizes 1..8, and verified all 114,688 saved train/heldout/full transcript records (57,344 full model-case records), Alice response consistency given x and code, Bob output consistency given j and transcript, exact bit counts, split identities, accuracy and ceiling. It audits saved finite outputs rather than checkpoint regeneration or runtime process isolation. Raw JSON was retained locally; its digest and all 28 aggregate rows are in summary_receipt.json. The raw file is not in this git commit.

Follow-up LOGRANK-BPD-04 uses a discrete assignment objective; it was chosen adaptively after observing joint-code collapse, reuses these development seeds, and cannot be treated as a prospective replication. No native intervention preflight or confirmation authorized.
