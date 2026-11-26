# LOGRANK-BPD-02: learned hard-bit indexing pilot

Status: development-only, 20 CPU-trained synthetic GRU models. Continuation of LOGRANK-BPD-01; no native TRM or BPD model was trained.

## Reproduce

`python experiments/logrank_bpd_02/pilot.py --steps 350 --out receipt.json`

PyTorch 2.10.0+cpu, one thread, deterministic algorithms, four initialization/split seeds (101, 211, 307, 401). Each arm receives the same initial random seed and same sampled training batch sequence within a root. Each has 16,644 allocated parameters; active parameter use and compute differ with the speaking schedule, and the fixed arm's query head is unused in the task path. The guided arm uses extra query-label supervision. 350 Adam updates at lr .003 and batch 128. Both agents have separate width-32 GRUs, input encoders and message heads. Every inference transcript contains exactly four hard binary messages. Bob produces the local final output. Training uses a straight-through gradient through the hard channel; gradients are not counted as inference communication. Evaluation asserts bit/logit equality with explicit integer reserialization and checks one-way messages are independent of Bob's input, and interactive query prefixes are independent of Alice's input, across the entire domain. These are in-process structural and replay checks, not sandbox adversarial side-channel verification. No process isolation, checkpoint archive, native BPD factorization or causal intervention performed.

For each root, split the 256 possible Alice vectors into 128 train and 128 held-out vectors; evaluate all eight query coordinates on each. The full evaluation covers 2,048 cases, including training vectors, and is the only one to compare directly with the uniform-domain bound. Four independently initialized models per arm share a single task matrix: these are independent training seeds, not four independent generator families. The raw per-case receipt was saved locally (SHA256 in summary_receipt.json); the committed summary preserves all root-level totals. Source git blob matches the locally executed source. No confirmation cohort, EvalMath causal power gate or native experiment registration. This is an exploratory diagnostic; no significance or mechanistic causation claim.

## Uniform-domain one-way ceiling

Any deterministic fixed four-bit one-way protocol has at most 16 reconstruction vectors, since Bob can decode its answer for each coordinate from the same message. For 256 uniformly sampled vectors, at most 16 have distance zero to any of these reconstructions, and at most 16*(1+8)=144 lie within distance one. The sum of distances is therefore at least (256-16)+(256-144)=352 (count the first and second error layers). Uniform pair accuracy is <=1-352/2048=82.8125%. This is a certified upper bound, not a tightness claim. Four-bit interactive query-and-response achieves 100% by construction.

## Results

| Arm | Held-out correct / 4096 | Full correct / 8192 | Query code counts | Roots above one-way ceiling |
|---|---:|---:|---|---:|
| oneway | 2450 | 5009 | n/a | 0 |
| interactive | 2606 | 5255 | 1,2,1,3 | 0 |
| quadratic | 2539 | 5121 | 2,1,1,3 | 0 |
| guided | 3688 | 7499 | 8,8,8,8 | 4 |
| fixed | 3935 | 7892 | 8,8,8,8 | 4 |

All 20 runs completed. No model reached all-pair exact correctness on the complete matrix. Guided and fixed query arms exceeded the proven uniform-domain one-way accuracy bound in all four seeds. Unguided query collapse was observed; paired products in the message head did not prevent it. Held-out guided minus interactive difference is +26.42 percentage points; no causal attribution to any individual BPD component.

## Next development design: LOGRANK-BPD-03

Hypothesis: an address-coverage objective that needs no external query code labels can prevent collapse of the discrete communication code. Under uniform J, the query prefix C in {0,1}^3 has capacity three bits. For deterministic mapping, H(C|J)=0 and H(C)<=3; achieving H(C)=3 implies eight distinct codes. For stochastic training, compute exact p(c|j) by enumerating all eight autoregressive bit paths (each prefix receives its own Bob hidden state) and calculate I(J;C)=H(C)-H(C|J). Add negative mutual information to task loss, optionally using a predeclared schedule. Marginal per-bit entropy alone is insufficient: the code 000 or 111 can have each marginal balanced and still convey only one bit. Do not approximate joint entropy by summing marginal entropies.

Keep original arms and add unsupervised joint-information, marginal-entropy sham, parameter-matched linear and quadratic heads with the same query objective. Calibrate on new development seeds/families, freeze architecture and budget prior to confirmation. Distinguish training guidance from inference transcript cost. Record full code histogram, minimum pairwise query-code Hamming distance, query collisions, message fidelity, exact matrix success, train/heldout accuracy, and recovery/corruption under matched component or bit perturbations. A distinct code can still fail to read the requested value; count address coverage and response decoding separately. Add a structured learned query-to-address readout as a ceiling/control, with explicit local computation and RAM accounting. For scaling k=16 or larger, preserve private input separation and sample held-out rows when exhaustive evaluation is infeasible; never compare a sample to a certified uniform ceiling without a suitable uncertainty analysis. Use EvalMath independent-root sham variation/preflight before native mechanism claims, with a new sealed confirmation population.
