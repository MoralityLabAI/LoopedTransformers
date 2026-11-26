# LOGRANK-BPD-05/06: hierarchical address scaling and retrieval factorization

Development evidence, not confirmation. CPU, PyTorch, width 32, 500 Adam steps per model. Two fresh seeds per k, isomorphic indexing tasks. All 16 recurrent models completed. Flat and hierarchical arms had identical allocated parameter counts at each k, but unequal auxiliary work. Fixed query is a structural control. The hierarchical rank-16 projection is a linear factorization before a scalar head, not a native BPD intervention. No GPU, no protected data.

## Reproduce
```sh
python experiments/logrank_bpd_05/pilot.py --k 16 --out receipt16.json
python experiments/logrank_bpd_05/pilot.py --k 32 --out receipt32.json
python experiments/logrank_bpd_05/retrieval.py receipt16.json receipt32.json --out retrieval_receipt.json
```

## 05: scaled recurrent addressing
| k | arm | root 1 correct | root 2 correct | total per root | codes per root | exact rows root 1/2 (of 256 each) |
|---|---|---:|---:|---:|---:|---|
|16|flat|2687|2915|4096|16|0/0|
|16|hier|2863|2977|4096|16|0/1|
|16|hier_factor|2680|3047|4096|16|0/0|
|16|fixed|3029|3289|4096|16|1/6|
|32|flat|4676|4731|8192|32|0/0|
|32|hier|4686|4701|8192|32|0/0|
|32|hier_factor|4663|4693|8192|32|0/0|
|32|fixed|4746|4731|8192|32|0/0|

Every hard address map injective. Retrieval is the apparent remaining bottleneck. Hierarchical assignment scores k log2(k) path positions per step vs k^2 log2(k) in the naive flat enumeration. Balanced greedy assignment chooses the upper half of each prefix group's logits, so its target is a bijection; global assignment optimality is not guaranteed. Timing does not establish speedup: at k=16 flat training took 6.42/9.49s vs hierarchical 9.71/7.34s; at k=32 flat 12.37/11.38s vs hierarchical 11.85/10.94s. Both dimensions ran concurrently, so timing is descriptive. Training and heldout transcript replays: 196608 records.

## 06: adaptive structured retrieval
Freeze each of the four learned hierarchical address maps. With local bipolar memory x and training verifier labels, accumulate label-coordinate correlations by address and select the signed coordinate of maximal absolute correlation. A single returned bit is decoded directly by Bob; this changes the local computation and training contract compared to 05. Compare to rank k/2 and rank k factorized static linear readouts trained 500 steps using the same training pool. Each row gets a separate address. All 1024 heldout vectors per root are scored across all coordinates.

| Arm | heldout correct / 98304 | exact full-domain certificates |
|---|---:|---:|
| sparse correlation | 98304 | 4/4 |
| rank half | 80692 | 0/4 |
| rank full | 98304 | 1/4 |

Sparse matrices equal coordinate permutations exactly, verified as integer rows; all domains certified. For a static affine threshold readout to return x_i on every bipolar input, its weights must be strictly diagonally dominant after reordering (see MathCan rank obstruction); hence full rank is necessary. Three dense full-rank readouts and all four rank-half readouts have explicit rational counterexamples to exact correctness. Perfect finite sample accuracy is not a proof. A sparse coordinate map needs k indices and k signs, at most k(log2(k)+1) stored bits excluding the address map and metadata. Rank and storage are separate. This is a supervised structure-exploiting diagnostic on selection-exposed task roots, not a general learned verifier or native BPD model.

Independent audit verified labels, counters, local transcript dependencies, disjoint split IDs, exact sparse row equality, rational dominance margins and seven negative-margin witnesses; 294912 additional retrieval prediction/label records. It did not independently replay every dense float matmul or checkpoint. No inferential claim on two roots/size.

## Provenance
Source Git blobs: pilot.py cc8279c0688abf969b6aaa9f8026ad61c4137189; retrieval.py e9e7efa9169a9065d7de5e468f5f7922cf83bc6f. Local source and remote blob hashes match. Raw receipts archived in the local experimental bundle, not committed here.
- receipt16.json SHA256 9e3ada02ec0265fd40853f727e2e203e6794ab5779021951c57de8ef4abaeef2
- receipt32.json SHA256 2323e11ceebc2fb96f2443ca67a301a99b5810067ff913764f9b38612b71fcae
- retrieval_receipt.json SHA256 8ec7d907bbd7f025253eb72a5044239a774ed340340de38511c36a48beb98ac1
- audit.json SHA256 b6f4ba6b8d96176b7aef42cf2051ab0a5bf0921f4778fd091a63cc7214d0cbfc

Next: prospective independently generated verifier families, nonlinear local retrieval and native BPD source-locked component intervention. Compare equally supervised sparse and dense architectures. The 05 factorized scalar projection is not the rank theorem's address-by-input matrix.
