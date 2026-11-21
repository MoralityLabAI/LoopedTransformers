# NEC-03I-B: source-frozen hash-rank active-feature controls

**Development-only protocol, November 21, 2026.** This is a new causal sensitivity test on the eight already exposed NEC-03H-v2 worlds dev-008 through dev-015. It preserves the original model weights, SAE dictionaries, source-rank policies, original ITT effects, and all 64 committed confirmation-world specifications.

## Question

NEC-03I established that the source-ranked active TopK16 donor beat the lower-ranked active TopK16 donor by +22 correct predictions in dev-008 and +26 in dev-009, while losing 25 in dev-013 under the same direct physical-dose constraint. The negative dev-013 root retained 46.65% of original left-donor dose and had 100% positive shared-dose coverage. Equal local L2 norm permits different donor directions and recurrent transport, so more than one alternative active-code control is useful.

## Frozen sham family

The model uses a 64-dimensional memory-token native GELU vector, a previously source-trained 128-feature TopK16 encoder/decoder, and a frozen per-visit feature ranking. At each selected recurrent visit, choose n source-highest **currently active** features, where n equals the original native community cardinality at that visit (1 through 8).

For each of four fixed variants (IDs 0, 1, 2, 3), construct a SHA256-ranked permutation of the 128 feature IDs. The SHA input is the literal salt NEC03IB-hash-active-shams-v1, the original world trunk seed encoded in eight big-endian bytes, the recurrent visit, the variant ID, and the feature ID. The control chooses n hash-highest **currently active** features after excluding all n original source-selected active features. Thus the source and control donors have exactly matched per-visit component count, both are active on the current input, and they share no selected feature IDs. The hash ranking uses no evaluation labels or correctness outcomes.

Each pair is added into the same hubs-only GELU recurrence at the same native source-selected visits. The direct frozen MLP-down output L2 norm of each donor is computed for that example and visit, and both are attenuated to their smaller available direct dose. Gains remain in [0,1]. A numerical dose pair needs >=99% of site/example pairs within 1% of the shared norm. A useful physical signal needs >=80% positive-dose sites and median >=25% original left-donor dose retained. Failing comparisons remain visible and their effects retain an exploratory label.

## Integrity controls

The input is the original exact eight-model NEC-03H development evidence from GitHub Actions run 44878411694. The code checks the registered world source assignments, the 900-step native-model file SHA256 and model-weight digest, frozen source-selected native sites, TopK16 source-trained checkpoint SHA256, source-only rank manifest, and the original V2 protocol lock.

Before any new hash-control evaluation, independently reproduce all 48 original clean/hubs/source-ranked/bottom-ranked-active logit arrays from the archived native model, SAE and source context group. The test must match the archived logits within a small floating tolerance; a mismatch aborts the experiment.

For each original world and each hash-rank variant, save 48 complete 16-state context records with truth labels, all clean/hubs/source-ranked/hash-control logits, all 16 active feature identities, selected top and disjoint control identities, and per-site physical dose/gain arrays. The total panel is 8 original roots × 4 hash variants × 48 context groups = 1,536 correlated candidate-context records.

An independent pure-stdlib auditor will recompute the hash permutations, source feature ranks, all sixteen teacher labels, signed correctness changes, realized local doses, physical matching and retained-signal gates, and check all original source commitments. A failed hash, wrong feature selection or source-data mismatch rejects the artifact.

## Evidential classification

This is **posthoc development** on previously opened Boolean worlds. The four SHA-rank variants are correlated controls, and their contexts share the same trained model and teacher. They are never independent task-world roots. The earlier NEC-03H intention-to-treat mean +0.423 percentage points remains unchanged, below the +2pp scientific floor, and its sham-null population remains insufficient for EvalMath's 95/95 upper tolerance bound.

The experiment can test whether the learned source-ranked active-feature policy behaves better than multiple same-budget, same-dose, disjoint alternative active-feature donors. It cannot establish unique circuit identity or transfer to a larger language model. All 64 confirmation worlds and all reserved within-world context splits stay sealed.

The exact four-variant family, source code blob digests, fixed source and confirmation SHA commitments, and physical admission policy are recorded in NEC03IB_SHAM_LOCK.json **before the new donor-replay workflow starts**. FSI-01 remains REDESIGN_OR_PILOT_ONLY.
