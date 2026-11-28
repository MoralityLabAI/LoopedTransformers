# LBF MMS development lanes

LBF-19 rec_prior_hs now has a companion runner with length-scoped banks for
candidate and gate writes, sharing one global page cap. LBF-20D Bellman
regret and LBF-20F CSP support ranking reuse their original training rows,
models, exact labels and evaluators. One selected shared recurrent visit
receives feedback during the ordinary backward pass. The existing registered
sources remain untouched, and old public development scores are exposed.
The default SVD map is a proxy; causal BPD fitting remains future work.

From repo root (PowerShell: set PYTHONPATH=. then use the same Python path):
```sh
PYTHONPATH=. python research/lite_braille_fresco/lbf_mms.py --gym 20f --mms-mib 16 --mms-mode bpd --threads 2 --updates 480 --out runs/lbf20f_mms
PYTHONPATH=. python research/lite_braille_fresco/lbf_mms.py --gym 20d --mms-mib 4 --mms-mode observe --out runs/lbf20d_observe
PYTHONPATH=. python research/lite_braille_fresco/lbf_mms.py --gym 19 --mms-mib 64 --mms-mode bpd --out runs/lbf19_mms
```

LBF-20M receives a separate exact RAM study. Its original Unit.query,
Factorized.answer, and witness checking remain authoritative. A shared LRU
memo bounds charged Python query-entry bytes, while immutable base tables
and global lift/project caches are excluded. Charges use recursive
sys.getsizeof plus a 128-byte entry allowance; measure RSS externally.
Zero budget disables query retention. Repeated public queries are a
deliberate stress workload, not a new generalization claim.
```sh
PYTHONPATH=. python research/lite_braille_fresco/lbf20m_ram.py --mms-mib 1 --repeat 4 --out runs/lbf20m_ram
```

LBF-18/19 fast controller has checkpoint-keyed word/pair caches and dense
frontier grids; future work should cap those caches separately and retain
exact model/version invalidation. LBF-20H's scheduler is NumPy, not a
backprop recurrent trainer. LBF-20I/L/M are exact certificate/search systems.
The experimental unit for learned tests is independently generated root
and fit seed. Use fixed exposure and fixed CPU studies separately, plus
observe/dense/sham controls, fresh confirmation lineage and EvalMath
preflight before promotion. No sealed datasets are opened. No large-memory
gain has been measured. The smaller datasets may saturate well below GiB.

All learned LBF adapters now accept --mms-map-manifest; keys block (20D/F), candidate and gate (19). V/U exports must match their physical layer. Hash verification establishes identity only; SVD fallback and exported-map admission remain explicitly labelled.
