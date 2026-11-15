# PEF-11 | Precision/noise robustness of the recovered PC signal

This standalone frozen-checkpoint follow-up measures numerical robustness of the PEF-10 ablation-angle recovery frontier. [The registered protocol](PROTOCOL.md) was committed before PEF-11 heldout scoring; PEF-10's outcomes were already known.

Six PEF-07 checkpoints, 120 frozen PEF-10 angle conditions, and their *existing clean-probe ridge readouts* are loaded and checked by SHA256. At visit 8, PEF-11 applies either Gaussian noise along the retained signal direction, an equal-norm off-plane control, one-dimensional midtread quantization of the retained direction, or coordinate-wise quantization. It separately scores the unablated natural-state control. All edits are ungated across the 16 focal masks, with three deterministic noise replicas and no new readout fitting.

The [local results](LOCAL_RESULTS.md) report the frozen PEF-10 recovery collapsing under 1%-PC-scale transverse noise at 0.25 degrees (538.7, 583.3, 541.0 correct /768 on three coalition seeds), while remaining at 768/768 at 4 degrees. The registered criterion passed in all 3 coalition seeds; threshold task controls showed similar fragility. Local 26/26 combined intervention tests pass. Hosted CI should be treated separately.

### Run tests (archive-free)

	sudo apt install python3
	python -m pip install pytest torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
	python -m pytest research/pef11/test_pef11.py -q

### Full reproduction

	DNNL_MAX_CPU_ISA=AVX2 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
	python -m research.pef11.pef11 --evidence /path/to/registered \
	  --selection-ledger /path/to/pef09_results/frozen_selection_ledger.json \
	  --angle-ledger /path/to/pef10_results/frozen_angle_and_ridge_ledger.json \
	  --previous-results /path/to/pef10_results/summary.json \
	  --out /tmp/pef11

Output: `frozen_precision_ledger.json`, `summary.json` with all condition-level error masks, and `frontier.csv`. The full run requires the original PEF-07 checkpoint evidence archive and PEF-09/10 frozen selection ledgers. Python 3.12 invariant CI does not attempt model replay without those historical artifacts.
