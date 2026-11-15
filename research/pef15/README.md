# PEF-15 | Fresh-seed context-level conformal intervals

PEF-14 correctly forecast heldout quantization accuracy on 2/3 independent coalition models but its nominal 90% normal intervals covered only 20/48 coalition condition means. PEF-15 tests a **different uncertainty target**: the accuracy of one complete sixteen-mask context family. The [standalone registered protocol](PROTOCOL.md) was committed at `20e72699aa9717bc7de3bb395302d48f9e5a0043` before new training. The [mathematical note](MATH.md) proves a conditional rank-based 16/17 marginal coverage guarantee and derives a separate conservative Serfling bound for the 48-context mean.

New model seeds: 251, 263, 277, separately for coalition and additive-threshold tasks. The independently SHA256-salted split allocates 160 trunk-training contexts, 32 fit-only contexts, 16 labeled calibration contexts and 48 unseen evaluation contexts. All sixteen focal-bit assignments remain in the same context group. Both arms use the unchanged PEF-07 32-channel tied recurrent architecture with precisely 450 AdamW optimizer updates.

The 32 fit contexts train ablation-angle ridge decoders and a fixed conditional phase-mixture point forecast. A fresh 16-context calibration set measures the actual per-context quantizer accuracy and chooses the **16th of 16** absolute residuals for a 90%-labeled interval. Nothing is re-fitted or selected using the calibration labels. Only after all six frozen checkpoints, probe forecasts and calibration intervals are sealed does the runner evaluate 48 heldout contexts. The paired threshold task is reported alongside coalition.

[Local results](LOCAL_RESULTS.md): the registered empirical 90% context coverage and <=0.85 mean-width criterion passes for all three fresh coalition seeds (3/3), with observed coverage 93.88%, 99.09%, and 99.22%. Mean intervals occupy 0.218, 0.239, and 0.280 of the [0,1] accuracy range. All 96 conservative aggregate Serfling intervals cover the 48-context heldout mean, though they are much wider. The independent audit recomputes all 96 phase forecasts and group calibration intervals from sealed inputs. The full PEF-07..PEF-15 local regression suite passed 65/65 tests.

From the Loop​edTransformers repository root with CPU PyTorch 2.10 and pytest:

```sh
python -m pytest -q research/pef15/test_pef15.py
DNNL_MAX_CPU_ISA=AVX2 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTHONHASHSEED=0 python -m research.pef15.pef15 --out /tmp/pef15
python -m research.pef15.audit_pef15 --out /tmp/pef15
```

Full training should use a fresh output folder. On an interrupted run, PEF-15 may reuse only already completed original PEF-14 trainer checkpoints with matching seed/task/step/weight hashes. Fit and calibration ledgers are content-addressed through their SHA256 parent receipts. The optional hosted six-model run must be evaluated independently from local Python 3.13 results. All claims concern the known synthetic Boolean-history functions, not unseen operators, Lean theorem-proving or pretrained language models.
