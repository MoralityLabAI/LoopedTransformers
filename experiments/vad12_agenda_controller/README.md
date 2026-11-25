# VAD-12: research-agenda controller gate (CPU)

Downstream training for EvalMath VAD-10/12. EvalMath owns the instance definitions, exact oracle and the
evaluation (`scripts/vad_agenda*.py`, receipt `output/verified_abstraction/vad_12_gate.json` in
MoralityLabAI/EvalMath, branch `research/vad01-verified-abstraction-20261125`). This directory holds the
data generator, the controllers and the training/evaluation driver.

    set EVALMATH_VAD=<path to EvalMath checkout>
    python gen_data.py --train 1000 --calib 250          # ~1 min; writes data/*.npz (not committed; hashes in data/manifest.json)
    python -u train_eval.py --epochs 25 --seeds 2 --fresh 60 --indist 100 --out results.json   # ~1 h on a loaded CPU

Arms (parameter matched to ~12.2k): weight-tied slot refinement with episode GRU (`trm`), memoryless, fixed depth (K=1),
flat MLP (`ff`), oracle one-step-gain feature, and a shuffled-memory ablation of `trm`. Test families are generated
only after training; see `results.json` for exact-evaluation regret against the optimum and the greedy-0/1/2 and
direct baselines.

Result: gate FAIL. No controller reaches greedy-2 regret on any fresh family; memory and recurrence add nothing
measurable. See the EvalMath note `paper/economics_of_mathematical_discovery.md` section 3.3.
