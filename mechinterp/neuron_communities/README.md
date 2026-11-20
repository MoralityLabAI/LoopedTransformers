# NEC-01: Goodfire-inspired recurrent native-neuron communities

**Status:** exploratory instrument, frozen PEF-07 checkpoint adapter, CPU-first. The original PEF-07 decisions and thresholds are unchanged. All new measurements are **post-PEF-07**, separate from its preregistration.

The protocol builds on [Goodfire's neuron community discovery](https://www.goodfire.com/research/neuron-communities) and its [MIT-licensed implementation](https://github.com/goodfire-ai/neuron-communities). Goodfire's method uses attribution, a co-attribution graph, Leiden clustering, and held-out necessity/sufficiency tests. NEC-01 uses a **seeded Louvain community detector** in a small, self-contained implementation. It is therefore a method-inspired adaptation, *not* a replication of Goodfire's full pipeline, Leiden partition, 10-million-prompt corpus, or Neuron Atlas.

## Model and unit of analysis

The actual adapter imports `research.pef07.pef07.Looper` from the registered **12-bit** PEF-07 branch. Its tied 32-dimensional block has **64 native GELU MLP neurons** repeatedly used during 12 bit visits. We define **768 computational occurrences** as (visit 0–11, neuron 0–63), while acknowledging that each neuron shares its weights across visits.

Only the **memory token** persists from one visit to the next. Forward-pre-hooks at `block.mlp_down` capture the GELU input. The discovery objective is the native model's own predicted two-class logit margin, with absolute activation × gradient attribution. Attribution does not use task labels.

There are 256 distinct eight-bit contexts. Each context contains all 16 possible assignments of the four oldest bits, whose teachers are either `(a AND b) XOR (c AND d)` or `1[a+b+c+d >= 2]`. The original SHA256 partition assigns 160 contexts to trunk training, 48 to downstream readout fitting, and 48 to heldout scoring. NEC-01 aggregates top-k attributed neurons **within each complete 16-state context**, then treats one context as one independent graph observation. Context identity never crosses splits.

## Discovery

1. Verify the frozen six-checkpoint PEF-07 training ledger, original source partition digest, chosen task and seed, checkpoint bytes, and model-weight digest. The adapter refuses a missing or mismatched ledger. No training is performed.
2. On training contexts only, identify the top two native GELU neuron occurrences **per visit** using mean absolute activation × gradient attribution across the 16 focal states.
3. Record context-level frequency and co-attribution counts, remove hubs with prevalence >=90%, and retain positive edges with Pearson phi >=0.18 and at least 3 independent co-occurring contexts.
4. Partition the graph using seeded NetworkX Louvain. Require 2–24 nodes and at least two visits. Rank candidates using only their discovery graph edges and freeze up to eight communities. Build a sham with the same number of neurons **at each visit** and the closest discovery frequency. It is frequency-matched, not activation-norm-matched.

The complete graph, candidate roster, shams, source SHA and model SHA are written in a canonical JSON file plus a SHA256 sidecar **before** evaluation. Manual changes invalidate evaluation unless a new discovery is explicitly frozen.

## Held-out causal intervention

The 48 heldout contexts are evaluated in their entirety. For each selected community, replay the frozen network on each complete context with a causal hook on the **memory token's GELU activations**, at precisely the visits containing community members. All other visits execute normally.

| Condition | Intervention within community-bearing visits |
|---|---|
| Clean | Ordinary network |
| Ablate | Zero community neuron occurrences |
| Sham | Zero a same-visit, discovery-frequency-matched set |
| Retain | Zero other GELU neurons; keep community and always-on hubs |
| Hubs | Keep hubs only |
| Blank | Zero all GELU neurons |

The model's attention, residual path, embeddings, head, and other visits remain intact. The `retain` intervention therefore tests **sufficiency inside a set of recurrent MLP occurrences**. It cannot demonstrate that the community performs the task by itself.

For each context, average KL(clean || intervention), native-head correctness change, and clean-top-class probability change over all sixteen focal states. Bootstrap the **48 context means**, keeping paired conditions within the same context. The reports give 95% percentile intervals for necessity excess KL (ablate minus sham), sufficiency KL gain (hubs minus retain), and blanket ablation. The unadjusted descriptive intervals do not establish statistical power or family-wise discovery significance.

The original PEF-07 task scores are retained as background, and the 12-bit task does not serve as an independent validation of new Boolean compositions. PEF-07's threshold teacher provides an appropriate matched learned-function control, though current NEC-01 reports one seed/task at a time. To claim learned neuron communities as a general mechanism, repeat across both tasks and seeds and preregister a full family-wise inference plan, meaningful effect threshold, matched shams, and independent trained checkpoints. Follow the repository's `AGENTS.md` and EvalMath FSI-01 evidence gate.

## CPU commands

Install `torch==2.10.0+cpu`, `pytest`, and `networkx==3.4.2`. With a local copy of the **original six PEF-07 checkpoints and `sealed_training_ledger.json`**, run:

```bash
python -m mechinterp.neuron_communities.pef07_adapter discover \
  --checkpoint-dir /path/to/pef07_sealed --out /tmp/nec01_seed103_coalition \
  --seed 103 --task coalition

python -m mechinterp.neuron_communities.pef07_adapter evaluate \
  --checkpoint-dir /path/to/pef07_sealed --out /tmp/nec01_seed103_coalition \
  --seed 103 --task coalition
```

For an explicitly random-initialized, CPU-only test:

```bash
python -m pytest -q mechinterp/neuron_communities/test_nec01.py
python -m mechinterp.neuron_communities.pef07_adapter discover --smoke \
  --out /tmp/nec01_smoke --train-contexts 4 --min-joint 2
python -m mechinterp.neuron_communities.pef07_adapter evaluate --smoke \
  --out /tmp/nec01_smoke --heldout-contexts 2
```

A smoke trace is evidence that the hooks, replay conditions, hashes and analysis code execute. It is **not** a learned circuit discovery. The CI workflow intentionally runs only the smoke and unit tests, and uploads receipts; it does not download unavailable local checkpoints, retrain the six registered models, or consume GPU capacity.

## Proposed follow-ups

* NEC-02 preregistration with minimum meaningful paired correctness effect, robust context-level inference, candidate-family multiplicity gate, independent trained seeds and predeclared stopping rules.
* A two-basis comparison with the existing LS-SAE source-locked physical dictionary at exactly matched intervention locations and component budgets. Native neurons and SAE features require a common local hidden-state intervention contract; the SAE stage is **not yet implemented** here.
* Recurrence transport tests distinguishing one neuron's different visit-specific roles from circuits preserved across depth, including PEF-08+ frozen causal baselines if their checkpoints are available.
* A separate adapter-training drift protocol using checkpoint-aligned weight data and appropriate controls before drawing any self-model interpretation.
