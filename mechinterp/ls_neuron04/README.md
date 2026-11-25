# LS-NEURON-04: progressive pruning and length curricula

**Development only. FSI-01 REDESIGN_OR_PILOT_ONLY.** No paid compute, native NEC04 or Sudoku evaluation. Stacked on [LS-NEURON-03 #364](https://github.com/MoralityLabAI/LoopedTransformers/pull/364). All students use synthetic affine parity recurrence; width four, 513 parameters at both saved endpoints.

## Methods
[Protocol](protocol.md) frozen locally before training; [extension](extension.md) records adaptive all-arm continuation before its execution. Three seed groups, eight training nuisance-prefix groups, four calibration groups, four held-out development groups per seed. New random focal/suffix histories, 512 per development group and horizon (2048 rows/checkpoint/horizon); repeated histories and horizons not independent. Parent hashes verified. Student cost is forward linear MAC proxy (skip 16*19, read 16, MLP 35*width), not total FLOPs or latency; teacher inference and ranking costs separate, verification overhead excluded. Equal-update controls share source minibatch stream. AdamW reset at four pruning milestones for every arm. All arms receive 2400 further updates, new matched stream and optimizer reset in adaptive extension.

## Initial endpoints
Mean final parity accuracy (%), T=visits, Q8/Q4 symmetric per-tensor weight fake quantization at T12 (floating-point activations).

| Arm | T4 | T6 | T8 | T10 | T12 | Q8 T12 | Q4 T12 | Student MACs | Teacher MACs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed4 |84.81|65.84|54.75|52.03|50.76|50.70|50.15|282624000|0|
| curriculum |80.97|74.19|71.14|70.52|69.04|68.62|57.37|452198400|0|
| long8 |79.88|69.97|63.05|58.02|54.22|54.35|52.78|565248000|0|
| kd_long8 |81.20|71.32|64.88|60.90|57.10|56.51|52.38|565248000|3145728000|
| prog_salience |95.74|91.44|85.14|75.20|66.81|63.00|52.59|675840000|0|
| prog_random |86.23|79.74|74.79|73.55|68.07|68.83|54.62|675840000|0|
| budget_curriculum |94.94|89.91|84.62|80.66|77.20|77.86|55.39|675942400|0|

## Extended endpoints

| Arm | T4 | T6 | T8 | T10 | T12 | Q8 T12 | Q4 T12 | Student MACs | Teacher MACs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed4 |100.00|88.98|71.50|58.97|54.96|54.95|51.14|847872000|0|
| curriculum |98.65|96.79|94.65|93.59|91.18|90.92|49.27|1582694400|0|
| long8 |99.64|98.42|96.32|93.20|88.43|88.25|51.77|1695744000|0|
| kd_long8 |96.53|92.45|88.43|83.02|73.83|74.30|50.18|1695744000|9437184000|
| prog_salience |96.81|94.38|92.14|89.83|84.65|84.33|49.20|1806336000|0|
| prog_random |96.35|90.71|85.61|84.49|80.71|80.96|51.76|1806336000|0|
| budget_curriculum |99.09|97.62|95.90|94.48|92.30|92.04|52.73|1806438400|0|

| Arm | Base T12 seeds 11/23/37 | Extended T12 seeds 11/23/37 | Extended T4 MLP removed |
|---|---|---|---:|
| fixed4 |51.95/52.34/48.00|62.11/51.22/51.56|51.92|
| curriculum |57.37/50.15/99.61|78.91/94.63/100.00|56.38|
| long8 |60.94/49.61/52.10|83.74/95.36/86.18|52.15|
| kd_long8 |64.31/49.90/57.08|54.25/91.26/75.98|51.09|
| prog_salience |58.30/63.87/78.27|72.17/96.97/84.81|49.41|
| prog_random |68.70/71.14/64.36|85.30/80.71/76.12|50.29|
| budget_curriculum |72.27/59.33/100.00|80.71/96.19/100.00|55.78|

## Interpretation

Base salience minus random at T12 = -1.25pp; seed differences -10.40/-7.28/+13.92pp. Extended +3.94pp; seed differences -13.13/+16.26/+8.69pp. Both fail the locally specified +5pp floor; base random also fails the 95% T4 competency gate. No calibrated physical sham floor, independent-world power or untouched confirmation. Progressive salience beats random at T8 but loses to scratch curriculum with approximately matched student forward MAC budget at T12. Extended narrow curriculum vs fixed4 +36.22pp T12, but training visits differ (1.583G vs 0.848G MAC proxy); compare long8 when considering curriculum order. Extended long8 beats KD-long8 by 14.60pp T12 despite 9.437G teacher forward MACs; teacher supervision only covers T<=8 and causal reason for weaker transfer not established. Teacher T4/T6/T8/T10/T12 = 100/99.92/99.06/83.04/58.07%. Four-bit T12 arms near chance under this quantization recipe. No actual low-bit kernel or transformer-scale claim.

## Reproduction and audit

PyTorch 2.10.0+cpu, deterministic single-thread workers, four concurrent workers. 42 student checkpoint hashes, 435 evaluation replays (420 student plus 15 teacher), independent scalar XOR and nuisance-prefix ownership checks, matching sampler receipts, 24 logged compactions, maximum compact-vs-masked logit error 3.21865082e-6. Audit checks recorded compaction errors/indices; does not independently retrain or replay intermediate pre-cut checkpoints. Three independent seed groups; no significance or equivalence claim. All seven arms retained at both endpoints. Full local archive contains raw correctness vectors, 42 student weights, three teacher weights, source and SHA-256 manifest. Archive SHA-256 7f5641001c6a99e213a8be34c014e5d2d0e72484d9af082d5301164d4de4985b.

```sh
for seed in 11 23 37; do
  for arm in fixed4 curriculum long8 kd_long8 prog_salience prog_random budget_curriculum; do
    python train.py --seed "$seed" --arm "$arm" --out results
    python extend.py --seed "$seed" --arm "$arm" --out extended
  done
done
python audit.py
```

Teacher weights and manifest are supplied in the companion local archive at parents/affine_long and parents/manifest.json. Source Git blobs verified byte-identical to executed local source: train.py 0d141c26f373bb8db7b68eb14d1b1647d0b25903, extend.py e9540e3894bca8136d8a4a7c1118551a443dfa14, audit.py fefd2b6e34b9e56e161a2f5a00cd7821a3fbc013.

Next: test the learned two-state transition under hidden-state intervention; report recurrence Jacobian, finite perturbation radii and quantization-aware training at fixed budgets. Native MLP-dependent models require their separate headroom and sealed-family protocol.
