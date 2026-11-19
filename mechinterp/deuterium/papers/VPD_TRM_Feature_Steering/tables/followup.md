**Execution status: local CPU: NO_REGISTERED_ADVANTAGE.**

The frozen synthetic recurrent-gate protocol was executed on local CPU. The comparison completion flag is True. Registration commit: `f1e3e9ba663ea906a6468d76426bb7381a6e23ad`. Protocol SHA-256: `c6bd49b7423091f02fd55812a51a24aae8f4f07332b3178e006a64a015f908a9`. GitHub Actions reproduction remains blocked by repository write access; the local freeze was recorded before held-out scoring.

| Prerequisite or pilot | Verdict | Observed evidence |
| --- | --- | --- |
| Recovered checkpoint task/replay gate | BLOCKED_ARCHITECTURE_PROVENANCE | 11,191,328 parameters; 91 frozen tensors checked; exact equality=True; task performance evaluated=False. |
| Recovered residual: bucket_head | MATRIX AUDIT | 16 learned components; residual/weight Frobenius norm=0.203684. |
| Recovered residual: networks.0.layers.0.1.to_out | MATRIX AUDIT | 32 learned components; residual/weight Frobenius norm=0.783704. |
| Recovered residual: networks.0.layers.1.1.ff.2 | MATRIX AUDIT | 32 learned components; residual/weight Frobenius norm=0.798766. |
| Auxiliary pilot seed 997 | PILOT_CALIBRATION_ONLY | Clean calibration accuracy=0.966797; runtime=4.48 s; held-out evaluated=False. |
| Pilot replay diagnostics | CALIBRATION ONLY | Components-only KL=0.000003; argmax agreement=1.000000; residual-restored maximum logit error=0. |
| Retained pilot failure, seed 997 | ENGINEERING_FAILURE | The initial pilot failed Config validation: slow_eval_freq must be a multiple of eval_freq. The failed receipt is retained. |

### 6.3 Local held-out comparison

| Seed | Clean test accuracy | Veto rows | Random minus VPD NLL | 95% paired-row interval | Runtime (s) |
| --- | --- | --- | --- | --- | --- |
| 103 | 0.9771 | 641 | 0.001378 | [0.000978, 0.001800] | 4.17 |
| 211 | 0.9712 | 627 | 0.000629 | [0.000303, 0.000981] | 3.13 |
| 307 | 0.9673 | 669 | 0.000664 | [0.000361, 0.001000] | 3.43 |

The mean primary effect across seeds is 0.000890402, with a 95% t interval [-0.000159859, 0.001940663]. The recorded verdict is `NO_REGISTERED_ADVANTAGE`. Every seed passes its prerequisites. Collateral gate passed: True. The across-seed interval includes zero. Each within-seed interval is positive; those intervals quantify row variation conditional on one trained model. The across-seed interval treats the independently trained model/decomposition as the replication unit.

![Local synthetic follow-up effects and uncertainty](figures/followup_effects.png)

*Figure 3. Positive values favor learned steering over the selected random arm. Seed intervals use paired-row bootstrap; the bottom interval uses the mean of three independent seed effects and Student t with two degrees of freedom. Both estimate the same contrast at different levels of variation.*

| Seed | Arm | Target NLL | Target accuracy | Collateral delta (pp) | False commits / veto |
| --- | --- | --- | --- | --- | --- |
| 103 | clean | 0.066945 | 0.9704 | 0.0000 | 0.0094 |
| 103 | sham | 0.066945 | 0.9704 | 0.0000 | 0.0094 |
| 103 | vpd | 0.064486 | 0.9750 | 0.0711 | 0.0062 |
| 103 | random | 0.065864 | 0.9704 | 0.0000 | 0.0094 |
| 103 | task_gradient | 0.062824 | 0.9735 | 0.0000 | 0.0094 |
| 211 | clean | 0.033766 | 0.9856 | 0.0000 | 0.0080 |
| 211 | sham | 0.033766 | 0.9856 | 0.0000 | 0.0080 |
| 211 | vpd | 0.032650 | 0.9856 | -0.0704 | 0.0080 |
| 211 | random | 0.033279 | 0.9856 | -0.0704 | 0.0080 |
| 211 | task_gradient | 0.032200 | 0.9872 | -0.1408 | 0.0080 |
| 307 | clean | 0.024806 | 0.9940 | 0.0000 | 0.0030 |
| 307 | sham | 0.024806 | 0.9940 | 0.0000 | 0.0030 |
| 307 | vpd | 0.023592 | 0.9940 | -0.1450 | 0.0030 |
| 307 | random | 0.024256 | 0.9940 | 0.0000 | 0.0030 |
| 307 | task_gradient | 0.023246 | 0.9940 | -0.1450 | 0.0030 |

The task-gradient baseline has lower target NLL than learned VPD in 3 of 3 seeds. Clean and sham outputs agree. VPD collateral accuracy changes in percentage points are 103: +0.0711, 211: -0.0704, 307: -0.1450. Model parameter counts: 1,763. The measured seed runtimes include model training, decomposition, selection, and scoring; environment setup and archive recovery are outside that timing. The CPU was AMD EPYC 9V74 80-Core Processor with 2 PyTorch threads. The frozen source used Python 3.12.14 and PyTorch 2.5.1+cpu. Eleven package versions accompany the result receipt; a contemporaneous full transitive package freeze is absent. The Actions workflow will capture `pip freeze`.

Residual-restored replay has maximum absolute logit error 0.0. Components-only calibration KL ranges from 1.900e-06 to 2.024e-06, with minimum argmax agreement 1.0. The literal edits, checksummed per-example held-out scores, loss traces, and selected candidate receipts remain available for replay.

Recovered-model missing prerequisites: experiments.hermes_skill_gym.train_hrm_hermes.HermesHRMClassifier; experiments.hermes_skill_gym.hermes_dataset.HermesSkillDataset; independent original forward logits for replay validation; historical x-transformers/package versions. These gaps prevent a recovered-task steering comparison.

Positive NLL differences favor VPD. The primary interval uses three independent model/decomposition seeds and two degrees of freedom; within-seed row intervals describe a separate level of variation. Secondary outcomes retain descriptive status.

The new experiment concerns a generated commit/veto/repair distribution with output-head edits. Its evidence is separate from the recovered arena, ARC, Wordle, governance and adapter records.
