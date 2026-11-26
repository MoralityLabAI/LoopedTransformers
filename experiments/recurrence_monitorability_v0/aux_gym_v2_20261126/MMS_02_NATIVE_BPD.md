# AUX-V2-MMS-02: native behavioral factor initialization

The native BPD source is pinned to MoralityLabAI/BPD@722613a80d80af0e80bb9f5ac1ce83eef111e4f7. The adapter installs BPDModel on the original dense tied MLP down projection, calls the actual BPDTrainer with pass-conditional neutral ablations, pass-shared PGD, stochastic trajectory reconstruction and minimality. The teacher weights are frozen. Teacher logits at each visit are exposed through the existing lens; fitting reads train tokens only, not gold labels. Closed-loop continuation is the sole supported regime. Validation is a small separate development reconstruction sample.

The warm-start memory arm verifies the initial teacher checkpoint hash, physical weight hash, dataset manifest and factor file hash. Learned V/U factors are ranked by their gauge-invariant outer-product norm; their selected output span is orthonormalized and frozen for replay. This is a native BPD-initialized output-span auxiliary loss, not native BPD component-ablation training of the task model. The physical intervention evaluator from MMS-01 still uses SVD; native component-specific interventions and tolerance-normalized causal tests remain future work.

For a fair warm-start test, every arm uses the same teacher per root. Set --freeze-basis for SVD and random controls. Include off and batch, and compare SVD/memory, random_basis, shuffled_memory and native_memory at 4/16/64 MiB. Fit BPD on teacher behavior before any task updates. Report factor relative reconstruction residual, teacher parity, teacher reconstruction KL, pass tolerances, active component counts and matched-rank warmstart results. A poor factor fit is an explicit failure mode, not evidence about native BPD's value. No locked test is authorized by a development win.

    git clone https://github.com/MoralityLabAI/BPD.git /tmp/bpd-native
    git -C /tmp/bpd-native checkout 722613a80d80af0e80bb9f5ac1ce83eef111e4f7
    python -m research_gym.interp.aux_gym_v2_native_bpd --bpd-checkout /tmp/bpd-native --data /tmp/aux-v2 --run /tmp/teacher --out /tmp/native-factors --seed 1 --components 32 --steps 100
    python -m research_gym.interp.aux_gym_v2_memory train --data /tmp/aux-v2 --init-run /tmp/teacher --factor-dir /tmp/native-factors --family modular_recurrence --out /tmp/native-mms --mode native_memory --mib 4 --rank 16 --seed 1 --steps 1000
    python -m research_gym.interp.aux_gym_v2_memory train --data /tmp/aux-v2 --init-run /tmp/teacher --family modular_recurrence --out /tmp/svd-mms --mode memory --mib 4 --rank 16 --seed 1 --steps 1000 --freeze-basis

The engineering CI uses a separate small seed, 8 teacher updates and two BPD updates, with tolerance calibration disabled only for that smoke. It cannot establish successful factor recovery or task improvements. The native source must be accessible and CI must pass before research execution.
