# Gated workspace reinforcement-learning pilot

The hybrid retains Plan A's single tied Transformer cell and eight typed proof slots. From Plan B it takes learned cross-pass communication. Each pass writes an elementwise-gated update to a typed bank. The next pass retrieves from that bank with attention and gates the retrieved message into its workspace. The control uses the repository's TRM post-norm blocks, two latent updates and one answer update per recursion, with the same observation encoder, action head and value head.

The actor samples tactics from the current checkpoint during native Lean search. A separately compiled axiom-free theorem produces terminal reward. Every tactic check incurs a small cost. The critic estimates returns and the actor receives gradients through the log probabilities of the sampled actions. The training command has no expert tactic labels or reference proof input.

## Development registration

Compare the hybrid at four passes and TRM at one recursion with the same parameter-width budget, eight Lean tactic checks, six process-tree CPU seconds per proof, and 1.2 billion estimated MACs per proof. Record actual MACs and native certificates. Seed 24011 freezes new LoopedTransformers Horn source roots and the original BPD-LG-02 `exact` public declarations. BPD structural groups stay together. One public binder is alpha-renamed for the shared native tactic menu. Source-side feasibility screens select development proofs that fit the eight-check cap. Training retains BPD positive and negative claims and obtains all outcomes through Lean. The pilot focuses on feasible search and decoy selection.

The primary future effect is a five percentage-point paired improvement in certified proof completion against equally trained TRM. The development pilot reports both initial and final policies and uses single-pass bank removal and a channel-permutation sham. The sham preserves the message bank's aggregate norm and changes the channels read by attention. Measurement of a causal effect also requires a separate development resolution study and sealed confirmation under EvalMath's FSI-01.

The optional `--mathcan-archive` arm interleaves recorded original-source experiences from the immutable RAM-08 training archive. It checks the independent Lean 4.35 audit and filters development and protected test modules. It joins a successor only when both the exact script prefix and the printed observation match. Unsupported continuations are censored for learning. Recorded closures may supply reward in this environment and do not count as new certified evaluation proofs. The hybrid, TRM, and frozen-backbone TRM can share the same archive schedule. The imitation control uses only the independently certified surviving paths of the live Lean 4.19 rollouts.

From this directory after installing the PPT-G04 CPU dependencies and building `lgym`:

```powershell
New-Item -ItemType Directory runs/ppt-hybrid -Force
python -m ppt_gym.hybrid_rl freeze runs/ppt-hybrid/cohort.json
python -m ppt_gym.hybrid_rl run runs/ppt-hybrid/cohort.json runs/ppt-hybrid/hybrid.sqlite --architecture hybrid --interventions --report runs/ppt-hybrid/hybrid.json
python -m ppt_gym.hybrid_rl run runs/ppt-hybrid/cohort.json runs/ppt-hybrid/trm.sqlite --architecture trm --report runs/ppt-hybrid/trm.json
python -m ppt_gym.hybrid_rl run runs/ppt-hybrid/cohort.json runs/ppt-hybrid/trm-calibration.sqlite --architecture trm --head-only
python -m ppt_gym.hybrid_rl run runs/ppt-hybrid/cohort.json runs/ppt-hybrid/hybrid-imitation.sqlite --architecture hybrid --objective imitation
```

For mixed-source trials add `--mathcan-archive PATH` to the three RL arms. Use the [original RAM-08 training artifact](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43689874169) with SHA-256 `a2ac363547858871c4e066c96bac67253dfeb7ff4e2a0050bb5886c8ca9edcfc`. Pair results with `python -m configs.analyze_ppt_hybrid_rl COHORT OUTPUT hybrid=HYBRID_SESSION trm=TRM_SESSION calibration=CALIBRATION_SESSION`.

The exact pilot database carries per-episode native verifier evidence and atomic model, optimizer and sampler state. `--steps` pauses at a ledger boundary. Repeat the same command without `--steps` to continue. Development scoring includes a separate native transcript replay. The original G04 pilot cohort, old Lean-CGRT archives, MathCan protected test modules and BPD test statements remain outside this trial.
