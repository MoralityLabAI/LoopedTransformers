# KITE-01B: harder development games

Follow-up to the ceiling in KITE-01. This is an exploratory, exposed development split, not prospective confirmation.

Replay with `node run_hard.mjs > hard_replay.json`. Three families, 48 generated rectangular game roots, 960 rows across four arms and five depths. Padded games contain strictly dominated filler actions; clone-skew games contain exact payoff duplicates; heterogeneous games combine per-action bias with binary payoff spikes. Seeds and raw exact certificates are in hard_receipt.json. No generator metadata is provided to solver arms.

Full-prior fictitious play assigns one pseudocount per action; normalized-prior assigns total prior mass one and adds one action's worth of mass per update. This corrects a dimension-dependent confound. Normalized multiscale mixes normalized full-history weights with recent windows 4,16,64. Preprocess-normalized removes strictly pure-dominated actions and exact two-payoff clones iteratively, then runs normalized fictitious play on the reduced game. It pays extra preprocessing cost and is a structural control. All candidate profiles are lifted and reverified on the original game. At all checkpoints the lifted exact regrets are asserted equal to the reduced-game regrets. Validation queries are reported separately per checkpoint. Equal solver query counts do not mean equal memory, FLOPs, or wall time.

In addition to the 0.2 certificate endpoint, raw regret thresholds 0.1 and 0.05 are diagnostic only. Kite repair certifies 0.2 only. No convergence guarantee at either finer threshold.

Interpretation: report all four arms; compare fixed memory to the normalized prior, not just the full prior. Preprocessing makes padding/clones a structural ceiling rather than evidence of learned strategic reasoning. Further hard families should use irreducible sparse games, adversarially screened development instances and fresh held-out generator seeds. No trained model, sham, sealed readout or causal inference yet.
