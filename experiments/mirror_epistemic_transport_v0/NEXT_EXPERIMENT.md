# MIRROR-02 proposal: real-model semantic transport

**Hypothesis.** Low-rank counterfactual belief transport plus two shared-weight refinement passes discovers verifiable inconsistent commitments with fewer target queries than a parameter-matched ordinary recurrent controller.

**Preconditions.** Freeze source commitments and candidate transformations. Have independent reviewers verify semantic equivalence, context scope, and active corrections. Preserve the exact target replies and transcript offsets.

**Arms.** MIRROR with two shared passes; one pass; two independent Transformer blocks matched by parameter count; matched GRU; no transport; no family-risk head; scripted paired questions; random; and a larger LLM question selector. Match target, question opportunities, token budgets, and training data.

**Curriculum.** Synthetic pretraining; target-response prediction on train-only dialogue; replay with independent verifier; optional online reinforcement only for substantiated contradictions or genuine useful self-repair. A reply's refusal, uncertainty, or sarcasm is not itself an inconsistency.

**Sealed holdouts.** New personas, subjects, target models, paraphrases, transformations, question order, and coherent scoped exceptions. Register the held-out targets before selecting hyperparameters.

**Primary metrics.** Witness precision, discovery at 4/8/12 questions, target tokens per valid witness, calibrated predictive likelihood, and paired bootstrap confidence intervals across targets. Stratify evidence by topic and transform.

**Adapter.** Add a distinct public TargetView with no oracle handles and a TargetAdapter.ask(question) returning model/version, prompt hash, decoding parameters, raw reply, exact response span, and relevant context provenance. Guard against target-state leaks at this interface. Optional VPD on measured target activations comes after behavioral replication.

**Research decision.** The paired heuristic is the immediate baseline to beat. Evidence of value requires more efficient valid witness discovery on unseen targets at matched budgets. The initial smoke experiment is an engineering check.
