# LBF-07: Causal typed-rule association preregistration

**Registered 2026-11-23, before LBF-07 model scoring.** Development-only experiment in LoopedTransformers PPT-B. LBF-06 established that readable-context order was not a convincing contributor. This experiment tests whether *correct public candidate-name to rule-premise/head associations* matter for Lean tactic search.

## Hypotheses and matched interventions

Three exact-byte/character-count-matched encodings use the existing hybrid parser and public Lean packet:

- **hybrid_rule_intact**: candidate-first native typed rule associations, unchanged header, identical inert (x-filled) readable-context suffix.
- **hybrid_rule_rebind**: same ordered names, header, suffix and multiset of complete premise/head payloads; cyclically permute rule payloads among candidate-named entries when two or more survive the budget, else among all emitted rule entries. These deliberately incorrect semantic associations are a controlled informational impairment. A SHA-256 of public packet and candidate menu determines the nonzero shift. No proof label is read.
- **hybrid_rule_order**: preserve each correct name-to-payload binding while cyclically rotating complete rule entries through the byte window, using the same public hash. This controls positional/order changes.

Choose complete rule entries for inclusion before any permutation. All variants retain the same selected clauses, total bytes and Unicode character histogram, and a 384-byte maximum. A registered no-op remains in scoring when fewer than two distinguishable payloads exist. The predecision audit must report the fraction with a changed candidate binding and the fraction with a moved intact clause. Fewer than 50% informative training frames labels the causal screen INSUFFICIENT_INPUT_INTERVENTION, while native scores remain published.

Let E(s,C)=((n_i,b_i)) be the included name/payload pairs and H the public goal/budget header. With public-only cyclic permutation pi and fixed inert suffix P_x:

\[
I(s,C)=H \Vert \prod_i (n_i:b_i;) \Vert P_x
\]

\[
R(s,C)=H \Vert \prod_i (n_i:b_{\pi(i)};) \Vert P_x
\]

\[
O(s,C)=H \Vert \prod_i (n_{\pi(i)}:b_{\pi(i)};) \Vert P_x.
\]

These model-facing R strings may be false descriptions of the original Lean world. All proof outcomes are still adjudicated by the unchanged native Lean world, never by the representation.

## Fixed study design

Freeze new G03 generated source suite seed **20261128**, 64 roots per split. Compile 48 train-only shortest Lean proofs with the Lean 4.19.0 native server and independent certifier. Score **32 distinct calibration roots** with the same eight-tactic budget; leave ordinary-test and OOD cohorts closed.

Train seed **107** and **109** once per view on the **same physical CPU worker**, 720 FP32 optimizer steps, one deterministic PyTorch thread, width 32, four heads, eight typed slots, 384-byte ByteEncoder, two recurrent training passes, AB middle order, LR 0.001, 57,475 unique parameters, same train-source root/frame order. Duplicate seed-107 intact fit must have exactly equal named weights and root/optimizer trace. Source, model and compiler hashes are retained. No external model or GPU spend.

Use native Lean tactic execution for every arm with identical candidate menu and check budget. All counted full proofs require actual Lean kernel closure, separate standalone compilation and independent model-free replay. Train/calibration root-lineage disjointness and source-manifest match are hard gates. No calibration, ordinary-test or OOD outcome may enter optimizer updates.

## Outcomes

Primary development contrast: intact minus rebind mean *certified root completion* averaged over seeds inside each of 32 unique roots, threshold **+0.05**. Register also intact minus order and the double difference (intact minus rebind) minus (intact minus order), with threshold **+0.05** for a specific association effect above the positional control. Report 2000 descriptive root-bootstrap draws seeded 20261123, wins/losses/ties, root-level receipts and all negative outcomes. The independent denominator is 32 theorem lineages; two seeds reuse those roots.

A promotion requires a subsequent independently powered cohort, a 40% trained-order verified close rate, >=0.10 improvement over an exact candidate-uniform native baseline, prospective matched sham and EvalMath FSI-01 calibration. Current pilot does not calculate that native uniform baseline, so scientific eligibility remains false and competence disposition remains blocked regardless of development effect. Never open locked ordinary or OOD tests on this run.

A positive edge effect with small order effect suggests learned reliance on correct candidate-name/premise relationships. Similar edge and order effects indicate positional serialization. An absent, adverse or ambiguous result directs next work to source-verified on-policy repair training. Neither visualization brightness nor frozen-coda message movement is itself a proof certificate.
