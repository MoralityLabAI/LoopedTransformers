# MIRROR-03: from synthetic belief transport to actual language-model dialogue

**Milestone:** runnable bridge with frozen, reviewer-attributed semantic contracts and complete raw-reply receipts. The local example uses a scripted target and a loopback mock HTTP server; it is **not** evidence of performance against a large language model.

The experiment lives in the `LoopedTransformers/experiments/mirror_epistemic_transport_v0/mirror03/` subfolder, next to the MIRROR-01 and MIRROR-02 experiments.

## Architecture and claim

The original question is whether a compact controller can do something resembling Bassem Youssef's cross-examination: inhabit the target's frame long enough to elicit its commitments, transport those commitments to a carefully chosen hypothetical, and let the target state the consequences itself. MIRROR-03 gives that proposal an explicit experimental interface:

1. A **reviewed proposition graph** fixes the exact words of each question. A SHA-256 digest binds every putative semantic-equivalence certificate to a text pair, a stated common scope, a reviewer identifier, and a rationale. Code validates that these records have the required structure; equivalence itself requires independent human review.
2. An **episodic target model** records target answers as `affirm`, `deny`, `qualify`, or `unclear`. Raw text stays in an auditable receipt. Invalid structured replies become `unclear`; no parser coerces conversational hedges into binary commitments.
3. A **pragmatic frame scheduler** can use apparent-agreement probes to elicit explanations and choose the next family of questions. An ironic frame is a probe, excluded from contradiction evidence.
4. A **two-step evidence planner** selects the next member of a certified pair that could produce a witness. An optional MIRROR-02 trained recurrent actor proposes a question using public 3-bit family features, 4-bit context features, and observed answers. Its logits never override the certificate gate.
5. A **witness checker** reports an inconsistency only when both sides of a pre-reviewed pair have current, explicit opposing commitments. Target corrections replace the earlier active commitment; the original exchange remains in the audit history.

The target-specific base commitments and target adapter are inaccessible to policy selection. The controller receives only the public contract and observed responses.

## Running

Requirements: Python 3.10+. Tests run with `pytest`. PyTorch is needed only to train/load the neural controller inherited from MIRROR-02.

```bash
python -m pytest experiments/mirror_epistemic_transport_v0/mirror03/test_v3_bridge.py -q
python -m pytest experiments/mirror_epistemic_transport_v0/mirror03/test_v3_neural.py -q
python -m experiments.mirror_epistemic_transport_v0.mirror03.bridge --target fixture --policy probe-first --budget 5 --output output/mirror03_fixture_receipt.json
python -m experiments.mirror_epistemic_transport_v0.mirror03.bridge --train-checkpoint --train-steps 120 --checkpoint output/mirror03_policy.pt
```

To use an OpenAI-compatible *locally hosted* large model, first copy `example_contract.json` into your own reviewed contract and have an independent reviewer approve the proposed equivalence edges. Then run:

```bash
python -m experiments.mirror_epistemic_transport_v0.mirror03.bridge \
  --target openai --model your-local-model \
  --endpoint http://127.0.0.1:8000/v1/chat/completions \
  --contract path/to/reviewed_contract.json \
  --policy neural --checkpoint output/mirror03_policy.pt \
  --budget 8 --output output/mirror03_live_receipt.json
```

For a hosted HTTPS endpoint, explicitly pass `--allow-remote` and put any API key in the `MIRROR_API_KEY` environment variable. The tool sends no tool-use requests, does not follow HTTP redirects, limits each response to 2 MB, and keeps API keys out of receipts. Receipts contain raw model answers; treat them as potentially sensitive and store accordingly.

The target gets the same structured-answer system instruction on every run. It is asked to emit `{"stance":"affirm|deny|qualify|unclear", "reason":"..."}`. Such instructions change target behavior and must be included as a fixed experimental condition. The stance is an explicit target self-report, not a claim established by parsing arbitrary natural language. Freeform dialogue and independent semantic extraction remain subsequent research tasks.

## Example fixture

The included demonstration contract has two families, an apparent-agreement probe, and two purportedly equivalent direct questions per family. The certificate fields say `example-author`; they are *illustrative*, and they need independent semantic review before use in a substantive real-model experiment. The built-in scripted target explicitly affirms one appeal question and denies the role-swap variant.

Results of the fixture run:

| Policy | Turns to first candidate witness | Non-evidentiary probes |
|---|---:|---:|
| Paired | 2 | 0 |
| Probe first | 3 | 1 |

The witness is mechanically reproducible from the response receipt and illustrative annotation. Its status is **candidate contradiction under the demo annotation**, rather than an independently validated finding.

## Locked MIRROR-03 evaluation protocol

**Train:** synthetic target grammars from MIRROR-02, using answer prediction and supervised question actions. Train the policy before inspecting held-out model dialogues. Build a separate set of paraphrases with true entailment and differing scope.

**Evaluation:** freeze target versions and generation parameters, contract hashes, reviewer decisions, question budgets (4/8/12), model-target pairings, and target-token accounting. Use the same target seeds and questions for matched paired, actor, and pragmatic-frame policies. Run positive and consistent-target controls, ambiguity controls, explicit revisions, unreviewed-transformation traps, and sarcastic replies. Separate a model's *asserted* stance from what a human reviewer judges it to have meant.

**Comparators:** scripted pair search, trained two-pass transport actor, a one-pass actor, no-transport actor, and parameter-matched GRU/untied-block controls. The last three need trained checkpoints and more examples beyond the current fixture. Collect independent annotations blind to policy and target label. Primary endpoint: valid contradictions per 1,000 target tokens at matched witness-precision thresholds, with clustered confidence intervals across target families and targets. Secondary endpoints: clarification quality and genuine self-repair, which should be scored favorably rather than counted as inconsistency.

**Acceptance:** the proposed transport model should outperform paired questioning on previously unseen semantically reviewed transforms and models. A large win on synthetic self-generated labels alone would not satisfy the hypothesis.

## Known constraints

- Written semantic equivalence and the explicit stance schema are needed for automatic witnesses. Reviewers may disagree, and the current machine audit cannot resolve their substantive judgments.
- Apparent-agreement framing is currently a non-evidentiary probe scheduler. It has no learned pragmatic-intention decoder and no demonstrated conversational irony competence.
- The optional MIRROR-02 neural policy uses manually annotated binary family/context embeddings. It is a proposal head rather than an end-to-end natural-language representation learner.
- Synthetic controller training and real-model dialogue measure different things; a positive fixture run cannot validate transfer.
