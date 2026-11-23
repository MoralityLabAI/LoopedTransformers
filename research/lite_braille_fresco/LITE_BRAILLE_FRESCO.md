# Lite-Braille Fresco (LBF): a typed luminous proof-trace gym

**Specification:** `lbf.v1`  
**Status:** executable representation and action-selection pilot, 2026-11-23. Synthetic fixtures are engineering controls. All native model-performance and causal-mechanism claims require separate evidence.  
**Parent:** [LSA-02 typed architecture calculus](../../docs/lsa_typed_architecture_calculus_v0.md), [PPT Plan A/B](../lean_cgrt_hrm400/PPT_PLAN_B_INKLING_BLOCK_MESSAGE_MEMORY.md), [native Lean AB/BA bridge](../../docs/lsa02b_native_lean_bridge.md).

## 1. Purpose and semantic contract

Build an inspectable, low-bandwidth visual alphabet for recurrent states and Lean proof-search traces. The physical metaphor combines an LED board's local intensity, a fixed Braille-cell dot geometry, hieroglyph-like shape families, and the arrangement of a narrative fresco. A fresco is a *sequence of measured predecision states*. Its symbols are a projection of an instrumented system, not evidence that the system internally uses a symbolic language.

The encoder never infers proof truth. Its output is observational. A simulated fixture never becomes a kernel certificate by rendering it. Genuine completed proofs require a separate native Lean run, terminal kernel result, standalone compilation, axiom audit, and independent replay, following PPT-G03.

The dot patterns use Unicode Braille-cell geometry as a positional visual code. They are **not** linguistic Braille and are not presented as tactile accessibility output. The exported JSON and SVG text/title labels provide a textual alternative.

## 2. Formal alphabet and encoding

Let the nominal slot-role alphabet be

\[
R=\{\mathrm{goal,context,lemma,plan,value,message,halt,delegate}\},\quad |R|=8.
\]

Let the frame alphabet be

\[
F=\{\mathrm{state,action,gate,memory,closure,error}\},\quad |F|=6,
\]

with polarity set \(P=\{-1,0,+1\}\), finite activation \(a\in[0,1]\), and four LED levels \(Q=\{0,1,2,3\}\). A tile has declared type

\[
T=(c,r,f,a,p),\quad c\in C\cup\{\mathrm{shared}\}.
\]

`c` is an **opaque candidate identifier**. Slot-role names are interface labels, not validated neural interpretations. Every tile is associated with an explicitly instrumented source field or a declared synthetic generator.

### Braille-cell assignment

Braille dots use the conventional 1/2/3 (left top-to-bottom), 4/5/6 (right top-to-bottom), 7 (left bottom) and 8 (right bottom) bit numbering. Encode \(i=\operatorname{index}_R(r)\in[0,7]\), \(j=\operatorname{index}_F(f)+1\in[1,6]\):

\[
\mu(r,f,p)=i+(j\ll3)+64\,[p=+1]+128\,[p=-1],\qquad
G(r,f,p)=\mathrm{U+2800}+\mu(r,f,p).
\]

The three lower bits encode the role, the next three bits encode frame, and dots 7/8 encode polarity. Both polarity bits set simultaneously and a zero/out-of-range frame are malformed. The canonical alphabet has \(8\times6\times3=144\) glyphs. The three fields occupy disjoint bit positions, so **the mapping is injective**. Decoding reconstructs the exact triplet \((r,f,p)\). The tests exhaustively check the 144 cases.

Quantize brightness deterministically:

\[
q(a)=\min(3,\lfloor4a\rfloor),\quad a\in[0,1].
\]

This preserves four ordinal bins. It **discards** within-bin magnitude; the glyph/LED pair represents \((r,f,p,q(a))\), not the original real-valued vector. For instrumentation with unbounded values, the adapter must declare its prespecified bounded mapping. For the initial PPT bridge, each frozen slot's RMS norm \(u\ge0\) maps to \(u/(1+u)\); candidate logits map through the logistic sigmoid. These scales are monotone, semantically distinct and must not be compared as calibrated probabilities.

### Fresco time and arrangement

A visit is \(v=(t,k)\), ordered lexicographically: proof tactic attempt \(t\in[1,32]\) and inner recurrent pass \(k\in[1,16]\). The visit contains a set of tiles, at most one per `(candidate_id, role)`. A fresco is a word of visits

\[
\mathcal F=(V_1,V_2,\ldots,V_n),\quad V_i\prec V_{i+1},\ n\le64.
\]

Columns depict these chronological visits. Rows are `(candidate, role)` identities. The drawn frame outline depicts `frame`, the populated dot positions depict `glyph`, and LED opacity corresponds to `light`. Palette redundantly conveys polarity. An omitted tile is blank rather than zero, maintaining missing-versus-observed distinction.

## 3. Data and evidence separation

`lbf.observation.v1` is **predecision-only** and carries exactly the following keys:

```json
{
  "schema": "lbf.observation.v1",
  "root_id": "fixture-7-0000",
  "lineage": "toy-family:7:0000",
  "split": "calibration",
  "source": {"kind": "synthetic_fixture", "source_sha256": "<sha256>", "checkpoint_sha256": null},
  "candidates": ["t0", "t1"],
  "visits": [{"attempt": 1, "pass": 1,
    "tiles": [{"candidate_id": "t0", "role": "value", "frame": "action", "activation": 0.7, "polarity": 1}]}]
}
```

The example is a **shape illustration**, with `source_sha256` standing for an actual 64-character lowercase SHA-256 string. Each observation must contain one or more candidates and visits, strict increasing `(attempt, pass)` clocks, and finite bounded activations. The encoder rejects unknown keys, including extra keys carrying hidden `verdict`, `certificate`, `success` or reference answers.

`lbf.encoded.v1` replaces each tile with only `candidate_id`, Unicode `glyph` and integer `light`; carries the source identity and a SHA-256 of the canonical raw observation; and discards source magnitudes. No source outcomes are copied into it. Its root and lineage identifiers allow joining with an independently held label bank only inside the scorer.

`lbf.labels.v1` is a **private evaluation artifact** with one status per candidate: `success`, `failure`, or `censored`. Its evidence class must agree with the observation source: `synthetic_fixture` or `externally_attested_native`. The local tool does not independently verify a native attestation, so **all score outputs are marked `scientific_eligible=false` and `certification_replayed=false`**. Real Lean imports require a separate verified-candidate receipt exporter and ancestry gate. A censored root is removed from matched policy scoring and counted in `roots_censored`.

No algorithm may train on calibration, ordinary test, OOD or previously exposed historical c4 roots. Root lineage rather than repeated visits is the independent unit of analysis. Immutable manifests record raw/encoded/label file hashes. `encode` and `render` perform zero Lean calls and no gradient updates.

## 4. Gym contract and reward

`FrescoEnv` exposes the **current and earlier** encoded visits and the candidate roster. It holds the label bank privately. An agent may issue:

- `continue`: reveal the next registered pass, pay \(c=0.02\) per step by default.
- `select(candidate_id)`: terminal action; receive \(1\) only when the privately recorded candidate label is `success` and \(0\) for `failure` or `censored`. Censored results are explicitly returned with their separate status.
- `abstain`: terminate with zero new reward.

For a fixed episode with selected candidate \(a\) and \(m\) continuation steps, the total utility is

\[
U=\mathbf1[\ell(a)=\mathrm{success}]-cm.
\]

`step().reward` is the **incremental** reward; `step().total_reward` carries the cumulative episode utility. A censored result is not proof failure. For scientific learning, censoring must be handled by a registered rule or bounded separately, with the native proof-verifier budget included in costs.

The included policy probes are `glyph` (highest `value`/`action` LED level), `led` (highest candidate LED level independent of role), and deterministic hashed `random`. They are fixed heuristic comparators and have no learned parameters. They evaluate against the same candidate roster and complete-label roots. The gym API also accepts external policies via `observation()` and `step()`.

## 5. Representation ablations and causal boundary

The scorer includes paired, root-matched **representation** controls:

| Control | Changed field | Test |
|---|---|---|
| `intact` | none | Original glyph/LED projection |
| `zero_value` | LED level of `value` tiles | Dependence on candidate-value visual signal |
| `zero_message` | LED level of `message` tiles | Sensitivity to displayed message channel |
| `roll_roles` | Role bits rotated through the eight-role alphabet | Dependence on codebook assignment, with raw LED levels fixed |

All four preserve the source neural model and ground-truth labels. They demonstrate whether a *reader/policy* relies on the encoded symbol. They **cannot** establish neural routing causality. A causal mechanism study additionally re-runs a frozen checkpoint with real `ppt_gym` message `normal`/`zero`/`roll` changes, equal check/forward budgets, matched sham and source-independent reference receipts.

The interface is a quotient projection of the continuous trace. Define \(E:S\to\mathcal F\), where \(S\) is the typed observed trace space. Two states are LBF-equivalent when \(E(s)=E(s')\). A learned reader acting only on \(E(s)\) is unable to distinguish these states. Whether the quotient preserves the information needed for proof decisions is an empirical question. An admissible paper result must compare LBF with a *same-budget full-precision readout*, quantized brightness without glyphs, shuffled glyph assignments, and an equivalent-token text representation.

## 6. PPT bridge and promotion protocol

`ppt_bridge.py` loads an already-trained Plan A/B checkpoint using the existing `ppt_gym.cli.checkpoint_load`, and re-forwards the stored public `events[].input`/`candidate_menu` from the native Lean episode ledger, with `return_all=True`. It creates fresco visits for each tactic attempt and recurrent pass. It exports model coda slot RMS norms and candidate-policy sigmoid outputs as observation tiles. `state` returned by Plan B is **coda output**, so the bridge is **not** an instrumented snapshot of raw middle-block messages or gate Q/K/V states. The nominal role labels follow existing eight-slot indexing. The file includes checkpoint and episode-ledger hashes and always says `role_names_are_nominal=true`, `labels_exported=false`, `lean_calls=0`. It does not export private verifier labels.

Near-term progression:

1. **LBF-00: codec/conformance.** Exhaust 144 canonical glyphs, finite-value guards, strict predecision schema, deterministic hashes, XML-render validity, and label nonleakage. Standard-library-only CI.
2. **LBF-01: matched toy reader.** Evaluate synthetic complete-label roots using glyph, LED, random and visual ablations. A good outcome is an engineering demonstration that typed dots can preserve a cue lost to brightness-only reading; it is not a novel model result.
3. **LBF-02: frozen PPT instrumentation.** Export native `PPT-G03`/`LSA-02B` live episode ledgers with pinned checkpoint; verify observed slot/value probes have stable hashes. Add explicit middle A/B/gate hooks before interpreting these channels as mechanism evidence.
4. **LBF-03: native prospective routing.** Freeze lineage-disjoint verified-candidate roots with complete candidate outcomes. Train matching readouts over full precision vs LBF vs LED-only at equal total compute and Lean budget. A continuation decision must be priced by *actual* extra forward and verifier calls.
5. **LBF-04: mechanism and transfer.** Run paired native message interventions, record real kernel-complete scripts, independent replay and matched shams. Test transfer across BPD directed, equality, Horn and later Mathlib families. Report root-paired effects and uncertainty under the EvalMath finite-sample gate.

Promotion needs a predeclared minimum useful utility effect (pilot reference +0.025), more than one independent model seed, matched root lineages, comparable outcome coverage and a prospective sham. A visually compelling fresco or a synthetic 100% score is never sufficient.

## 7. Resource bounds and reproducibility

The pure codec uses Python standard library on CPU; no GPU and no Lean environment. It bounds candidate count to 64, visits to 64 and tiles per visit to 512. Python source validates every raw numeric activation. It writes new directories only and keeps source observations immutable. SVG is generated with escaped labels and does not execute scripts or fetch remote resources. Rendering cost grows with the number of visible tiles and visits, so clients should subsample for large sessions.

The PPT bridge optionally requires CPU PyTorch and the pinned Lean-CGRT `ppt_gym` package only to load and re-forward a checkpoint. It performs no native Lean calls and cannot claim transfer accuracy from replayed original episode outcomes. Tests and CI exercise the pure codec; a native-checkpoint integration run remains a separate registered gate.

### Initial engineering receipt

Local Python 3 run of `fixture(40,20261123)` followed by `encode`, `render` and `score` passed the codec path and produced a well-formed SVG. On this **deliberately constructed** synthetic control, `glyph` selected the designated candidate on 40/40 roots, `led` on 9/40, and `random` on 7/40. Role rotation reduced the glyph reader to 9/40. The source labels were generated directly by the fixture itself, so these numbers are a program sanity check only.


## 8. First native-PPT integration receipt (LBF-02/03)

A frozen **synthetic-fixture-trained PPT-B checkpoint** was re-forwarded over the public predecision packets from a real Lean 4.19 episode campaign. The native suite and original paired AB/BA/auto8 episodes were generated by the existing LSA-02B runner; its separate model-free replay certified all six arm episodes. The LBF adapter verified the source session manifest and replay receipt SHA-256, excluded the non-neural auto8 arm, and generated four neural-arm fresco traces, with three recurrent passes per decision. See [successful native CI run 45178884393](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45178884393) and its downloadable `lbf02-native-fresco` workflow artifact.

One frozen checkpoint SHA-256 is `cc85678316aec738d195187088877d0036fc1a89597a7442ae62022830f6760f`; native episode ledger SHA-256 is `7124cb3d19d5595b43bb0c70888d95f1266f3fd4622cbed40b5f94125311fc26`. Four observed neural-arm episodes were recorded under `lbf.ppt_bridge.v1`, and render validation succeeded. These were **observational outputs of a model trained for two synthetic optimization steps**. The native proof outcomes are meaningful as test-harness events; neural performance and learned mechanisms remain scientifically ineligible.

LBF-03 adds two actual frozen-model **counterfactual** message modes, `zero` and `roll`, re-evaluated on every identical public input and recurrence depth. Each paired row records `(root_id, attempt, pass, message_mode)`, max absolute candidate-logit displacement, RMS displacement of coda output, and whether the highest-logit candidate changes. This is a controlled computational intervention on a neural network, although the pilot checkpoint is weak and any apparent effect on coda activations does not establish change in full Lean proof success. Pass one must be equal under all message modes because the saved previous-pass bank is identically zero; the CI tests this structural invariant. As with the fresco, these probes do not execute new Lean tactics, reveal future proof outcomes, or update weights.

A future causal proof-completion experiment should preselect trained, proven-competent checkpoints, paired source-disjoint native roots, and a finite-sample matched sham. Its endpoint is kernel-certified completion at an equal computational budget. The displayed `message` slot presently measures a nominal coda channel; direct middle-bank hooks are a separate instrument.
