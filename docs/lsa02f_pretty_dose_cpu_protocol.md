# LSA-02F: CPU pretty-text dosage and independent random-menu Lean baseline

**Registered 2026-11-23.** This is a new *development-only* study following [LSA-02E](../reports/lsa02e_typed_pretty_cpu_20261123.md). The previous 96-byte hybrid improved certified full-proof completion on 32 development roots to 6/32 at both seeds, compared with 2/32 and 5/32 for raw. Its 18.75% completion was below the standing 40% competence floor. The [LSA-02D preflight](../reports/lsa02d_same_worker_field_retention_20261123.md) established exact repeatability on one deterministic CPU worker and cross-host numerical differences. This study obeys that same-host condition and adds a **model-free native uniform-menu proof baseline**.

## Exactly one registered public-state intervention

Five model inputs share the identical Plan B architecture, token/candidate encoder, 57,475 parameters, 384-byte byte-level input cap and AB recurrence:

| Input view | Public proof-state content |
| --- | --- |
| `raw` | Original canonical Lean JSON, whose last 384 UTF-8 bytes reach the model; historical input control |
| `pp0` | Mandatory typed current goal and complete candidate premises, 0 pretty-text bytes |
| `pp48` | The same typed grammar with up to 48 UTF-8 bytes of public Lean pretty-text tail |
| `pp96` | The same typed grammar with up to 96 UTF-8 bytes, direct follow-up to LSA-02E |
| `pp160` | The same typed grammar with up to 160 UTF-8 bytes |

The four dose views are identical except for the **maximum permitted pretty-text length** and the consequent number of optional public background hypothesis relations that fit into the remaining byte allowance. The public text is normalized for whitespace and restricted to complete UTF-8 codepoints. Every mandatory typed candidate's full named premise/head relation and current goal budget is preserved. When the mandatory data exceeds the model input cap, the encoder fails closed rather than truncating a proof-relevant candidate. No private theorem generator fields, oracle labels, verdicts, reference proof or shortest-path length enter inference.

Their checkpoint schemas are `lsa02f.public_pretty_dosage.v1:dose=0`, `dose=48`, `dose=96`, and `dose=160`; they remain distinct from the previously registered `typed_pp` view. Reusing the original `typed_pp` and `typed` syntax is intentional. Its pretty dosage is implemented in `ppt_gym.typed_state` via the same serializer.

## Fresh theorem source and CPU matrix

Freeze a **new native Lean-CGRT Horn suite**, generator seed `20261127`, with 64 roots per split before collecting any outcomes. Only 48 distinct `train` lineages are independently explored and compiled in Lean 4.19.0 to provide shortest-continuation supervision. Only the first **32 disjoint `calibration` roots** are scored. The `ordinary_test` and `ood_family` shards are hash-checked for source integrity by the loader, with their proof outcomes sealed and unused.

Two model seeds are registered: **137 and 139**. Every seed trains the five views with exactly matched starting weights and source-root/frame sample streams. On one physical GitHub Actions CPU worker, each cell undergoes **720 FP32 AdamW training updates**, two AB recurrent passes, four heads and width 32. Each view is independently fitted twice on the same CPU worker to confirm exact parameter, checkpoint and optimizer-step trace equality, yielding **20 fits**. The runtime uses one PyTorch CPU thread, `torch.use_deterministic_algorithms(True)`, disabled MKLDNN and pinned Torch 2.5.1+cpu. Different hosts may still produce different final numerical weights; the within-host matched design controls that experiment variable.

The ten trained checkpoints each receive a **full native Lean proof run** on the same 32 independent calibration roots. The checker allows eight tactic attempts per proof, and supplies actual goal states, tactic verdicts and kernel certificate. Every terminal proof is independently compiled as a standalone axiom-free Lean declaration. All full action transcripts are replayed model-free against the same pinned Lean binaries; frozen block-order AB/BA differences and public-order `auto8` control are retained as secondary data.

## Native random-menu baseline

The prior architecture competence gate needs a registered **uniform public-menu native proof policy**, distinct from the deterministic `auto8` heuristic. The new baseline samples each attempted tactic **uniformly from the same live public menu** after previously rejected tactics at that exact state are excluded. It receives the same eight-tactic budget as every model, and Lean verifies every action exactly as in the neural episodes.

Three fixed random streams are registered, `17`, `31` and `47`, yielding **96 model-free native episodes on 32 roots**. Each draw uses only the fixed experiment seed, theorem-root ID, public state packet, available tactic menu and remaining action budget. Seeded SHA-256 plus `random.Random(...).randrange(n)` implements discrete uniform choice, with independent root-specific seeds. It reads no generator labels, oracle answers or neural logits. Each kernel closure is independently compiled and the episode is replayed from scratch without the model. The final baseline score is the root-weighted average closure over the three streams; its per-seed closure counts and full action traces are archived. The random result is not silently substituted by `auto8`, whose action ranking is deterministic.

This is an empirical benchmark of an independently sampled uniform-menu policy, with **three finite random streams**, not the exact expected success probability over every possible stochastic policy. A stronger confirmatory competence test needs enough random streams to resolve the +0.10 gate with independently measured uncertainty.

## Outcomes and predeclared decisions

The primary developmental contrast is **`pp96` minus `pp0`** on native, kernel-and-standalone-verified full-proof closure. Root-level differences are first paired within each seed and then averaged across the two seeds; a 2,000-draw **descriptive** bootstrap samples the **32 unique theorem lineages**. The secondary contrasts `pp48` and `pp160` against `pp0` and `pp96` against raw are reported in full, including null outcomes.

**Positive developmental screen:** mean `pp96 - pp0` closure at least **+0.05 absolute**, and `pp96` ties or beats `pp0` at **both** matched seeds. This is an exploratory advancement signal, not a confirmatory statistical claim. We will report the entire five-view curve and independent uniform-menu baseline regardless of the screen.

The formal **competence preflight** separately requires 32 independent development roots, trained-order close rate **at least 0.40**, and at least **+0.10** over the independent uniform-menu native closure rate. Meeting it would justify a **new sealed confirmation**, subject to explicit resolution of root and random-baseline uncertainty, FSI-01, and generator-ancestry checking. It does not itself authorize architecture promotion or merging with outside Mathlib proofs. If this gate fails, the disposition remains `BLOCKED_INCOMPETENT_CHECKPOINT`.

The random baseline is measured on the same roots but uses three distinct random streams; its 96 episodes are **correlated by 32 theorem roots**. Neural seed/root combinations are also repeated measures. Neither count may be used as an independent `N=96` confirmation sample.

## Evidence custody, costs and merge rule

The [CPU GitHub Actions workflow](../.github/workflows/lsa02f-pretty-dose-cpu.yml) has six stages: code/unit tests, source generation/standalone certification, one-worker same-source/deterministic training, uniform-menu native baseline, ten native checkpoint proof evaluations (at most two simultaneous), and one source-hash-bound root-paired comparison.

The final audit confirms matching source/checkpoint SHA-256 receipts, full Lean kernel/standalone certificates, replay manifests, fixed root and seed rosters, retained candidate premise types at each dose, byte-budget compliance, same-worker parameter identity and zero query-label updates. Any failed receipt is a hard research infrastructure failure, which is fixed visibly in a subsequent commit. All work uses GitHub-hosted CPU and RAM, with no GPU job or paid inference API.

The first development run is retained in short-lived GitHub Actions artifacts (14-day retention). A committed Markdown report will preserve the results and hashes after CI. Locked ordinary/OOD test outcomes remain unopened. A gym requires the competency and independent confirmation gates to pass.
