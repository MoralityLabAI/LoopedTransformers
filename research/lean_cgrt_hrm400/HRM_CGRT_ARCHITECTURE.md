# HRM + CGRT architecture (lcgrt v2)

Code: [`lcgrt/model.py`](lcgrt/model.py), [`lcgrt/encode.py`](lcgrt/encode.py), [`lcgrt/search.py`](lcgrt/search.py). Parameter counts are exact (`count_parameters`); FLOPs are analytic forward FLOPs per decision on a 42-token graph, the training mean ([`results/hw_diag.json`](results/hw_diag.json)).

## Input: the Lean proof state as a typed graph

The tokens are:
- one per proposition atom;
- one per hypothesis (fact or rule; each is the candidate action `apply <hyp>`);
- one `constructor` token;
- one virtual conjunction node for `A ∧ B` goals.

Edges come from the hypothesis types Lean reported. There are 9 typed incidences: self, hyp→head, hyp→premise, and their reverses, plus conjunction links. Per-token features (12) are what the search state makes observable:
- main goal, open goals and their order;
- goals on the current path;
- goals this search has shown to be dead;
- Lean-legal (from `expand`), and pruned by the cycle rule;
- head matches the main goal;
- arity, and premises on the path or dead.

Teacher actions, proof depth and generator metadata are never inputs. The graph has no positional or name embeddings, so the network is equivariant to atom renaming and hypothesis order.

## Two-clock HRM

```
x0 = W_feat·features + E_type                    (once per decision)
s  = slot_init + W_glob·global                   (M = 8 slow slots)
x  = 0
for t in 1..T:                                   (fast clock; ONE block, tied weights)
    u = x + x0                                   (input re-injection)
    u = u + TypedAttention(LN u; edge-type bias)
    u = u + CrossAttention(LN u → slots)         (fast reads slow)
    u = u + MLP(LN u);   x = LN(u)               (post-norm bounds the recurrent state)
    if t % k == 0:                               (slow clock, k = 2)
        s = LN(s + SlotSelfAttn + Read(s ← x) + MLP)
        readout: candidate logits (from [x_c, x_main, x_c ⊙ x_main]), atom provability, halt(s)
```

- **Fast module:** one shared-weight block, applied *T* times over the frontier graph (goal atoms, candidate tactics, dependency edges).
- **Slow module:** eight slots updated every *k* fast steps. They read the whole graph, write back through the fast block's cross-attention, and carry the halting head.
- **Deep supervision:** every slow tick's readout is scored (the shortest-proof set loss of LR-01, atom-provability BCE, halting BCE). The halting target is "this tick's top choice starts a shortest proof".
- **Adaptive depth:** at inference, `--halt τ` takes the first tick whose halting probability exceeds τ. The fast steps used are reported per decision.
- **Test-time depth:** `--fast-steps 16` runs the tied block longer than in training.
- **Stability:** post-norm on the state, input re-injection, gradient clip 1.0. The repo's earlier tied-recurrence runs showed gradient norms of 149–166 at high loop counts.

### Built and not built

| question in the brief | status |
|---|---|
| fast computation over frontier, candidates, dependency edges | built (typed graph attention, tied) |
| slow persistent state managing goals, branches, backtracking, compute | slow slots run on their own clock inside a decision and drive halting. **Across decisions they are re-initialised.** Branches and backtracking are handled by the search controller, using the model's predicted outcomes. Search history reaches the model as explicit features (dead goals, path), not as carried slow state |
| slow state as a compressed search history | **not tested.** The carry-over variant (slots persisted across expansions, trained by truncated BPTT on search traces) is the next experiment. This campaign keeps the explicit-history version, so the comparison has a working baseline |
| global budget/goal features into the slots | wired (`N_GLOBAL = 5`), but **zeros in training and evaluation in c1** (training states come from shortest-proof replays, which have no budget history) |
| adaptive inference depth and halting | built and trained; evaluated by `evalhalt_*` jobs |

## Parameter-matched models

| preset | arch | width | params | fwd GFLOP/decision |
|---|---|---:|---:|---:|
| HRM-S (`S`) | tied fast + slow, T = 8 | 384 | 5,186,387 | 1.75 |
| HRM-M (`M`) | tied fast + slow, T = 8 | 672 | 15,849,875 | 5.33 |
| HRM-L (`L`) | tied fast + slow, T = 8 | 928 | 30,202,771 | 10.14 |
| flat-S | 8 untied graph blocks | 232 | 5,358,682 | 0.46 |
| flat-M | 8 untied graph blocks | 400 | 15,891,442 | 1.35 |
| flat-L | 8 untied graph blocks | 552 | 30,236,442 | 2.57 |
| S-loop | tied fast block only, no slots or halting (single clock) | 592 | 5,278,354 | – |

At equal parameters the flat model has as many propagation hops as HRM's eight fast steps, at about a quarter of the FLOPs. HRM at T = 2 is roughly FLOP-matched to flat. The recurrence ablation (T = 1, 2, 4, 8) spans that range, and evaluation reports model FLOPs next to Lean calls.

## Typed control layer

The [`search.Controller`](lcgrt/search.py) enforces, for every policy:
1. **Legal actions:** the only actions are tactics Lean accepted on the main goal (`expand`).
2. **State transitions:** the next state is the goal list Lean returned.
3. **No goal repeated on its own path:** a child whose new goal equals one of its ancestors is discarded, since no minimal proof does that.
4. **Dead goals:** a goal is dead when Lean offers no tactic for it, or when every Lean-legal tactic leads to dead goals with nothing pruned. Children containing dead goals are discarded, and the dead set is fed to the model as history.
5. **Verifier acceptance:** success means a node with no goals *and* a passing standalone certificate ([`lcgrt/certify.py`](lcgrt/certify.py)).

## CGRT (Cross-Generation Reasoning Tree) at the proof frontier

v1's contexts (rule-order permutations) were a symmetry of the network, so every branch was identical. The v2 contexts are variations the network is **not** invariant to, applied to the same Lean state:

| context | variation |
|---|---|
| c1 | trained recurrence depth T |
| c2 | 2T (more fast computation) |
| c3 | T with the search's dead-goal history hidden |

At each expansion, all contexts score the Lean-legal children. Children are shared across contexts, giving one tree in which the same tactic from different contexts is one node. Each child gets the 9 branch features in `BRANCH_FEATURES`:
- mean, max and min log-probability across contexts;
- the fraction of contexts that rank it first;
- predicted child value: Σ log σ(atom-provability) over the goals it opens;
- number of new goals, number of siblings;
- Lean-verified dead or pruned siblings.

A logistic **branch scorer** maps these features to P(the tactic starts a shortest proof). It is fit by [`lcgrt/branch.py`](lcgrt/branch.py) on search traces of *training* theorems, labelled by Lean's exploration table, and checked on held-aside training theorems. Search is best-first on the cumulative log branch score.

Controls at the same Lean-call budget:
- `cgrt_shuf`: the same contexts and budget, but the branch scores are permuted among siblings at every expansion;
- `best`: best-first with one context and the untrained prior score;
- `dfs` and `greedy`;
- model-free `random_dfs` and `bfs`.

CGRT spends 3 forward passes per expansion; the reports show model forwards and GFLOPs next to every success rate.

## Smallest convincing experiment first

The campaign order follows the brief. First HRM-S vs flat-S on identical data (one seed, then three). Then recurrence depth, graph removal and the single-clock loop at S. Then M and L. CGRT is evaluated only on HRM checkpoints that already exist. Nothing larger was built until the S-size comparison was running.
