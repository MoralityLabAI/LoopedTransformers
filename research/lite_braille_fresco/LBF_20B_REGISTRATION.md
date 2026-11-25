# LBF-20B: native conditional ground rewriting and factored verifier control

Development-only engineering pilot. Continue from LBF-20A and the LBF-19 independent review. Freeze this source and design before the first cloud run. No sealed evaluation, historical checkpoint reuse, or prospective superiority claim.

## Question and scope

Can a graded ground-rewrite distance help rank native Lean tactics when a potential rewrite requires a separate premise proof? The domain deliberately exposes a strong alternative: a context-wide guard can be proved once and reused at every occurrence. Do not infer that this restricted domain requires native checking of every intermediate successor. The public guard text and hypotheses are visible; a stronger dedicated arithmetic/list solver may predict the guard outcomes. There is no information-theoretic black box claim.

Goals are equalities of ground terms over an arbitrary type, six constants and unary/binary constructors. Rules have hypotheses h_i : P_i → lhs_i = rhs_i. Side conditions use local propositions, Nat order/addition and List expressions. Lean 4.19.0 executes a fixed `first | (assumption; done) | (rfl; done) | (simp; done) | (omega; done)` portfolio in the actual proof context. A rejected portfolio is not evidence of logical impossibility. Successful intermediate tactics must return the exact predicted equality; a terminal proof is checked by Lean's kernel. Final standalone source recompiled separately. Standard Lean axioms propext, Quot.sound and Classical.choice are permitted and explicitly counted; sorryAx or any other axiom fails.

## Freeze

Seed 2026112503; 12 train and 18 development fixtures generated independently using the fixed procedure. Three shared structural templates (leaf, growth, composite), random symbol/guard order and tree shapes. This is a split of generated fixtures, not a guaranteed canonical-lineage or theorem-family holdout. Labels come from actual native acceptance, never a hidden generator truth label. Shared template and guard vocabulary may enable interpolation. No train update from dev.

Train a 129-parameter logistic guard scorer on all train-rule native receipts: normalized hashed character 2/3-grams (128 dimensions plus bias), full-batch gradient descent, 160 steps, lr 0.6 and L2 0.01. Serialize the checkpoint and SHA-256. This is a simple learned baseline, no recurrent neural architecture claim.

## Seven paired arms

1. Eager edit-distance action search (cheap structural mismatch heuristic).
2. Eager edit plus learned guard score.
3. Eager exact optimistic min-plus ground distance (reverse automaton rooted at target).
4. Eager graded plus learned guard score.

Common best-first action priority g+1+h+3(1-p), p=1 for unlearned arms. Same candidate order, horizon 20, no hidden outcome access. Lean must execute each selected conditional equality proof step before successor admission.

5. Lazy graded: find a shortest optimistic ground path; probe its unresolved guards in path order; retain successful guard hypotheses, remove failed portfolio rules and replan. When all needed guards are proved, construct the final proof from cached guard witnesses and submit it in one native tactic.
6. Lazy fail-first: same, except choose the lowest learned success probability among unresolved guards on the current optimistic path.
7. Probe-all then defer: probe every rule once in fixed order, use exact weighted completion over accepted rules, and submit the complete proof.

All arms use a maximum 24 native tactic requests per root and 4-second soft wall cap checked before requests (late terminal results flagged and excluded from timely closure). Wall includes reset, initialization, proposal computation, tactics and kernel check. Archive separate timings; standalone batch compile reported separately. The native process has a bounded per-request timeout; errors are not silently relabeled as ordinary failed tactics. The optimistic distance ignores guards, and exactness is only inherited for the ground unconditional abstraction.

## Engineering gate and descriptive outcomes

All native states match the predicted equality, all kernel-accepted paths compile with only permitted axioms, no budget violations, 126 paired development episodes, both acceptance/rejection classes in train. Report closure at 8/16/24 calls, actual root-level closure, calls, rejected calls, wall and native time, late or exhausted statuses, axioms. Do not treat roots or repeated arms as independent training-seed observations. Retain all negative outcomes; no binary superiority gate. If a source-level failure prevents execution, record and repair it explicitly, preserve the original registration and distinguish any change to design.

Follow-up if viable: move beyond context-wide premises to state-dependent obligations (unification, branching subgoals, local-context edits) and test a recurrent prior with matched direct controls. Any recurrence training requires separate registration and population-level evaluation. Candidate checkpoints can enter DataGoblin's discovery-only Model Atlas after provenance review; dev outcomes stay in evaluation custody.
