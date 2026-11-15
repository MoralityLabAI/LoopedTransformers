# AF-13 mathematical follow-up: bounded fairness and fixed-point stutter stability

This is a **post-scoring deduction** from the AF-13 local failure. It is separate from the frozen AF-13 training registration and changes no scientific weights or cutoffs.

## Finite Horn closure under bounded fairness

Let L be the least definite-Horn closure of initial facts S0 in a finite universe U. Define the *minimal proof height* d(v) recursively: d(v)=0 for v in S0; for v derivable from other facts, let

d(v) = 1 + min_{(P→v) in R} max_{u in P} d(u),

with infinity for unreachable atoms. Consider a monotone asynchronous execution S_{t+1}=S_t∪E_t that only fires rule heads enabled by the current state. Call it **b-bounded fair** if an enabled rule whose head remains absent fires within the next b update opportunities (b>=1).

**Lemma.** Every derivable atom v of height d(v) belongs to S_{b d(v)}. Proof by induction on d(v). Initial facts have height zero and belong to S0. Choose a rule minimizing the height of v. Each parent has height <=d(v)-1 and belongs to S_{b(d(v)-1)} by induction. All these parents remain present because the update is inflationary. Therefore the chosen rule remains enabled and by b-bounded fairness its head appears by the next b updates, at time b d(v). QED.

**Corollary.** The exact Horn closure L is reached by time bD, where D=max{d(v):v in L}. This is a conditional discrete guarantee. AF13's neural recurrence has no observed bound b and its probabilistic losses do not certify either soundness or fairness, so the lemma is a proposed design principle rather than a trained-model theorem.

## Stability after closure

Finite definite Horn closure is idempotent: Phi_R(L)=L. If a recurrent code c_t encodes L, then a semantically faithful *stutter* should preserve the decoded abstract state under additional calls:

  decode_abstract(F_theta^k(c_t)) = L, for every k>=0.

This is logically stronger than a single endpoint CE: it requires all extra recurrence lengths to stay correct. AF13's eligible seed17 hits depth4 joint-Q accuracy 1.0 after 8 calls but only 0.765625 after 10 calls. At depth5, the final-relay model improves from 0.765625 at 10 calls to 1.0 at 12 calls, while the asynchronous arm improves from 0.5625 to 0.765625. These observed horizon dependencies support testing an explicitly supervised **first-correct / persistently-correct** mechanism at multiple prefixes.

## Prospective AF-14 hypothesis

A future registration could couple (a) a proof-depth-sensitive progress or fairness witness that bounds enabled-but-unfired Horn obligations, with (b) a stutter-invariant abstract Q state once a correct closure is reached, comparing against final-relay and asynchronous baselines at matched extra recurrence calls and training FLOPs. Label-producing source remains the host Horn oracle, while latent state and layer-local enforcement remain experimentally observed. The AF13 negative-result ledger remains unchanged.
