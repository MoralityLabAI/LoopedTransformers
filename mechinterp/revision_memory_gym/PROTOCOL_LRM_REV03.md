# LRM-REV-03: continuation memory and Bayes reminder frontier

**Evidence class:** exact finite theorem, exact unconditional population calculation and source-only tests. The results are mathematical properties of the declared 12-bit generative law. No neural model training, causal mediation, scientific confirmation, or revised LRM-REV-02 scoring occurs here.

## Why this study

LRM-REV-02 trained four 4,995-parameter GRUs to process twelve named original values, six accepted replacements and six truthful reminders. All four models failed the 0.02 competence gate after reminders; corrected-slot reminder advantage averaged 0.008553539884 unhalved Brier versus four random-slot active controls. We need to separate the **online update memory** needed for later edits from the **current posterior** and the information gained by disclosure to a truly compressed observer.

## Generative law and observational contract

Use `n=12`, exactly `j=6` uniformly chosen corrected positions S, independent of the latent K. The regime prior is `(9/20,7/20,1/5)` and its bit-match probabilities are `(19/20,13/20,3/20)`. Old bits in S are fair provisional values; all other old bits and the six accepted replacement values are independent Bernoulli with probability `p_K`. The reference `p_H` is full named-history Bayes after the six accepted replacements.

A learner with the complete original X, schedule S and replacement Y already has all evidence for `p_H`. Truthful repetitions are conditionally redundant for that learner. A **count-only observer** discards the identity of each original old bit, retains only `m=sum(X)`, subsequently receives S and Y, and can optionally see truthful old-bit disclosures at named reminder positions. For the count observer, those messages reveal information lost through compression.

The observational interface `count_posterior_from_public(m,S,Y,D)` is deliberately restricted to `(m,S,Y,D)` where D maps disclosed position to old value. It receives neither undisclosed old bits nor the private K. Its posterior is calculated with exact rational arithmetic. The named oracle receives the original X and returns exact `p_H` after overwrites.

## The continuation partition and bit lower bound

Fix all-zero replacement Y. For each `X ∈ {0,1}^12`, define its future-output signature over the 924 six-position schedules S as the vector `(|X outside S|)_S`. Each accepted count gives a distinct posterior under the declared regime likelihoods. The source recomputes every signature and verifies all 4,096 are distinct.

The distinctness also follows without enumeration. For any two different old vectors with the same count, their difference has positive and negative coordinates. Choose one of each and any five other indices. Swapping these two coordinates inside a six-position schedule changes the removed sum by two, so at least one schedule distinguishes the original states. If original counts differ, an analogous schedule exists: if every six-subset sum of their difference were zero, swapping any two indices makes all differences equal, and their only possible nonzero constant would have nonzero six-subset sum. Thus every different pair is distinguishable.

An unconstrained exact fixed-width online state needs 12 bits to distinguish 4096 possibilities. If the original count `m` must remain a separate retained label, its count-6 fiber has `C(12,6)=924` distinct continuation states. The minimum **additional** fixed-width memory is `ceil(log2 924)=10` bits, assuming arbitrary per-count tag assignments. This counts neither the original count label, its encoding bits, external archives nor computation. The result is a conditional lower bound for exact operation on every legal edit sequence; it is not a neural hidden-state capacity theorem.

## Exact correction-stage teacher for a named-history model

After r of the six replacements have arrived, the learner knows their positions and values. **The event order is ascending**, which reveals that all future correction positions must exceed the largest position seen so far. This restriction matters even when the schedule was sampled uniformly before the data were generated. Ignoring it can change the posterior substantially.

For r=0, define an empty revealed prefix. For r≥1, let `c` be the last revealed corrected position, `A={i>c}` be the still-eligible suffix of size `a`, and `B={i≤c : i not already corrected}` be the necessarily unchanged prefix of size `b`. Let `m_A=sum(X on A)`, `m_B=sum(X on B)`, `h=6-r`, and `y=sum(Y_revealed)`. The unnormalized weight for regime match probability p is

```
prior(K) * p**(y+m_B) * (1-p)**(r-y+b-m_B) *
  sum_{z=0..h} C(m_A,z) C(a-m_A,h-z) (1/2)**h
      * p**(m_A-z) * (1-p)**(a-h-(m_A-z))
```

with out-of-range binomial coefficients zero. At r=0, take `A` as all twelve old positions and `B` as empty. The source implements this causal teacher in `partial_named_posterior`, using the actual sorted-event information set. At r=0 it equals the old-count initial posterior; at r=6 it equals `p_H`. An independent test oracle enumerates only those remaining correction schedules whose positions are strictly larger than the last revealed one. For the illustrative old stream `111111000000` with first corrected position 4 and accepted replacement 1, the ordered teacher is approximately `(0.463024,0.536250,0.000727)`. Averaging over every remaining slot as though order were uninformative gives a different target, approximately `(0.114830,0.794562,0.090608)`.

This teacher supports a proposed prospective GRU training intervention: supervise the post-OLD state and **each of the six replacement states** against its properly timed Bayes target. It would give the recurrent model a learning signal for indexed subtract-and-add updates while conditioning on the event protocol. Any future training run must register separate model roots, checkpoint choice, matched compute/exposure and reserved data before fitting.

## Count-only exact posterior after old disclosures

Let `dc` corrected slots and `du` unchanged slots have disclosed original bits, with respective sums `xc` and `xu`. Put `rc=6-dc`, `ru=6-du`, `m'=m-xc-xu`, and `y=sum(Y)`. The compressed observer's likelihood weight for p is

```
prior(K) * p**(y+xu) * (1-p)**(6-y+du-xu) *
  sum_{a=0..rc} Binom(rc,a;1/2) Binom(ru,m'-a;p)
```

where `Binom` is the binomial probability mass function. The formula marginalizes all undisclosed original bits under the declared law. The independent test oracle enumerates all `2^12` original bit vectors and checks equality for arbitrary mixed corrected/unchanged disclosures.

## Exact population comparison of reminder policies

The analytic risk at reminder stage r is the unconditional expectation over K, all old/replacement bits, uniform S and an independent uniform six-position T of `||p_count(r) - p_H||²`. In the corrected arm, the first r reminders disclose corrected old slots. In the random arm, the first r reminders are the first r positions of T; uniform S implies that its overlap count q follows the appropriate hypergeometric law. We sum independently over q, old-bit counts and replacement counts with exact `Fraction` arithmetic, including coincident T=S. All stages r=0..6 are reported; only the endpoint r=6 is the specified policy contrast.

The program reports decimal displays plus SHA-256 of the full rational numerator/denominator. The `--check` mode independently reruns the complete finite enumerator and verifies the checked-in receipt and source-file hashes. The policies supply different information to this count-only observer. They are active controls in the earlier GRU setting, not matched-null shams.

## Interpretation and follow-up

The count-only oracle's stage-wise risk curve is a **population Bayes information bound for this deliberately compressed observation interface**. It is separate from the four trained GRUs' empirical model-root means, since those GRUs originally received all named old observations. Any reminder improvement in a competent named-history model would indicate limitations in its learned update/retrieval process; assigning the cause to a particular fiber requires independent neural probing and intervention controls.

A follow-up learned-state pilot should compare a GRU trained with intermediate exact correction targets to a compute-matched original objective, with all model seeds and source partitions fixed in advance. Add a deterministic 12-slot write/overwrite memory as a positive capability control, with its explicit inductive bias and extra memory counted. Do not promote the previous four development roots into a heldout confirmation sample. For causal probes, retain equal physical-dose shams and EvalMath's independent-root preflight.
