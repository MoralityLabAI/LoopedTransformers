# RMP-MG-01: usable computation and observation-null causal effects

Date: 2026-11-11

Status: mathematical derivations and constructed controls. Trained-model evaluation is pending. The linear algebra and Taylor arguments below use standard tools; no claim of priority is made.

## Motivation and source boundaries

The uploaded Systems Philosophy transcript raises concerns about reading reasoning traces and about incentives to conceal reasoning, especially at 37:04–39:44. It supplies a motivation for an experiment. It supplies no numerical evidence for the measurements proposed here.

Relevant external research:

- Xu et al., *A Theory of Usable Information Under Computational Constraints* (ICLR 2020), https://arxiv.org/abs/2002.10689. Predictive V-information accounts for an observer's computational restrictions.
- Baker et al., *Monitoring Reasoning Models for Misbehavior and the Risks of Promoting Obfuscation* (2025), https://arxiv.org/abs/2503.11926. Direct optimization against CoT monitors can produce obfuscated reward hacking under the studied conditions.
- Geiping et al., *Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach* (2025), https://arxiv.org/abs/2502.05171. The architecture iterates a recurrent block within token generation.
- OpenAI, *What affects chain-of-thought monitorability across the model lifecycle?* (2026-11-10), https://alignment.openai.com/cot-monitorability-model-lifecycle/. No-CoT capability and CoT controllability are studied as imperfect proxies. Appendix B reports an equal-capability pair for which controllability does not order evasion success.

The new assay below is our proposed experiment. Claims about this assay should be tied to its own receipts.

## 1. The previous conditional-information statistic

Let the recurrence be h_(t+1)=F(h_t,x), with fixed parameters and deterministic evaluation. Let Z be any target random variable. Condition on both the complete previous state and every input to the transition:

    I(h_(t+1); Z | h_t, x) = 0.

Proof: h_(t+1) is a measurable function of the conditioning variables, so its conditional distribution is a point mass. Its conditional mutual information with any other variable is zero. This argument works for discrete variables and, with the usual regular conditional distributions, for continuous state variables. It avoids subtracting potentially undefined differential entropies.

Writing only I(h_(t+1);Z|h_t) leaves fresh access to x unconditioned. The LoopedChase implementation reinjects the input embedding at each visit. A positive quantity can therefore describe access to x. It does not isolate reasoning progress. Independent stochasticity does not change the zero result after conditioning on the transition inputs under the corresponding conditional-independence assumption.

The complete state can be written S_t=(h_t,x). Data processing gives I(Z;S_(t+1)) <= I(Z;S_t). Computation can still make a target easier for a restricted decoder to recover.

## 2. A bounded-readout quantity

Fix a predictive family Q, a loss, and a target Z. With log loss, define

    R_Q(Z | V) = inf_{q in Q} E[-log q(Z|V)].
    U_Q(t;K) = R_Q(Z_K | V_0) - R_Q(Z_K | V_t).

For pointer chasing, Z_K=P^K(s) stays fixed for every compared visit t. Replacing the target with P^t(s) changes the problem at every visit and cannot establish progress on one target.

Use V_t=[Embed(x); h_t] and V_0=[Embed(x); 0], with a fixed feature layout. The native model's initial carried state is zero, so comparing h_t only with h_0 would confound input ingestion with additional useful computation. Include an embedding-only baseline and disclose the cost of constructing each feature representation.

Empirical cross-validated log loss estimates a particular fitted observer's risk. Optimization error and finite samples separate this estimate from the population infimum. Report the estimates as bounded-probe risk changes. Estimating ordinary Shannon information or proving an exact V-information value requires additional assumptions.

A four-point control demonstrates the distinction. Let x1,x2 be independent uniform signs and Z=1[x1*x2=1]. An affine logistic predictor of (x1,x2) has optimal log loss log(2), by symmetry and convexity. After exposing x1*x2 as a coordinate, the fixed logit 8*x1*x2 has log loss log(1+exp(-8)). Both representations determine Z completely for an unrestricted observer. The calculation validates the choice of quantity.

## 3. A local causal test

Fix an input, an observed trajectory prefix, a visit t, and a continuation horizon k. Let g(h) be the explicitly declared observation channel at the intervention time. Let f(h) be a smooth vector of future task-output logits after k native recurrent visits.

At a reference state h, write

    C = Dg(h),       B = Df(h).
    beta = || B P_ker(C) ||_2.

P_ker(C) is the Euclidean orthogonal projection onto the kernel of C. A unit vector v attaining the operator norm obeys Cv=0 and ||Bv||=beta.

### Proposition: rank criterion

The following are equivalent:

    beta > 0;
    there exists v with Cv=0 and Bv != 0;
    ker(C) is not contained in ker(B);
    rank([C;B]) > rank(C).

Proof: the projected operator is nonzero exactly when B acts nontrivially on ker(C). The last equivalence follows by taking orthogonal complements of the kernels and comparing row spaces.

The implementation uses a thin singular-value decomposition of C and removes its row-space component from B. It avoids forming a state-width-squared projection matrix. Report the numerical rank tolerance and the actual residual ||Cv||. An exact rank statement and a floating-point rank decision carry different evidential force.

A positive beta gives a local causal effect missed by this observation channel to first order. It is compatible with random untrained networks. It does not establish a learned reasoning algorithm, an intention to conceal, or a failure of every possible monitor.

### Proposition: finite-radius bounds

Assume g and f are twice differentiable along h+a*v for 0<=a<=r. Suppose their second directional derivatives have norm bounds M_g and M_f throughout that segment. Then

    ||g(h+r*v)-g(h)|| <= r*||Cv|| + (M_g/2)*r^2;
    ||f(h+r*v)-f(h)|| >= r*||Bv|| - (M_f/2)*r^2.

The lower bound can be clipped at zero. For an exact observation-null maximizing direction, these become

    observation drift <= (M_g/2)*r^2;
    future drift >= r*beta - (M_f/2)*r^2.

Proof: apply Taylor's theorem with integral remainder to each map. The upper estimate uses the triangle inequality. The lower estimate uses its reverse.

The linear/quadratic separation suggests a finite-radius diagnostic. Measure both drifts across a preregistered radius grid and both signs. Empirical slopes supply evidence about the sampled radii. A finite sweep does not certify a Hessian bound over a continuous segment.

For f(x,y)=y and g(x,y)=x+y^2 at the origin, v=(0,1), beta=1, M_g=2, and M_f=0. A radius of 0.01 gives observation drift 0.0001 and future drift 0.01. This is an exact constructed control.

The observation map must be named in every result. A native answer head, a generated explanation, and a full-history safety monitor are separate channels. A monitor of the full history has access to information absent from a single snapshot. Prefix observations fixed before an intervention remain fixed; newly generated later observations require their own measurement.

## 4. From observation laws to monitor limits

For balanced alternatives b in {0,1}, let mu_b be the distribution of everything the monitor sees, including its prompt, transcript, side information, and any relevant timing. Its optimal classification accuracy is

    Acc* = (1 + TV(mu_0,mu_1))/2.

Here TV is one half the L1 distance for a finite observation space. Proof: summing the larger of the two equal-prior probability masses gives the formula. Identical observation laws imply accuracy 1/2.

A small Euclidean distance between two deterministic vectors alone provides no total-variation bound: distinct point masses have TV=1 regardless of their proximity. Connecting the Taylor result to a monitor limit needs a specified noise channel, discretization, monitor Lipschitz bound, or an actual measured monitor response. The code keeps the claims separate.

## 5. Scaling and scope

Beta depends on the coordinates and units of the state and outputs. Euclidean orthogonal changes of state coordinates preserve it; general invertible reparameterizations need not. Report actual finite perturbation size, state norm, output scale, and numerical precision. Cross-model comparisons require a fixed normalization or a metric defined from a development set.

LayerNorm gauge directions offer an essential control in LoopedChase. The model normalizes the carried state before reinjecting the prompt. A constant shift of each token across its feature coordinates can disappear at that operation, producing near-zero future effect as well as negligible readout changes. A monitor-blind direction with zero task effect contributes no evidence of hidden causal computation.

Finite perturbations can leave the distribution of natural states. Matched natural donor interventions and oracle-aligned semantic edits are required before interpreting a local sensitivity direction as a semantic variable. An arbitrary output change can represent damage to the computation.


## 6. A graph split that survives node renaming

For a permutation P, let c(P) be the sorted multiset of its disjoint cycle lengths. For a node relabeling sigma,

    P' = sigma P sigma^{-1}.

Then c(P')=c(P), because sigma sends each P-cycle to a P'-cycle of the same length. Conversely, two permutations with the same cycle-length multiset admit a bijection matching corresponding cycles and preserving cyclic order. That bijection conjugates one permutation to the other. Thus

    c(P)=c(Q)  iff  Q=sigma P sigma^{-1} for some sigma.

Assigning each c(P) to one dataset split excludes all renamed copies from the other splits. Different start nodes and edge presentation orders remain within that split. A labelled edge-list hash alone lacks this guarantee.

This restriction changes the statistical question: testing on held-out cycle types measures structural generalization of a fitted observer. Results are conditional on the actor, which may have seen those graph structures during its own training. The implemented paired estimator averages each structural group's loss difference before averaging groups. Its bootstrap samples groups, so repeated prompts cannot inflate the reported independent-group count.

## 7. Widening a fixed observation channel

Suppose C1 is the Jacobian of one observation channel and C2 stacks C1 with more observed coordinates. Then ker(C2) is a subset of ker(C1). For the same future Jacobian B and the same Euclidean state metric,

    ||B P_ker(C2)||_2 <= ||B P_ker(C1)||_2.

Proof: the operator norm is the supremum of ||Bv|| over unit vectors in the respective kernel. The second supremum is taken over a subset of the first domain. The zero-dimensional kernel gives norm zero.

This yields a controlled numerical check when comparing cursor-only logits with logits from every token. Both assays must use the same actor, inputs, state, and continuation horizon. Floating-point rank choices need their own residual and tolerance checks. The local smoke satisfies the inequality in all 12 paired cases; its untrained actor supplies an implementation control.
