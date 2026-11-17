# AF-30 constructive certificate: a blocked-premise path cost solves every positive source grammar

**Scope:** Exact theorem about the source-registered AF-30 grammar families, not a general theorem for Horn entailment or learned neural proof search. This is a nonlearned control and the reason AF-30's registered learned advantage fails.

## Setup

Let each rule be a finite positive Horn implication \`premises -> head\`. The known root facts are \`{0,1,2,3}\`; the target is atom 14. A productive path begins with \`(r,s)->4\` for registered known roots \`r,s\` and continues through the unary chain

- depth 5: \`4 -> 5 -> 9 -> 11 -> 14\`
- depth 6: \`4 -> 5 -> 9 -> 11 -> 6 -> 14\`

Two distracting, initially enabled rules derive atoms 7 and 12. Their apparent paths to the goal consist of \`(7,10)->13\`, \`(12,10)->13\` and \`13->14\`. No rule concludes atom 10 and atom 10 is absent from the known roots. Additional source rules conclude only atoms 7,8,12 or, for depth 5, atom 6; none provides a new route to atom 14.

Each reachable source has a source-registered middle rule premise swap location \`j in {1,2,3}\`; its paired unreachable source replaces productive premise 4, 5, or 9 by atom 10, and places that productive premise on an otherwise irrelevant decoy rule. Every physical graph is an arbitrary fixed relabeling of the atoms and rule slots within these constraints.

Define \`P\` as the public set of rule heads (atoms with at least one producer). For a current proven set \`S\`, a reverse goal cost \`C_lambda(v)\` starts with \`C_lambda(14)=0\`. If a rule has premises \`Q\` and head \`h\` with finite cost, each \`p in Q\` receives a candidate

\`\`\`
C_lambda(p) <= C_lambda(h) + 1
               + lambda * |{q in Q \ {p} : q not in S and q not in P}|
\`\`\`

Take the least cost over these reverse relaxations. An atom with no reverse path has infinite cost. The implemented extractor uses penalty \`lambda=7\`, a finite missing-value sentinel \`30\` and a bounded number of exact backward relaxation sweeps.

The PATH_CERT policy selects, among currently legal rules, the rule whose **head** has minimal reverse cost, breaking ties by physical slot. A once-per-root public Horn fixed-point certificate selects STOP exactly when the goal is absent from the least fixed point. Both reverse and forward symbolic work are charged in the AF-30 evidence.

## Proposition 1: Correct positive-proof selection

For every registered reachable AF-30 grammar, at both depths and for every bijective physical node/rule relabeling, PATH_CERT with \`lambda > 3\` selects the productive rule on each step and finishes the goal in the exact shortest 5 or 6 firings.

### Proof

Because the productive chain is unary after the initial root-fact rule, each productive head's reverse cost is its number of remaining unary implications to reach atom 14. The initially legal productive rule concludes atom 4, whose cost is 4 at depth 5 and 5 at depth 6. After each productive firing, the next productive rule's head has a strictly smaller cost.

The first enabled distraction concludes atom 7 and the second concludes atom 12. Every route from either head to the goal uses a rule with atom 10 as an additional premise, followed by \`13->14\`. Atom 10 belongs neither to the initial known set nor to the set of producer heads. Thus each distracting head has reverse cost \`2 + lambda\` under the registered rules. For \`lambda > 3\`, this exceeds the productive head cost at the root, even at the longest registered depth; the gap only increases after each productive step.

The other legal decoy heads, including atom 8, have no reverse goal path and consequently rank below the finite productive path. Source-position-specific decoy changes preserve that fact. Therefore every chosen rule is productive. Exactly one new productive chain fact becomes known per firing, so the goal is reached after the registered 5 or 6 firings.

The argument depends only on graph incidence, initial facts and goal marker, which are invariant under simultaneous bijective relabeling of physical atoms and rule slots. The result follows for every source physical permutation. QED.

## Proposition 2: Correct STOP on paired negative sources

The exact public Horn least fixed-point certificate stops on every registered unreachable paired source.

### Proof

The negative member's productive path contains a required premise atom 10 at a middle rule. No rule concludes atom 10 and it is absent from the initial root facts. Thus the productive chain cannot proceed past that rule. The alternative apparent goal corridor concludes atom 13 only from a rule whose premises also contain atom 10. Consequently no derivation of atom 14 exists. An exact public fixed-point closure therefore excludes atom 14 and the certified controller immediately stops. QED.

## Interpretation

The theorem explains the exact scored result: the nonlearned PATH_CERT controller completes **768/768 reachable cases at each held depth across eight source fit seeds**, with **768/768 correct initial negative STOP**, matching the learned RANK_CERT controller. Its evidence depends on source-registered *acyclic producerless bottlenecks* and should not be extended to arbitrary graphs whose unsupported prerequisites have producers inside an unreachable strongly connected component.

An adversarial grammar in which every misleading prerequisite has a producer but belongs to an unsupported cycle removes the producerless-atom shortcut in this theorem. Any subsequent learned-ranker experiment on such graphs must compare against an equally adapted symbolic controller and exact public closure, with the new source distribution frozen before model fitting. The present AF-30 theorem is a source-family certificate, with CI-backed exhaustive confirmation, rather than an unverified generalization to unrestricted Horn systems.
