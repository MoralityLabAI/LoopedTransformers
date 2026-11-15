# AF-14 evaluation-layout repair

The frozen AF-14 scientific protocol is `cd7b7e283ea8508d34b2f9c1648060185b7a4231b4b3c3462ffbab7eba1fe65a`.
The initial committed source `9a565b7196e9a7dd99a220b2c6e91c7f2c0d16c3` passed ten local
unit/integration tests and trained seed 17 for all 160 registered updates per
arm. All three arms reached 1.0 development goal and joint-Q accuracy. The
heldout evaluator aborted before scoring its first new physical graph layout:
`AF-13 unregistered layout`. The inherited `admissible_status` and
`terminal_truth` helpers relied on AF-13's physical-layout map covering only
IDs 20–25, whereas AF-14's already frozen prospective layouts are 26–31.

The repair restores these two *evaluation-side* functions and `closure_bounds`
inside AF-14's own namespace, binding them to its original layout permutation
map. The AF-13 training loss, AF-14 bounded-progress margin objective and
weights, source checkpoint bytes, dataset, sampler, seeds, optimizer steps,
scientific thresholds and protocol are unchanged. An added regression validates
fresh-layout source truth against the independent Horn interpreter.

The initial seed-17 training checkpoint remains an unscored engineering-run
receipt. Full scoring must use a new training receipt under the repaired source
hash, with all models re-fitted from their original qualified AF-06 weights.
This is a source-provenance repair; its success does not imply scientific success.
The original failed CI runs before runner assignment also remain in GitHub.
