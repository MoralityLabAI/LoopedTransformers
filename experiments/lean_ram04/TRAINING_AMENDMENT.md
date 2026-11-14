# Argument-order amendment before fitting

During implementation review, plain mean token pooling was found to discard
argument order: `exact h hp` and `exact hp h` have the same token multiset.
Before fitting or observing prospective goals, RAM-04 adds fixed sinusoidal
position gates to token pooling. These preserve order information without new
trainable parameters. The same pooling applies to every architecture and the
identical CPU/CUDA calibration. A unit test checks that swapping arguments
changes the pooled representation. Candidate pools, labels, source split,
budgets, optimization and development selection remain unchanged.

This is an explicit pre-fitting implementation revision after collection began,
not a claim that every implementation byte was frozen at protocol registration.
The fitted model/search revision and selected weight hashes must be committed
before any prospective goal or reference proof is inspected.
