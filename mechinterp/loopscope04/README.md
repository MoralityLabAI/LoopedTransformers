# LOOPSCOPE-04: supervised gradients versus recurrent activation updates

The existing 44,001-parameter TinyRecursiveModel architecture can be trained to map 81 random digits to a one-position cyclic copy of their input. We measure, at every one of its four tied-block passes, whether the hidden-state transition agrees with the negative **local supervised output-loss gradient**. An independently instrumented ledger tracks actual AdamW parameter updates during training. A third set of native interventions perturbs one token's recurrent activation and continues the exact original remaining tied-block operations. The Bayes abstention diagnostic applies BPD-OUT06's ternary probability-mass rule to the proposed recurrent update's **head-only** candidate-digit margin effect.

The precise preregistration, seeds, split hashes, model credits, fixed training budget and scientific acceptance criteria are in [PROTOCOL.md](PROTOCOL.md), committed before all LGA-01 results. The task and 44K model are an explicitly labeled independent *small supervised routing control*, distinct from the user's larger 7M TinyRecursiveModel Sudoku solver, Nanbeige looped LLM or SMC-27 original Sudoku checkpoint.

## Run locally

Requires PyTorch CPU, Python 3.11 or later:

    python -m unittest discover -s mechinterp/loopscope04/tests -p test_gradient_flow.py -v
    python -m mechinterp.loopscope04.gradient_flow --smoke --out runs/loopscope04/smoke
    python -m mechinterp.loopscope04.gradient_flow --out runs/loopscope04/full

The source-independent CI uses torch==2.7.1 CPU, two PyTorch threads, and uploads SHA256-stamped receipts. The smoke experiment trains one seed for ten steps; the prospective full campaign evaluates three paired seeds with 200 updates, two training objectives and four frozen checkpoints. Dataset x/y SHA256s and raw per-sequence metrics are included. A scientific FAIL or poor training admission is preserved as an outcome.

## Three different derivatives

For hidden activation h_k, the measured native proposed update is Δh_k=F(h_k)-h_k. The **local supervised loss gradient** is

    g_k = (softmax(head(h_k)) - onehot(true_label)) @ W_head.

The local diagnostic cosine is cos(Δh_k,-g_k), and its first-order expected decrease is -g_k·Δh_k. The actual CE decrease is L(head(h_k),y)-L(head(h_(k+1)),y). SGD/AdamW during training instead changes shared parameters θ: its separate gradient and update are compared as cos(Δθ,-∇θ L_train).

Head-null/visible activity is separated by the SVD rowspace of the learned 10x48 head weights, and a native heldout control inserts +Δ, -Δ or same-L2 random vector at one goal state before exactly the original remaining F blocks. Head-only Bayesian action decisions are explicitly secondary, recorded after a proposed step has been computed and before its correct test label is consulted for evaluation. The tool never claims such decisions save inference compute.

## Research boundaries

Positive local gradient alignment alone cannot establish that the model explicitly runs SGD internally, nor that its recursive dynamics globally minimize a potential function. A model can have aligned updates that make the head loss worse after nonlinear transition, or head-null updates that are useful after later mixing. The per-loop CE change, matched native interventions, task admission, seed variation and fixed-puzzle bootstrap help discriminate those possibilities.
