# NAN-WINDOW-66 delayed readout localization

INVERSE65 identified layers 11–17 as the interval where the sampled
counterfactual answer contrast grows. This instrument captures every
layer from 11 through 17 plus layer 21 in both passes of the genuine
last-valid-token B2 prefill. Layer indices are zero-based. Each layer
records the incoming residual, attention write, pre-MLP residual, MLP
read and write, and pre-patch outgoing residual. Reconstruct both native
BF16 additions and check adjacent-layer residual continuity.

The edit remains at the last token of the pass-1 final layer. The existing
SHAM56 and ROUTE58 checks retain the native competence, physical dose,
sham, and all-vocabulary phase-1 preservation gates. The window adapter
consumes each of their existing four signed continuations and the single
duplicate baseline. It checks the unedited companion in both passes and
the edited row's physical operands before the boundary patch. A failed
gate retains its original refusal with zero delayed window outcomes.

Use the same two bare label rows and final RMSNorm at every sampled site
as a lens. At each second-pass layer, evaluate the actual edited row
against its duplicate baseline. PREMLP64 decomposes that readout into
incoming, attention, MLP and measured residual-add rounding terms
through 32 BF16 coalitions. The continuous decomposition also retains
RMS scaling and head rounding. Normalize positive-minus-negative effects
using the realized dose, then subtract the matched sham effect separately
at every layer. The separate native full-vocabulary final score remains
the endpoint. Verify its selected logits against the CPU replay at the
registered 0.125 tolerance, with the mismatch carried through the
independent score. Treat this depth profile as a readout allocation of
the measured edit effect. It does not isolate a unique mediating path.

Commission with the snapshot's real decoder code under random BF16 CPU
weights. The fixture uses a short window across resident and reused
staged layers, multiple B2 chunk positions, and the existing live minimax
signed continuation. This tests engineering closure at a fixture-specific
dose. The production window and unchanged ROUTE58 GPU dose must be frozen
under a separately admitted lane before any pretrained run. The current
workers retain their original registrations.
