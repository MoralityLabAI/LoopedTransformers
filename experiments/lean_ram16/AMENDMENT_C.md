# RAM-16 methodological amendment C: leak-free DEV loss, before model fitting

Dated 2026-11-16. This amendment is committed while the native TRAIN/DEV macro collector is running, **before any model fit, checkpoint selection or TEST source access**. It changes neither the first collection nor the registered hard admission gate.

The original RAM-08 old-source DEV includes a few theorem roots in RAM-16's newly selected TRAIN modules (`HookState`, `TOEGraph`, `Intervention`). Reusing those historical DEV observations directly to select the macro-trained models would leak original theorem roots across RAM-16's fit/selection boundary.

For **RAM-16 checkpoint selection only**, set aside whole original RAM-08 **TRAIN** theorem roots with:
`SHA256("RAM16-OLDTASK-VALIDATION-20261116/" + original_source_file + ":" + theorem) mod 6 == 0`, **provided the original source module is absent from every registered RAM-16 macro TRAIN or macro DEV module**. Drop all corresponding candidate observations from the optimizer's old-task TRAIN set. The remaining original RAM-08 TRAIN rows make up the old-task fitting pool, and the registered native RAM-16 source DEV positives make up a separate DEV macro diagnostic. The loss for seed/step selection is the recorded original four-label CE plus 0.2 SmoothL1 delta computed on the disjoint old-task holdout, with a second descriptive metric on original source-module DEV macros. Freeze source, theorem roots, candidate row counts and SHA for this deterministic partition before any optimizer update.

All original RAM-08 DEV rows remain **read-only historical diagnostics** and are excluded from both new optimizer and new checkpoint selection. RAM-16 module DEV and macro TRAIN are disjoint by complete file path. The full original RAM-08 TRAIN archive remains immutable, its original 20-label order is unchanged, and this method cannot use any RAM-16 TEST proof/goal state. Retain training split and seed receipts per model.
