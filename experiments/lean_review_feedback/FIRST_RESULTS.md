# LRF-01 first verified result

[Run 45380937893](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45380937893), native-feedback job 138399461087, head bc94c93b2e8b937f70d2bb95e9aecbdac073114c, Lean 4.19.0 commit 6caaee842e94. Six unit tests passed. The frozen original corpus SHA256 is 7e15b7d055b49c829b901a43adfddb868e51fee5fc4293a9a313b970c9083d09 and compressed prediction Git blob is 9ba8352c169667d39de5a96f73af1d306862fd20.

All 384 historical traces independently rechecked. Native sample: 64, of which 61 originally solved and all 61 accepted, three incomplete censored. Validation controls: 128, 112 completed native proofs, 32 verified target rewards of 1, 80 deceptive submission rewards of 0, 16 censored incomplete. Zero false positive rewards or mismatches. 346 actual compiler calls, 42.939 seconds for the recorded replay. Gate PASS. No model trained and no original score changed. The independent type-contract check tests definitional equality/application of the fixed structured task, not arbitrary prose equivalence.

[Native receipts artifact 14912639273](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45380937893/artifacts/14912639273), ZIP SHA256 eb2057acdf42eb49af21b9435cf8247bf69c41e025a46c2acaf54eedbec288d6.

Next: LRF-02 prospective proof generator and learned critic, with the native gate as adjudicator, matched controls, and fresh proof families. Measure actual native calls and certified success; no neural score can override the gate.
