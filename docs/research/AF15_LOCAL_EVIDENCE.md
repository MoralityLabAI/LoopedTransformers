# AF-15 local evidence inventory

**Status:** SHA-256-checked local archive, scientific result **FAIL**, local engineering **PASS**; hosted Python 3.12/Torch 2.5.1 and Lean 4.19.0 checks remain pending.

- Archive basename: `AF15_local_evidence_20261115.zip`
- Archive bytes: `525782`
- Archive SHA-256: `44541a8541f2e20b91c8713303565c64a5a7f41aa78796af999cdc056e259ff0`
- Archive members: **33**; ZIP integrity test passed.
- Source commit: `eedb938b7d8025801a7b107693ad7fb63b723331`.
- Exact protocol SHA-256: `4b01129a6afc0c1a96975a939210128c421bd659525c3a5764c403a1f3f0b90f`.
- Independently recomputed aggregate `results/results.json` SHA-256: `61168b2f564d6b0f3e58a43c6b0030051df4408a298b96035164b4246bb43677`.
- Each local evidence member has its own SHA-256 in the archived `manifest.json`; this page documents the archive and primary receipts. The complete archive is available as a downloadable attachment in the AF-15 research conversation. **This Markdown inventory is not a GitHub-hosted binary release.**

## Scored evidence

| Receipt | Rows | SHA-256 |
|---|---:|---|
| Seed-17 episodes | 3280 | `cea8c8bdcdb06f84c316eb20e20d353cb5d1c420ad06679d31eb2859dd405dc2` |
| Seed-29 episodes | 3280 | `41f9e593ec8492c6ed15004131101d67fee0887be7f7fdc87101b6a848d76ef1` |
| Seed-17 independent audit | 44 forward replays | `712dd5179384d3bf4ae0687252a7c034d6277430385d7de00ba36dd820317d4c` |
| Seed-29 independent audit | 47 forward replays | `90ceb16a96ab54e6dddcec8944f9f2c1f4243d0156ac9ac926ab59b160a8c0db` |

## Frozen model weights

| Model | Torch .pt SHA-256 | Named NumPy .npz SHA-256 |
|---|---|---|
| seed17 imitation | `f2df2ac46c6e035e325b49d03af5364288b03e03608d1f4624911ac06cba63f7` | `6f98d565aec5fb3cb00eda1690e572dcf9d23dde3f27983a6db3fc8a0c16a40e` |
| seed17 frontier | `f3ac8e2c498a717472405ac295ac68916b7237bc68792734dc46db1ea301afa0` | `2e7d02385af1265a408e348b90662317695810a3eae26017047f34e23222e200` |
| seed29 imitation | `ed84d38428956950df5dbe813d78f18fc389f6c7c92c72f9b357bf04101b2ab0` | `88a12bf602a9bb7c05b816957743643b995cb0caeb28af3cd2f8065b1665a3ef` |
| seed29 frontier | `0dc5365d90009fe34b9b16bf45f877eeeff0ecb2efc07a7f1861f46446fa9d4e` | `f0e756667977b650f544eab4eca1c73c518c6d8b3c5d3d2b1e3aaf0d57bfc2d6` |

## Verification boundaries

The independent semantic verifier replays accepted Horn-rule firings against public premises, current proved atoms and goals. It distinguishes a legal local action from a solved reachable goal within budget. Every accepted transition was checked; zero false accepted proofs occurred because the host enforces the rule gate. Torch checkpoint and named-array equivalence plus sampled forward replay validate the scored artifacts.

Python 3.13 / Torch 2.10 CPU was used for this local experiment. GitHub Actions [run #43701257848](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43701257848) failed before preflight runner execution on the original attempt and subsequent retries. The planned Lean `RuleSoundness.lean` file has not yet passed hosted Lean 4.19.0 checking. The experiment remains a draft PR and is not approved for merging based only on this local archive.
