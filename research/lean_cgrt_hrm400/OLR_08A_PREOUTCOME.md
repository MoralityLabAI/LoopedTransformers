# OLR-08A pre-outcome registration

Written and committed before any OLR-08A menu, search or human-replay outcome is collected.
Protocol: [olr08/RUN.md](olr08/RUN.md) and [olr08/run08.py](olr08/run08.py), unchanged.

## Package

OLR08_CPU_gym.zip, 210361 bytes, SHA-256
e5e4b26af9837a828b743a9479e396621fbec2d6f3e846711a02c639d53750cb, equal to the digest
registered in [OLR_08A_CPU_HANDOFF.md](OLR_08A_CPU_HANDOFF.md). All 14 files match
package_manifest.json and no unlisted file is present. The ZIP and its unpacked files are
committed under [olr08/](olr08/). A `-text` attribute keeps the package's CRLF bytes, so the
committed blobs hash to the manifest values on every checkout.

## Native build and sentinel

`lake build` of olr08/lean with Lean 4.19.0 (commit 6caaee84, x86_64-w64-windows-gnu)
succeeded with no source change. lgym.exe SHA-256
61923c657261feb458c60c7049f83cc0f2c121ce8dd3dcad09b039f8d3c5dede.

`run08.sentinel` on that binary:

| Probe | Step | Certificate |
| --- | --- | --- |
| exact ScopeSentinel.earlier n | closes | kernel true, no axioms |
| exact ScopeSentinel.current n | closes through the recursion variable | kernel false, self-reference |
| exact ScopeSentinel.later n | unknown identifier | none |
| simp only [ScopeSentinel.current] | closes | kernel false, self-reference |

The graph and premise endpoints answered and the legacy post-command candidate endpoint
refused the request.

## Roster audit against all spent cohorts

The fresh roster in run08.py has 5 calibration files (Int, Array, Fin) and 27 evaluation
files (BitVec, Array, Vector, UInt). Audited with
[audit/olr08_spent_cohorts.py](audit/olr08_spent_cohorts.py) against the 69 spent files in
[results/olr08_spent/spent_cohorts.json](results/olr08_spent/spent_cohorts.json):

| Role | Files | Spent overlap | SInt | New file, seen family | New family |
| --- | --- | --- | --- | --- | --- |
| Calibration | 5 | 0 | 0 | 5 | 0 |
| Evaluation | 27 | 0 | 0 | 22 | 5 (BitVec) |

The package's spent.json lists 62 files. The 7 others in the inventory are files declared in the
07B roster that contributed no selected root: Nat/Bitwise, Nat/Div, List/Nat, Int/Bitwise,
Int/DivMod, Fin/Bitwise, Fin/Log2. None is in the fresh roster. The exact 07B registry also has 2
goals and 2 declarations (Array.map_fst_zip, Array.map_uncurry_zip_eq_zipWith) absent from
spent.json. The frozen fresh registry is checked against the full inventory before outcome
collection; the result is recorded below.

Array, Vector, Fin, UInt and Int results are new-file evidence within families already tested.
Only BitVec is new-family evidence. SInt stays sealed for the separately registered 500K gate.

## Pre-registered stop before fresh evaluation

run08.py collects fresh-evaluation outcomes for the fixed controls even when no reader was
fitted. RUN.md says a failed gate stops the run. Fresh files spent on controls cannot be
recovered, while a stop can be resumed from the checkpoints. So the run uses
[olr08/drivers/run_full.py](olr08/drivers/run_full.py), which calls the package's launch.main
unchanged and stops before the first fresh-evaluation stage if no reader was fitted. That
covers a failed source-validity, headroom or source-frame gate. Resuming launch.py on the same
output directory continues under the same contract.

## Resources and layout

8 threads, 2 native workers, 12 GiB Windows Job Object process-tree commit limit, Torch 2.5.1
CPU, Python 3.12.10. The run directory is on C:, because other jobs keep the D: volume near
full. The pinned Lean source is D:\lean4-419-source at 6caaee84 with a clean tracked tree.
Reissue prerequisites are the committed, custody-authenticated artifacts in
[results/olr07_reissue/](results/olr07_reissue/).
