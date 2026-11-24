# OLR-07A library preflight: invalid human audit

Native workflow [45327039231](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45327039231) completed on e5eec16. Scientific validity FAIL; no policy outcomes scored, no learned fit. Artifact [14890402416](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45327039231/artifacts/14890402416). Independently verified all 89 manifest-listed files in downloaded archive SHA256 1bd5225cfef801941bc29c5466e7c6223c69a68e53a46737d711534cfb532814.

| Role | Selected | Scored | Human certified |
| --- | ---: | ---: | ---: |
| TRAIN | 101 | 82 | 82 |
| Calibration | 21 | 17 | 17 |
| Evaluation | 85 | 75 | 68 |

All source file elaboration error counts zero. The 7 evaluation human replay misses: four Array.Basic termination wrappers beginning decreasing_by (parser expects a tactic), three heartbeat failures (Array.toList_filterMap', Array.all_iff_forall, Fin.coe_sub_iff_le). Original evaluation validity is 90.67%, below registered 95%. No retrospective removal or headroom claim. New 07B selects fresh source files, excludes unsupported wrapper syntax before freezing, and uses a separately budgeted human audit. Candidate heartbeat cap stays 500. The old archive is used only to exclude exact goal text and declaration identities.
