# Nanbeige local gym reports

- [Completed native BF16/NF4 pilot and recurrence geometry](NAN_GYM05_NATIVE_REPORT.md)
- [Native grades, SAE fits and audit progress](NAN_GYM05_PROGRESS.md)
- [Actual local Prime model rollouts](NAN_PRIME06_PROGRESS.md)
- [Fresh-source VPD/BPD fitting pilot](NAN_FIT07_PROGRESS.md)
- [Resident-head 1K/2K/4K context challenge](NAN_CTX08_PROGRESS.md)
- [Full objective and remaining requirements](ROADMAP.md)
- [Completed native BPD recomputation fit](NAN_RECOMP09_PROGRESS.md)
- [Learned VPD/BPD comparison and specificity limits](NAN_LEARNED_VPD_BPD_COMPARISON.md)
- [Official long-context Prime evaluation](NAN_PRIME_CTX10_PROGRESS.md)
- [Measured cached versus full-prefix execution](NAN_KV11_PROGRESS.md)
- [Explicit-effective NF4 random engineering and next registration](NAN_EFFECTIVE_NF4_ENGINEERING.md)
- [Six missing rank/site pipelines and native o_proj finite-write bridge](NAN_RANKSITE13_ENGINEERING.md)
- [Frozen six-cell pretrained rank/site extension](NAN_RANKSITE13_PROGRESS.md)

The eight SAE fits and weight-only SVD baselines are separate from trained
VPD/BPD. BF16 VPD and native two-pass BPD have completed genuine eight-step
fits and independent audits; NF4 failed its fixed fitting screen. Static-mean
and learned controls retain the same small-DEV accuracy, so donor-specific
causal discrimination remains unproven. Complete-reply
Prime scores are separate from native prefix-stop scores. The original
168-case ARG-03 test remains sealed.

The bounded local report stream watches the real fitting supervisor handle and
new baseline/receipt/probe files. Completed primary Prime runs receive the
independent tokenizer/state/SDK audit; fresh-source fits receive independent
source, loss, checkpoint, controller and diagnostic checks. Only four named
aggregate report files can be committed and pushed by that stream. It cannot
stage raw runs, change frozen protocols or launch another decoder. Its finite
monitoring budget leaves GPU workers untouched on expiry or a reporting error.
A failed postprocessing stage is held back while other independent report
stages continue; a failed final Prime audit cannot suppress valid fitting
milestones. Failed SDK episodes retain missing graded coverage. The raw saved
assistant message is bound exactly; the SDK's separate outer-whitespace-trimmed
`last_reply` accessor is recorded without changing any model reply or grade.
