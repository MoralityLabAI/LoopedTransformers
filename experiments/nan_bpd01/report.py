"""Generate the separate research addendum from exact saved development receipts."""
import argparse
from fractions import Fraction
import json
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    runs=Path(a.runs);results=json.loads((runs/'graded/results.json').read_text());diagnostics=json.loads((runs/'analysis/diagnostics.json').read_text())
    geometry=json.loads((runs/'analysis/intervention_geometry.json').read_text());source=json.loads((runs/'cohort/source_receipt.json').read_text())
    fsi=json.loads((runs/'graded/fsi_contrasts.json').read_text());backends=results['backend_results']
    def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,row))+' |' for row in rows])
    taskrows=[]
    for arm in ['native1','native2','calibrated','open_loop','equal_compute','frequency','bare_native2']:
        for b in ['bf16','nf4']:
            s=backends[b]['task'][arm];taskrows.append([b.upper(),arm,s['n_cases'],f"{s['future12_correct']}/{s['n_cases']}",f"{s['top1_future_correct']}/{s['n_cases']}",f"{s['mean_source_rank']:.2f}"])
    memrows=[]
    timerows=[]
    for b in ['bf16','nf4']:
        m=diagnostics['backend'][b]['memory'];memrows.append([b.upper(),f"{m['elapsed_seconds']/60:.2f}"]+[f'{m[k]/2**30:.3f}' if m[k] is not None else 'unavailable' for k in ['cuda_allocated_bytes','cuda_reserved_bytes','sampled_process_dedicated_bytes','sampled_process_shared_bytes']])
        timerows.append([b.upper(),f"{m['cohort_elapsed_seconds']/60:.2f}",f"{m['prefix_repair_elapsed_seconds']/60:.2f}"])
    phaserows=[]
    channelrows=[]
    for b in ['bf16','nf4']:
        for site in ['decoder_output_pre_final_norm','loop_boundary_post_norm']:
            s=diagnostics['backend'][b]['phase'][site];phaserows.append([b.upper(),'raw' if site.startswith('decoder') else 'postnorm',f"{s['loop1_mean'][0]:.3f}",f"{s['loop2_mean'][0]:.3f}",f"{int(s['dev_last_token_loop_classification_correct'][0])}/16",'/'.join(f'{x:.3f}' for x in s['channel2279_mean_donor_recipient_absolute_gap_by_loop'])])
        raw=diagnostics['backend'][b]['phase']['decoder_output_pre_final_norm'];post=diagnostics['backend'][b]['phase']['loop_boundary_post_norm']
        for i,c in enumerate(raw['channels']):channelrows.append([b.upper(),c,f"{int(raw['dev_last_token_loop_classification_correct'][i])}/16",f"{int(post['dev_last_token_loop_classification_correct'][i])}/16"])
    editnames=['phase_global_suppress','phase_context_suppress','phase_context_replace','channel2278_global','channel2280_global','channel173_global','channel1531_global','random_dark_173','random_dark_1531','random_dense_dark_741','random_dense_dark_947','random_shared_dark_741','random_shared_dark_947']
    editrows=[]
    for name in editnames:
        for b in ['bf16','nf4']:
            rr=[r for r in geometry if r['backend']==b and r['arm']==name]
            editrows.append([b.upper(),name,f"{sum(r['eligible'] for r in rr)}/{len(rr)}",f"{sum(r['canonical_prefix_consistent'] for r in rr)}/{len(rr)}",f"{max(r['current_full_vocab_max_lp_error'] for r in rr):.4g}",f"{sum(r['delayed_lp_l2_per_physical_l2'] for r in rr)/len(rr):.3f}"])
    donorrow=[]
    for name in ['zero_write','full_donor_state','source_input','phase_context_suppress','phase_context_replace','random_shared_dark_741','random_shared_dark_947']:
        for b in ['bf16','nf4']:
            gs=[r['edits'][name]['donor']['future_correct'] for r in backends[b]['graded_cases'].values() if name in r['edits']]
            donorrow.append([b.upper(),name,f'{sum(gs)}/{len(gs)}'])
    alignment=diagnostics['cross_backend_rank_alignment'];alignrows=[]
    for arm in ['native1','native2','open_loop','equal_compute']:
        values=[r['spearman64'] for r in alignment if r['arm']==arm];alignrows.append([arm,f'{sum(values)/len(values):.4f}',f'{min(values):.4f}',f'{max(values):.4f}'])
    gate_text=[]
    for b,r in backends.items():
        gate_text.append(f"- {b.upper()}: native two-loop future@12 {r['task']['native2']['future12_correct']}/8; both-clean pairs {r['both_native_future12_pairs']}/4; both-top-one pairs {r['both_native_top1_pairs']}/4. Clean competence gate **{'PASS' if r['clean_competence_gate'] else 'FAIL'}**. Donor-state closure errors: {r['checkpoint_state_closure_errors']}; closure gate **{'PASS' if r['state_closure_gate'] else 'FAIL'}**.")
    powers=[]
    for b in ['bf16','nf4']:
        r=json.loads((runs/f'graded/{b}_power_result.json').read_text())
        try:d=json.loads(r['stdout']);powers.append(f"{b.upper()}: `{d['decision']}`, reasons `{d.get('reasons',[])}`")
        except Exception:powers.append(f"{b.upper()}: planner refusal, exit {r['exit_code']}; see raw receipt")
    min_holm=min(Fraction(r['conditional_holm_p_exact']) for r in fsi)
    rejections={}
    for row in source['attempts']:
        for k,n in row['rejections'].items():rejections[k]=rejections.get(k,0)+n
    eligibility=[]
    normrows=[]
    for b in ['bf16','nf4']:
        s=diagnostics['backend'][b]['physical_dose'];eligibility.append(f"{b.upper()}: {s['edits_measured']} measured edits, {s['ineligible']} dose-ineligible, {s['zero_dose']} non-sham zero-dose writes; {s['current_dark_edits']} current-dark writes, of which {s['canonical_current_dark_edits']} are prefix-coherent, with maximum current full-vocabulary LP error {s['current_dark_max_full_vocab_error']:.3g}.")
        normrows.append([b.upper()]+[f'{v:.6f}' for v in s['whole_state_norm_ratio_range']])
    text=f'''# NAN BPD 01 phase conditioned causal fibers development results

This separate Dark Fibers v015 addendum studies one pinned Nanbeige4.2-3B
checkpoint on four fresh development source roots, with two additional
calibration roots. Native two-loop future@12 is 3/8 for BF16 and 4/8 for NF4;
both fail the registered source-discriminating competence gate. The loop marker
is reproduced, while behavioral prerequisites for native BPD remain unmet.
The study distinguishes a reproducible loop marker, current-dark
continuous causal response, verified task competence and BPD donor transfer.
The proposed BPD primary is **unrun**, rather than a zero effect: native clean
competence must pass before dictionary fitting and behavioral mediation.
The original 168-case ARG-03 test remains sealed and is not regenerated or used
for exclusions. Historical v014/v015 text and ARG-03 criteria are unchanged.

![NAN BPD 01 phase, task, intervention, memory and ranking summary](../../experiments/nan_bpd01/receipts/nan_bpd01_summary.png)

## Clean task competence and controls

Each source pair shares inputs, query, topology, operator multiset and prompt
length, while its verified query targets differ. The model receives public
demonstrations only. Exact EvalMath search checks at most 12 of the original
64 programs. An independent geometric-series/convolution implementation also
verifies all twelve source worlds through degree 32; this is a finite integer
certificate, not an infinite-series identity proof.

The main prompt uses the snapshot's native chat template with thinking disabled.
Bare ARG-02 format is a predeclared source-only instrument control on the first
two recipient roots. Temperatures are fitted on CAL only before DEV grading.
Frequency scores use CAL-only per-position counts; equal compute averages
canonical and swapped-demo one-loop log scores. No model result selects cases.

{table(['Backend','Arm','Cases','Future correct at 12','Top one future correct','Mean source rank'],taskrows)}

Bare results use two recipient worlds, whereas the main controls use eight
worlds. Each world has sixteen candidate-prefix rows. Those rows, multiple
operators, tokens and backend executions do not increase the independent-root
count. The two backend executions are paired measurements of one checkpoint.

The frozen competence gate requires at least 4/8 native two-loop future@12
successes, at least two both-clean pairs and at least one pair whose two
distinct targets are both top-one correct. An input-blind constant-program
predictor cannot pass that last requirement.

{chr(10).join(gate_text)}

Source construction used {sum(r['source_trials'] for r in source['attempts'])}
base draws for six admitted pairs. Its rejection-event counters are
`{json.dumps(rejections,sort_keys=True)}`. These are structural candidate
comparison events, not independent causal trials. All four admitted DEV pairs
and all their model outcomes are retained; no model-based eligibility filtering
or replacement occurs. Target collisions among admitted pairs: zero.

## Phase marker and normalization

The predeclared site is decoder layer index 21, channel 2279. The original
[community observation](https://huggingface.co/Nanbeige/Nanbeige4.2-3B/discussions/20)
is a hypothesis; the values below are independent measurements from the pinned
checkpoint. The last-token measurement averages the sixteen candidate rows.
CAL-only nearest-mean loop classification is evaluated on sixteen DEV
world-loop observations, clustered in four roots.

{table(['Backend','Site','Loop one mean','Loop two mean','Loop classification','Mean pair gap loop one/two'],phaserows)}

The phase gap and paired source-content gap answer different questions.
Successful loop classification alone does not show source-specific reasoning
information. Neighbor/random channels, token-position profiles, whole-token
norms and raw/post-RMSNorm values are retained in the diagnostic receipts.
Backend similarity in these summaries does not establish identical mechanisms.

{table(['Backend','Predeclared channel','Raw loop classification','Postnorm loop classification'],channelrows)}

These control-channel classifications use the same CAL-only rule and DEV
world-loop denominator; they do not add independent units.

## Physical dose and delayed continuous response

A post-readout audit found a defect in the initial dense random controls:
different noise was assigned to candidate rows with the same prefix. Their
delayed operator-LP prefix spread is 1.47–3.08, whereas native and all other
initial arms have spread exactly zero. Initial dense behavioral scores are
retained as candidate-indexed diagnostics and invalidated as a canonical sham.
Corrected `random_shared_dark` arms broadcast the original seed's first noise
row across candidates. Only these controls are rerun on the same two recipients
per backend, requiring exact native-score repetition and zero hidden/LP prefix
error. The initial grades and CAL choices are independently checked unchanged.
This repair follows initial DEV exposure and is exploratory; see the
[frozen amendment](../../experiments/nan_bpd01/PREFIX_CONTROL_AMENDMENT.md).

Each nonzero edit targets 0.02 times the entire clean first-loop state's L2
norm, measured after BF16 rounding, with maximum relative mismatch 0.005.
Current-dark arms edit eight context positions and leave all three scored
positions byte-identical. Global diagnostic arms edit scored positions too.
The direction called suppression points toward zero; it is a calibrated-radius
edit, not necessarily complete removal of a channel.

{chr(10).join(eligibility)}

{table(['Backend','Edit','Dose eligible','Prefix coherent','Current full-vocab LP max error','Mean delayed LP L2 / physical L2'],editrows)}

Ineligible edits remain in the receipts and denominators; they cannot support
an equal-dose advantage. Suppression and CAL replacement may round to identical
physical writes and are not independent replications. The delayed norm sums
all full-vocabulary log-probability changes over the sixteen rows and three
positions. Its magnitude is a continuous diagnostic, not verified reasoning.

Full donor-state replacement is a positive state-closure control with its
natural displacement, not a dose-matched dark intervention. Source-input
uses the donor's actual public demonstrations. Their donor-target outcomes
and the clean baseline (zero write) are:

{table(['Backend','Arm','Donor future correct at 12 on two recipients'],donorrow)}

These two-root phase/control outcomes are distinct from the unrun BPD primary.
A donor target already present in a clean top-12 verifier search is not a new
causal recovery. Preserving the current readout plus moving future logits does
not establish a necessary reasoning component.

On these two intervention roots, both full-donor and source-input behavioral
controls recover 0/2 donor targets in each backend, despite exact state closure.
The clean donors themselves are unsuccessful there. NF4's CAL phase edit has
a 1/2 donor-target scoring event; it uses a common CAL loop mean, not that
pair's donor information. This event does not establish source-conditioned
transfer or rescue the failed behavioral instrument.

{table(['Backend','Minimum whole-state norm ratio','Maximum whole-state norm ratio'],normrows)}

Raw/post-normalization token norms and these edit norm ratios are audited.
Individual context-token radii are not held fixed; an equal physical dose can
therefore have different normalization effects across directions. The phase
diagnostics do not separate that contribution. A later mediation instrument
needs token-radius controls after establishing clean source competence.

## BF16 and NF4 memory and ranking geometry

The 166144 by 3072 BF16 output head stays on CPU. Vocabulary rows stream in
chunks of 8192 with stable full-vocabulary log-sum-exp. Operator logits and
the original sixteen-prefix/64-program scoring contract are preserved.
Before any 3B load, the exact-dimension synthetic GPU fixture gave operator
logit error 0, maximum logprob error 9.54e-6 and program-score error 1.53e-5.
The random-model native two-loop streaming closure error was 0.

{table(['Backend','Minutes including load','CUDA allocated GiB','CUDA reserved GiB','Sampled process dedicated GiB','Sampled process shared GiB'],memrows)}

Dedicated/shared entries are sampled Windows process allocations. CUDA reserved
and driver usage measure different quantities. Shared allocations are not a
measurement of paging, and completion is not proof of wholly dedicated residency.
The initial BF16 CIM observer was sandbox-blocked; the external permitted
observer supplies its driver figures. NF4 reads counters directly. Other user
applications remain active. Timings include verification, load and telemetry.
BF16 holds all directions eagerly; NF4 uses a byte-validated lazy iterator.
Their observed memory/runtime difference is not an isolated quantization effect.

{table(['Backend','Original cohort minutes','Additional prefix-repair minutes'],timerows)}

The memory table includes the maximum over both stages, and its elapsed time
includes the additional repair loads. NF4's original cohort itself used one
bounded resume for its four pending cases; no source cases were replaced.

{table(['Program-score arm','Mean BF16/NF4 Spearman rho','Minimum','Maximum'],alignrows)}

These correlations compare 64 correlated program scores within each world.
They do not measure BPD subspace alignment or certify causal fidelity under NF4.
Each backend has a distinct source and quantization manifest. Both share the
same source cohort, head contract, checkpoint revision and native-loop clock.

Native two-loop correctness pairing is
`{json.dumps({k:v for k,v in diagnostics['cross_backend_native2_correctness'].items() if k not in ['cases','scope']},sort_keys=True)}`.
The accuracy difference is descriptive. No cross-backend superiority or
equivalence test was registered or claimed.

## BPD engineering and blocked native stages

The actual BPD `TiedComponentLinear` API passes rank-8/16 synthetic two-pass
full-product-plus-residual closure below 5e-7. A GL shear changes a raw hard-mask
operator; canonicalized masked operators agree below 2e-15. Near-degenerate
singular components are grouped. The reached-state routing fixture changes
its mask after an intervention, unlike a static/clean-oracle mask. Synthetic
SVD/PCA/gradient/Gramian/random basis constructors pass orthogonality checks.
These are finite engineering checks, not learned Nanbeige components.

Native final-block o_proj/down_proj fitting, task-gradient and delayed-Gramian
selection, trained read/write factors, BPD donor writes, static/online learned
routing and cross-backend component/subspace alignment require the competence
and state-closure gates. Their exact statuses are in `receipts/results.json`.
Unrun BPD arms are unavailable, never imputed as behavioral failures or zeros.
The effective NF4 operator check uses actual quantization state rather than
the original BF16 matrix; its separate engineering receipt records residual
closure and kernel comparison.

## Statistical resolution and next cohort

Unit: source-generator world/root. Development: four roots/eight cases;
phase interventions: two roots/two recipients; calibration: two other roots.
Proposed primary: selected BPD donor-specific future@12 minus same-dose
random-dark, with scientific minimum 0.15. Physical relative dose: 0.02.
Primary comparisons: two backend-specific contrasts, both unrun.

The original exploratory allocation was 40; before correctness grading the
family expanded to 42 to include both frequency contrasts. After detecting
the prefix defect, four corrected backend/control contrasts bring actual
accounting to 46, retaining four invalid initial dense entries as p=1
placeholders. Corrected hypotheses are marked not prospectively registered.
All controls and unavailable entries are retained. FSI-01 computes
exact paired accounting, root bootstrap descriptions and conservative bounds.
It withholds inferential p-values because exchangeability is unestablished.
Conditional root-sign-flip sensitivity calculations use exact arithmetic and
Holm across 46; minimum adjusted value is `{min_holm}`. Their symmetry
assumption is unverified and none constitutes confirmation.

The strict pre-data planner refused an empty empirical pilot. Empirical
phase-versus-random preflights (not the unrun BPD contrast) report:

{chr(10).join('- '+s for s in powers)}

Those empirical preflights use **recipient** future@12 preservation under CAL
phase replacement versus corrected shared random A; their shams are corrected
shared random A/B
relative to the clean recipient. They do not estimate the donor-specific BPD
primary's power or treat the NF4 donor-target scoring event as confirmation.

The two-root binary mean has 0.5 pilot granularity. A 95th-percentile sham
threshold at 95% confidence cannot be calibrated from two roots; the EvalMath
skill requires at least 59 independent sham worlds for a sample-maximum bound.
Its hard stops include “Fewer than the planned independent roots” and “No sealed
held-out readout”. Evidence remains development-only and power-limited.

Before adding intervention complexity, establish source-dependent clean
competence on a repaired source-only task with independent CAL and DEV roots.
Then obtain at least eight independent development pilot roots, independent
sham calibration, and a genuinely untouched new confirmation readout sized by
the empirical planner and effect-floor bounds. The nominal 60-root plan is
not a power-certified sample size. The original 168-case test remains sealed.

## Reproducibility and preservation

Use the [experiment README](../../experiments/nan_bpd01/README.md) for exact
local commands and [source manifest](../../experiments/nan_bpd01/SOURCE_MANIFEST.json)
for repository/PR pins. Source registration: `18b102e289b340cd002bc96c3d310eb7ebbbc3aa`;
grading/storage lock: `5d21ddb52e0d792d618097c594e088a41cbfa26e`;
pre-grading resolution/multiplicity correction: `238e0d3`.
Checkpoint: `451ed48c3273ecef7ea8faaa43c31ce529763bb1`. Model execution uses
Transformers 4.46.3, PyTorch 2.6.0+cu124 and bitsandbytes 0.50.2.

Compact receipts, paired grades, phase profiles and the figure accompany this
addendum. Full model-only activation tapes stay local and are SHA256-indexed.
Existing PAPER.md/v014 text, historical result receipts and the frozen ARG-03
registration are untouched. See the [research context](../../experiments/nan_bpd01/RESEARCH_CONTEXT.md)
for how CQ/PD/ROC/SCM/LS-SAE evidence constrains the interpretation and the
[EvalMath mechanistic-power skill](https://github.com/MoralityLabAI/EvalMath/blob/accf4d9040bee8daeabc6b07003bb484708996c0/skills/evalmath-mechinterp-power/SKILL.md)
for the confirmation gate.

Fifteen CPU tests pass. The independent stdlib receipt audit verifies 228
diagnostic grade rows (212 prefix-coherent), all 212 original scalar grades
unchanged, unchanged CAL choices, 40 exact conditional root sign-flip
calculations and 46-entry Holm accounting. These are repeated measurements
within the four development roots, not 228 independent replications.
'''
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(text,encoding='utf-8');print(str(out))

if __name__=='__main__':main()
