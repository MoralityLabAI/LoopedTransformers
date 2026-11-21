"""Every original accuracy/geometry stratum, under separate stop provenance."""
import csv
import io
import json
from pathlib import Path
from .common import ROOT, RUNS, OLD, OLD_RUNS, read, rows, sha, write, Guard, verify, no_torch
from .audit import run
from experiments.nan_ctx21.report import groups, geometry

OUT = Path('reports/nanbeige_bpd_gym')
STEM = 'nan_ctx21_resource29_results'
OUTPUTS = (str(OUT/'NAN_CTX21_RESOURCE29_RESULTS.md'), str(OUT/(STEM+'.json')),
           *(str(OUT/(STEM+'_'+name+'.csv')) for name in ('accuracy','paired','geometry')))
METRICS = ('loop1_relative_l2', 'loop1_cosine', 'loop2_relative_l2', 'loop2_cosine',
           'recurrence_displacement_relative_l2', 'recurrence_displacement_cosine')


def number(value):
    return '—' if value is None else f'{value:.9g}'


def table(headers, values):
    return ['| '+' | '.join(headers)+' |', '| '+' | '.join('---' for _ in headers)+' |',
            *['| '+' | '.join(str(x) for x in row)+' |' for row in values]]


def csv_file(path, values):
    stream = io.StringIO(newline=''); writer = csv.DictWriter(stream, fieldnames=list(values[0]))
    writer.writeheader(); writer.writerows(values)
    path.write_text(stream.getvalue(), encoding='utf-8', newline='')


def presentation(payload):
    OUT.mkdir(exist_ok=True)
    accuracy = [dict(backend=b, **g) for b,v in payload['backends'].items() for g in v['groups']]
    paired = payload['paired_physical_geometry']['paired_correctness']
    physical = payload['paired_physical_geometry']['groups']
    scalar = []
    for group in physical:
        for metric in METRICS:
            scalar.append({k:v for k,v in group.items() if k not in METRICS} | dict(metric=metric, **group[metric]))
    assert len(accuracy)==96 and len(paired)==48 and len(physical)==96 and len(scalar)==576
    # Lists stay exact JSON in CSV cells; null scalar fields are empty, never zero-filled.
    csv_file(Path(OUTPUTS[2]), [{k:json.dumps(v) if isinstance(v,list) else v for k,v in g.items()} for g in accuracy])
    csv_file(Path(OUTPUTS[3]), paired); csv_file(Path(OUTPUTS[4]), scalar)
    Path(OUTPUTS[1]).write_text(json.dumps(payload, indent=2, allow_nan=False)+'\n', encoding='utf-8', newline='\n')
    dev = {b:[g for g in v['groups'] if g['split']=='development' and g['context_budget']==4096 and g['loop']==2 and g['arm']=='main']
           for b,v in payload['backends'].items()}
    counts = {b:(sum(g['correct_worlds'] for g in gs),sum(g['observed_contexts'] for g in gs)) for b,gs in dev.items()}
    resource = payload['audit_resources']
    lines = ['# CTX21: separate audit after the BF16 RAM stop','',
        f"At 4K, loop-2 development main accuracy is BF16 **{counts['bf16'][0]}/{counts['bf16'][1]}** and NF4 **{counts['nf4'][0]}/{counts['nf4'][1]}**. These are repeated worlds/positions from **16 roots**, not 96 independent puzzles.",'',
        'This new, frozen CPU-only audit verifies all 378 native BF16 cases closed before the external host-RAM stop: 312 at 4K and 66 at 8K. It pairs all 312 shared 4K initial prefixes with the existing independently audited NF4 cache. NF4 produced no closed 8K cases. All remaining registered outcomes stay unknown.','',
        '**The original BF16 terminal audit remains failed because its normal receipt is absent.** No original receipt, successful terminal status, final weight hash or missing inference was synthesized. This report does not replace the canonical CTX21 terminal report or relax its gate. Final target hashes are unavailable for both stopped conditions; before-load parameter bindings and saved output/state integrity are checked.','',
        'The retrospective administrative cutoff and lightweight tokenizer wrapper were frozen and pushed before this separate grading pass. Original per-case scientific checks are AST-identical; the parser, tokenizer ID rules, six geometry formulas and root weighting are unchanged. Earlier NF4 outcomes and the BF16 execution counts were already exposed. This is an exposed development cache audit, not a fresh prospective competence test.','',
        '## Coverage and resource limits','',
        *table(['Backend','Audit provenance','Checked / registered','Missing','Vectors checked'],[
            ['BF16','separate closed-prefix audit; original terminal gate failed', '378 / 624',246,15120],
            ['NF4','original capacity-stop terminal audit','312 / 624',312,12480]]),'',
        f"CPU audit private-memory maximum observed: **{resource['maximum_sampled_private_bytes']/2**20:.1f} MiB**; minimum system available RAM: **{resource['minimum_sampled_available_physical_bytes']/2**30:.3f} GiB**. Admission required 5 GiB available, self-only watchdog reserved 4 GiB available and capped private memory at 0.75 GiB. One numerical thread, Below Normal priority, no Torch import, model load, GPU allocation or pretrained forward. Two-second samples are observations, not continuous peak measurements. NF4 RAM28 inference continued separately.",'',
        'BF16 original inference used four resident decoder layers plus a stage and BF16 head on GPU; NF4 used 22 NF4 decoder layers and a BF16 head. Both used CPU embeddings, the pinned checkpoint, identical initial token IDs, two native loops, no KV cache, and a single free full-vocabulary greedy token per loop. The native head parity checks passed. Placement and memory policies differ, so this is not an isolated quantization-only performance experiment.','',
        'BF16 was stopped to protect host RAM, not on an answer-quality criterion. Missingness follows deterministic source order and resource stops; 8K BF16 results are partial and unpaired. No 8K precision comparison is available. No terminal BF16 runtime/memory peak is claimed.','',
        '## Every accuracy stratum and both loops','',
        'Missing outcomes are unknown. Bounds cover every registered world/root; complete-root success requires both worlds. Copy/erasure are semantic controls, not dose-matched causal interventions. All nulls and format-valid counts remain in the JSON/CSV.','',
        *table(['Backend','Split','Budget','Position','Arm','Loop','Correct / seen','Missing','Whole-roster correct bounds','Both-correct / complete roots','Whole-roster root bounds'],[
            [g['backend'],g['split'],g['context_budget'],g['position'],g['arm'],g['loop'],
             f"{g['correct_worlds'] if g['correct_worlds'] is not None else '—'} / {g['observed_contexts']}",g['missing_contexts'],
             f"{g['whole_roster_correct_bounds']} / {g['registered_contexts']}",
             f"{g['both_world_roots_correct'] if g['both_world_roots_correct'] is not None else '—'} / {g['complete_world_pairs']}",
             f"{g['whole_roster_both_world_bounds']} / {g['registered_roots']}"] for g in accuracy]),'',
        '## Every matched correctness stratum','',
        'These paired counts share prompts and worlds. They do not supply independent root-level formal comparisons.','',
        *table(['Split','Budget','Position','Arm','Loop','Both correct','BF16 only','NF4 only','Neither','Matched','Missing pairs'],[
            [g[k] for k in ('split','context_budget','position','arm','loop','both_correct','BF16_only_correct','NF4_only_correct','neither_correct','matched_contexts','missing_matched_contexts')] for g in paired]),'',
        '## Every physical geometry stratum','',
        'Physical residuals before final loop RMSNorm, at zero-based layers 5/11/17/21. Displacement Δ = h₂ − h₁; relative L2 uses the BF16 vector norm. Available worlds are averaged within each root, then roots equally. Zero-norm metrics are undefined. No coordinate alignment or independently fitted SAE-index subtraction is used. The scalar CSV includes defined roots and undefined-context counts separately for every metric.','',
        *table(['Split','Budget','Position','Arm','Layer','Matched / registered','Observed / registered roots','Partial roots','h1 relL2','h1 cosine','h2 relL2','h2 cosine','Δ relL2','Δ cosine'],[
            [g['split'],g['context_budget'],g['position'],g['arm'],g['layer_zero_based'],
             f"{g['matched_contexts']} / {g['registered_contexts']}",f"{g['observed_roots']} / {g['registered_roots']}",g['partial_world_roots'],
             *[number(g[k]['equal_root_mean']) for k in METRICS]] for g in physical]),'',
        '## Interpretation and provenance','',
        'Accuracy, state similarity and displacement differences are descriptive. High cosine is not equivalence, mediation or faithful reasoning evidence. Effect/sham floors and causal power remain uncalibrated; planned/actual formal comparisons are zero, with resolution **REDESIGN_OR_PILOT_ONLY**. No BPD donor transfer or SAE analysis is claimed for this cache. Earlier failed competence gates remain unchanged. The original 168-case test stays sealed; the continuous research goal remains active.','',
        f"Original source seal: `{payload['original_source_manifest_sha256']}`. Separate audit seal: `{payload['resource_audit_source_manifest_sha256']}`.",'',
        f"BF16 closed-prefix audit: `{payload['backends']['bf16']['audit_sha256']}`. Original NF4 audit: `{payload['backends']['nf4']['audit_sha256']}`. Stop attestation: `{payload['stop_attestation_sha256']}`.",'',
        '[Protocol](../../experiments/nan_ctx21_resource29/PROTOCOL.md) · [Registration](../../experiments/nan_ctx21_resource29/registration.json) · [All aggregate data](nan_ctx21_resource29_results.json) · [Accuracy CSV](nan_ctx21_resource29_results_accuracy.csv) · [Paired CSV](nan_ctx21_resource29_results_paired.csv) · [576 geometry scalars](nan_ctx21_resource29_results_geometry.csv) · [Original terminal report](NAN_CTX21_PROGRESS.md)','']
    Path(OUTPUTS[0]).write_text('\n'.join(lines), encoding='utf-8', newline='\n')


def main():
    import argparse
    p = argparse.ArgumentParser(); p.add_argument('--snapshot', required=True); args = p.parse_args()
    with Guard() as guard:
        seal = verify(); guard.stage='closed_native_BF16_audit'
        bf16 = run(args.snapshot, guard)
        nfpath = OLD_RUNS/'audits/nf4.json'; nf4 = read(nfpath)
        assert nf4['status']=='passed' and nf4['contexts_checked']==312
        assert nf4['receipt_sha256']==sha(OLD_RUNS/'nf4/receipt.json')
        public = rows(OLD/'fixtures/public.jsonl'); cfg = read(OLD/'config.json')
        audits = dict(bf16=bf16, nf4=nf4)
        guard.stage='matched_geometry'; physical = geometry(public, audits, cfg)
        assert physical['matched_initial_prefixes']==312
        backends = {}
        for backend, audit in audits.items():
            path = RUNS/'bf16_closed_prefix_audit.json' if backend=='bf16' else nfpath
            backends[backend] = dict(status=audit['status'], audit_sha256=sha(path),
                checked_contexts=audit['contexts_checked'], missing_contexts=audit['missing_contexts'],
                physical_vectors_checked=audit['physical_vectors_checked'],
                orphan_artifacts_not_scored=len(audit['orphan_case_artifacts_not_scored']),
                target_parameter_sha256_after=None, groups=groups(public, audit['records']))
        guard.check(); no_torch()
        payload = dict(experiment='NAN-CTX21-RESOURCE29', status='completed_separate_closed_prefix_audit',
            original_source_manifest_sha256=seal['original_source_manifest_sha256'],
            resource_audit_source_manifest_sha256=sha(ROOT/'SOURCE_MANIFEST.json'),
            stop_attestation_sha256=sha('experiments/nan_ram28/RESOURCE_OUTCOME.json'),
            original_BF16_terminal_audit='failed_missing_normal_receipt', original_gate_changed=False,
            original_receipt_synthesized=False, backends=backends, paired_physical_geometry=physical,
            packages=bf16['packages'], original_target_parameter_hashes_after_available=False,
            source_selection='All 378 consecutive native case_closed ordinals, fixed from the pre-existing resource-stop attestation; no output-based selection.',
            unit='16 DEV/4 CAL roots; worlds, positions, arms, loops and layers are repeated measures; one pinned checkpoint.',
            missingness='Ordered resource-stop prefix, not random; 246 BF16 and 312 NF4 outcomes unknown; 8K unpaired.',
            scientific_causal_floor=None, matched_sham_floor=None, estimated_causal_power='Uncalibrated',
            resolution_verdict='REDESIGN_OR_PILOT_ONLY', planned_formal_comparisons=0, actual_formal_comparisons=0,
            BPD_donor_transfer_attempted=False, changes_old_failed_gates=False, pretrained_forwards=0,
            model_loaded=False, Torch_imported=False, sealed_test_accessed=False, continuous_goal_remains_active=True)
    resource = read(RUNS/'resources.json'); samples = resource['samples']
    assert resource['status']=='passed'
    payload['audit_resources'] = dict(resources_sha256=sha(RUNS/'resources.json'), sample_count=len(samples),
        maximum_sampled_private_bytes=max(v['private_bytes'] for v in samples),
        maximum_sampled_working_set_bytes=max(v['working_set_bytes'] for v in samples),
        minimum_sampled_available_physical_bytes=min(v['available_physical_bytes'] for v in samples),
        CPU_threads=1, priority='BelowNormal', no_Torch_import=True)
    presentation(payload)
    write(RUNS/'completion.json', dict(status='completed_separate_closed_prefix_audit',
        public_outputs_sha256={name:sha(name) for name in OUTPUTS}, pretrained_forwards=0, sealed_test_accessed=False))
    print(json.dumps(dict(status=payload['status'], BF16_contexts=378, NF4_contexts=312,
                          matched_prefixes=312, audit_resources=payload['audit_resources'])), flush=True)


if __name__=='__main__': main()
