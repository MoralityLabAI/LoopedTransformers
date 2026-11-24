"""Matched CTX08 development accuracy and complete physical-geometry summary."""
import argparse
from collections import defaultdict
from datetime import datetime,timezone
import hashlib
from pathlib import Path
from experiments.nan_fit07.io import read,rows,sha,write
from .audit_native import final_answer


def pair_counts(pairs):
    result={'both_correct':0,'BF16_only_correct':0,'NF4_only_correct':0,'neither_correct':0}
    for b,n in pairs:
        key='both_correct' if b and n else 'BF16_only_correct' if b else 'NF4_only_correct' if n else 'neither_correct'
        result[key]+=1
    return result


def report(runs,out):
    root=Path('experiments/nan_ctx08');seal=read(root/'SOURCE_MANIFEST.json')
    assert sha(root/'SOURCE_MANIFEST.json')=='7f729d76bd0b37eabbeaf4a2fd8dc0faa7621c01f8830554bbf424a63dd4d9ec'
    assert sha(root/'fixtures/public.jsonl')==seal['worker_files_sha256']['fixtures/public.jsonl']
    assert sha(root/'fixtures/host_labels.jsonl')==seal['host_only_files_sha256']['fixtures/host_labels.jsonl']
    public=rows(root/'fixtures/public.jsonl');gold={r['id']:r['answer'] for r in rows(root/'fixtures/host_labels.jsonl')}
    behavior=read(out/'nan_ctx08_progress.json');towers=read(out/'nan_ctx08_tower_probes.json')
    assert behavior['source_manifest_sha256']==towers['source_manifest_sha256']==sha(root/'SOURCE_MANIFEST.json')
    assert towers['paired_physical_geometry'] is not None
    for b in ('bf16','nf4'):
        r=behavior['backends'][b];assert r['status']=='completed' and r['completed_contexts']==291 and r['physical_vectors_checked']==11640
        assert r['worker_receipt_sha256']==sha(runs/b/'receipt.json')==towers['backends'][b]['worker_receipt_sha256']
    strata=defaultdict(list);formats=defaultdict(lambda:[0,0]);digest=hashlib.sha256()
    for index,source in enumerate(public):
        records=[]
        for b in ('bf16','nf4'):
            file=runs/b/f'case_{index:04d}.json';r=read(file);digest.update(sha(file).encode())
            assert all(r[k]==source[k] for k in ('id','root','split','context_budget','position','task','world','copy_control'))
            assert len(r['phases'])==2;records.append(r)
        if source['split']!='development':continue
        for t in (0,1):
            values=[final_answer(r['phases'][t]['free_token_text'],source['schema']) for r in records]
            key=(source['context_budget'],source['task'],source['copy_control'],t+1)
            strata[key].append((source['root'],values[0]==gold[source['id']],values[1]==gold[source['id']]))
            for b,v in enumerate(values):formats[key][b]+=int(v is not None)
    summaries=[]
    for key,values in sorted(strata.items()):
        counts=pair_counts([(b,n) for _,b,n in values]);roots={r for r,_,_ in values}
        assert len(values)==24 and len(roots)==4
        summaries.append(dict(context_budget=key[0],task=key[1],copy_control=key[2],loop=key[3],contexts=len(values),source_root_groups=len(roots),
            BF16_correct=sum(b for _,b,_ in values),NF4_correct=sum(n for _,_,n in values),
            BF16_format_valid=formats[key][0],NF4_format_valid=formats[key][1],**counts))
    geometry=[]
    for layer,r in sorted(towers['paired_physical_geometry'].items(),key=lambda p:int(p[0])):
        states=[g for g in r['states'] if g['split']=='development' and g['loop']==2 and not g['copy_control']]
        deltas=[g for g in r['loop_displacements'] if g['split']=='development' and not g['copy_control']]
        def bounds(rows,key):
            values=[g[key] for g in rows if g[key] is not None]
            return {'minimum':min(values) if values else None,'maximum':max(values) if values else None,'defined_strata':len(values),'registered_strata':len(rows)}
        geometry.append(dict(layer_zero_based=int(layer),loop2_state_relative_l2=bounds(states,'physical_relative_l2'),
            loop2_state_cosine=bounds(states,'physical_cosine'),loop_displacement_relative_l2=bounds(deltas,'delta_relative_l2')))
    result=dict(updated_UTC=datetime.now(timezone.utc).isoformat(),source_manifest_sha256=sha(root/'SOURCE_MANIFEST.json'),
        implementation_sha256=sha(Path(__file__)),behavior_report_sha256=sha(out/'nan_ctx08_progress.json'),tower_report_sha256=sha(out/'nan_ctx08_tower_probes.json'),
        matched_record_sequence_sha256=digest.hexdigest(),matched_DEV_contexts=288,independent_DEV_source_families=4,
        paired_correctness=summaries,paired_physical_geometry_ranges=geometry,
        formal_inferential_comparisons=0,BPD_donor_transfer_attempted=False,sealed_test_accessed=False)
    write(out/'nan_ctx08_comparison.json',result)
    lines=['# NAN-CTX-08: matched BF16/NF4 development comparison','',
        'Both registered backends completed all 291 contexts/profiles and passed independent token/score/state audits. '+
        'The 288 DEV contexts come from four source families; one separate CAL family supplied three resource profiles. '+
        'These are exposed development results on one checkpoint, not 288 independent trials.','',
        '| Context | Task | Control | BF16 correct | NF4 correct | BF16 only correct | NF4 only correct |',
        '|---:|---|---|---:|---:|---:|---:|']
    for g in summaries:
        if g['loop']==2:lines.append(f"| {g['context_budget']} | {g['task']} | {'copy' if g['copy_control'] else 'main'} | {g['BF16_correct']}/24 | {g['NF4_correct']}/24 | {g['BF16_only_correct']} | {g['NF4_only_correct']} |")
    lines+=['','The table uses the free first token after loop 2, parsed with the same bounded anchored label grammar. '+
        'The JSON retains both loops, invalid outputs and paired discordance counts without dropping wrong cases. '+
        'Direct lookup and two-record joins remain separate tasks; explicit answer-copy controls do not establish join competence.','',
        '| Layer | Loop-2 state relative L2 range | State cosine range | Recurrence-displacement relative L2 range |',
        '|---:|---:|---:|---:|']
    for g in geometry:
        def span(key):
            r=g[key];return f"{r['minimum']:.3f}–{r['maximum']:.3f}" if r['minimum'] is not None else 'undefined'
        lines.append(f"| {g['layer_zero_based']} | {span('loop2_state_relative_l2')} | {span('loop2_state_cosine')} | {span('loop_displacement_relative_l2')} |")
    lines+=['','Geometry ranges cover all length/position/task main-DEV strata. Each stratum averages within root '+
        'before averaging its four roots. State differences are normalized by BF16 state norm; recurrence differences '+
        'compare (loop 2 − loop 1) in physical residual coordinates and normalize by the BF16 displacement norm '+
        '(denominator floor 1e-12). All position/control/loop groups are in the tower JSON. Residuals precede final '+
        'loop RMSNorm. Independently fitted BF16/NF4 SAE feature indices are never subtracted.','',
        'Fixed CAL-only SAE dictionaries and their random initial controls were applied to every context at all '+
        'four layers, with every feature/group retained. The eligible BF16 FIT07 T1 VPD asset includes SVD/Haar '+
        'controls and cached-input controller/SAE correlations; loop 2 is outside fit depth. NF4 has no eligible '+
        'learned asset. This is observational geometry, not online intervention or causal transfer.','',
        '| Backend | Peak allocated GiB | Peak reserved GiB |', '|---|---:|---:|']
    for b in ('bf16','nf4'):
        r=behavior['backends'][b];lines.append(f"| {b.upper()} | {r['CUDA_peak_allocated_bytes']/2**30:.3f} | {r['CUDA_peak_reserved_bytes']/2**30:.3f} |")
    lines+=['','Both conditions keep the unchanged BF16 head on GPU. NF4 retains every decoder layer there; '+
        'BF16 retains four layers plus a reusable stage. One CPU thread and Below Normal priority favor the separate '+
        'math process. Latency reflects contention and is not an isolated throughput benchmark.','',
        'Zero formal inferential comparisons; no equivalence, mechanism confirmation or donor transfer. Original '+
        'failed algebraic gates remain failed. The original 168-case ARG-03 test stays sealed.','',
        '[All paired accuracy counts](nan_ctx08_comparison.json) · [All tower/physical geometry](NAN_CTX08_TOWER_PROBES.md) · '+
        '[Behavior and capacity audit](NAN_CTX08_PROGRESS.md) · [Frozen protocol](../../experiments/nan_ctx08/PROTOCOL.md)','']
    (out/'NAN_CTX08_COMPARISON.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n')
    print({'matched_DEV_contexts':288,'source_families':4,'paired_strata':len(summaries),'physical_layers':len(geometry)})


def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',default='runs/nan_ctx08');p.add_argument('--out',default='reports/nanbeige_bpd_gym');a=p.parse_args()
    report(Path(a.runs),Path(a.out))


if __name__=='__main__':main()
