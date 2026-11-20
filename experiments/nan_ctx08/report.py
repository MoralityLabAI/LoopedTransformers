"""Independent structural grades and aggregate long-context evidence only."""
import argparse
from collections import defaultdict
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import re
import numpy as np
from experiments.nan_fit07.io import read,rows,sha,write
from mechinterp.nanbeige_bpd_gym.audit_native import final_answer


def source_answer(text):
    records={k:(n,l) for k,n,l in re.findall(r'^Record (R\d{6}): next=(R\d{6}); label=([A-P])\.$',text,re.M)}
    match=re.search(r'Question: label\((next\()?((?:R)\d{6})\)?\)',text);join,key=match.groups()
    assert len(records)>=8 and set(records)=={v[0] for v in records.values()}
    if join:key=records[key][0]
    answer=records[key][1]
    if 'Copy control:' in text:
        supplied=re.search(r'supplied label: ([A-P])\nAnswer:$',text).group(1);assert supplied==answer
    return answer


def report(runs,out):
    root=Path(__file__).parent;seal=read(root/'SOURCE_MANIFEST.json');public=rows(root/'fixtures/public.jsonl')
    for field in ('worker_files_sha256','host_only_files_sha256'):
        for name,h in seal[field].items():assert sha(root/name)==h,name
    labels={r['id']:r for r in rows(root/'fixtures/host_labels.jsonl')};by={r['id']:r for r in public}
    token=read(root/'fixtures/tokenizer_receipt.json')
    for r in public:assert source_answer(r['prompt'])==labels[r['id']]['answer']
    backends={}
    for backend in ('nf4','bf16'):
        folder=runs/backend;groups=defaultdict(list);seen=set();vectors=0;times=defaultdict(list);max_alloc=0;max_reserved=0
        for file in sorted(folder.glob('case_*.json')):
            record=read(file);key=record['id'];assert key not in seen;seen.add(key);source=by[key]
            assert all(record[n]==source[n] for n in ('root','split','context_budget','position','task','world','copy_control'))
            ids=record['initial_prompt_ids'];binding=token['contexts'][key]
            assert len(ids)==binding['tokens'] and hashlib.sha256(json.dumps(ids).encode()).hexdigest()==binding['initial_prompt_ids_sha256']
            assert all(type(i) is int and 0<=i<166144 for i in ids) and len(record['phases'])==2
            for t,phase in enumerate(record['phases'],1):
                assert phase['loop']==t;score=phase['score'];free=score['free_top_token_id']
                assert 0<=free<166144 and score['vocabulary_size']==166144
                assert math.isfinite(score['full_log_normalizer']) and score['full_log_normalizer']>=score['free_top_token_logit']-1e-8
                assert score['free_top_token_logit']>=max(score['selected_logits'].values())
                value=final_answer(phase['free_token_text'],source['schema'])
                group=(source['split'],source['context_budget'],source['position'],source['task'],source['copy_control'],t)
                groups[group].append(dict(root=source['root'],correct=int(value==labels[key]['answer']),format=int(value is not None)))
            statefile=folder/record['state_file'];assert sha(statefile)==record['state_file_sha256']
            expected={(t,l,s) for t in (1,2) for l in (5,11,17,21) for s in ('residual_after_layer','down_proj_input','down_proj_output','o_proj_input','o_proj_output')}
            assert len(record['state_index'])==40 and {(r['loop'],r['layer_zero_based'],r['site']) for r in record['state_index']}==expected
            with np.load(statefile,allow_pickle=False) as z:
                assert set(z.files)=={r['array_key'] for r in record['state_index']}
                for o in record['state_index']:
                    v=z[o['array_key']];width={'down_proj_input':10752,'o_proj_input':6144}.get(o['site'],3072)
                    assert v.shape==(width,) and v.dtype==np.float32 and np.isfinite(v).all();vectors+=1
            times[source['context_budget']].append(record['elapsed_seconds'])
            max_alloc=max(max_alloc,record['CUDA_peak_allocated_bytes']);max_reserved=max(max_reserved,record['CUDA_peak_reserved_bytes'])
        if seen:
            parity=read(folder/'resident_head_parity.json');assert parity['source_manifest_sha256']==sha(root/'SOURCE_MANIFEST.json')
            assert len(parity['checks'])==2 and all(r['vocabulary_logits_bit_exact'] and r['all_scorer_fields_exact'] and r['full_vocabulary_logits_checked']==166144 for r in parity['checks'])
            assert read(folder/'manifest.json')['head_device'].startswith('cuda')
        else:parity=None
        summary=[]
        for (split,length,position,task,copy,loop),values in sorted(groups.items()):
            roots=defaultdict(list)
            for v in values:roots[v['root']].append(v['correct'])
            summary.append(dict(split=split,context_budget=length,position=position,task=task,copy_control=copy,loop=loop,
                contexts=len(values),independent_source_roots=len(roots),correct=sum(v['correct'] for v in values),
                format_valid=sum(v['format'] for v in values),root_macro_correctness=float(np.mean([np.mean(v) for v in roots.values()]))))
        receipt=read(folder/'receipt.json') if (folder/'receipt.json').exists() else None
        if receipt:
            assert receipt['completed_contexts']==len(seen) and receipt['source_manifest_sha256']==sha(root/'SOURCE_MANIFEST.json')
            max_alloc=max(max_alloc,receipt['CUDA_peak_allocated_bytes']);max_reserved=max(max_reserved,receipt['CUDA_peak_reserved_bytes'])
        backends[backend]=dict(status=receipt['status'] if receipt else 'partial_evidence' if seen else 'queued',completed_contexts=len(seen),
            registered_contexts=len(public),physical_vectors_checked=vectors,groups=summary,
            resident_head_parity=parity,CUDA_peak_allocated_bytes=max_alloc,CUDA_peak_reserved_bytes=max_reserved,
            worker_receipt_sha256=sha(folder/'receipt.json') if receipt else None,
            levels=receipt['levels'] if receipt else [],failure_type=(receipt.get('failure') or {}).get('type') if receipt else None,
            observed_latency_seconds_by_budget={str(k):{'contexts':len(v),'mean':float(np.mean(v)),'maximum':max(v)} for k,v in sorted(times.items())})
    result=dict(experiment='NAN-CTX-08',updated_UTC=datetime.now(timezone.utc).isoformat(),source_manifest_sha256=sha(root/'SOURCE_MANIFEST.json'),
        backends=backends,evidence_class='source-only exposed development context challenge; no causal or equivalence claim',
        independent_DEV_roots=4,independent_CAL_capacity_roots=1,formal_inferential_comparisons=0,
        original_algebraic_gate_unchanged=True,BPD_transfer_attempted=False,sealed_test_accessed=False,
        privacy='aggregate source-root statistics and hashes only; raw tokens, replies and physical vectors remain local')
    write(out/'nan_ctx08_progress.json',result)
    lines=['# NAN-CTX-08 resident-head long-context pilot','',f"Updated {result['updated_UTC']}",'',
        'Fresh 1K/2K/4K-token ledgers test direct lookup and a two-record join, with query records early/middle/late, '+
        'two worlds differing in one relevant label, and explicit copy controls. Four independent DEV root families generate 288 contexts; '+
        'one separate CAL family supplies three capacity profiles. Longer variants and worlds are repeated measures.','',
        'NF4 retains every decoder layer on CUDA. BF16 retains four decoder layers plus a reusable staging layer. '+
        'Both keep the unchanged BF16 vocabulary head on CUDA and use the same full-vocabulary chunked scorer; '+
        'only the embedding lookup remains on CPU in NF4. This removes repeated CPU-to-GPU head-weight transfers. '+
        'Real-checkpoint head parity must pass before cases can be reported. Old frozen conditions are preserved.','',
        '| Backend | Evidence status | Contexts | Physical vectors checked | CUDA allocated peak GiB |',
        '|---|---|---:|---:|---:|']
    for b,r in backends.items():lines.append(f"| {b.upper()} | {r['status']} | {r['completed_contexts']}/291 | {r['physical_vectors_checked']} | {r['CUDA_peak_allocated_bytes']/2**30:.3f} |")
    lines+=['','| Backend | Context budget | Position | Task | Control | Native loop | Correct | Format valid | Independent roots |',
        '|---|---:|---|---|---|---:|---:|---:|---:|']
    for b,r in backends.items():
        for g in r['groups']:
            if g['split']!='development':continue
            lines.append(f"| {b.upper()} | {g['context_budget']} | {g['position']} | {g['task']} | {'copy' if g['copy_control'] else 'main'} | {g['loop']} | {g['correct']}/{g['contexts']} | {g['format_valid']}/{g['contexts']} | {g['independent_source_roots']} |")
    lines+=['','The endpoint is one free first token after each native loop, scored with a bounded anchored label grammar. '+
        'It is distinct from native prefix-stop and Prime complete-final-reply generation. No forced candidates, answer-based filtering '+
        'or long rationale generation is used. Wrong CAL answers do not block DEV: the CAL profile gates hardware capacity only.','',
        'A CUDA OOM stops that backend at the current level; all attempted outcomes remain, and unattempted contexts stay missing. '+
        'Each backend has a 7200-second between-context budget and an 85% CUDA allocator limit. '+
        'CPU threads are limited to one and priority is Below Normal to favor the concurrent math process. '+
        'Observed latency includes shared-machine contention and is not an isolated throughput benchmark.','',
        'Layer-5/11/17/21 residuals and projection inputs/outputs are retained locally for SAE and learned-factor cross-reference. '+
        'Residuals precede loop-final RMSNorm; the down_proj identity bridge does not extend to o_proj. '+
        'No dictionaries or factors are fitted on these DEV contexts. No donor transfer is enabled. '+
        'The original 168-case test stays sealed.','',
        '[Protocol](../../experiments/nan_ctx08/PROTOCOL.md) · [Frozen source](../../experiments/nan_ctx08/SOURCE_MANIFEST.json) · [Aggregate data](nan_ctx08_progress.json)','']
    (out/'NAN_CTX08_PROGRESS.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n')
    print({b:{'status':r['status'],'contexts':r['completed_contexts']} for b,r in backends.items()})


def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',default='runs/nan_ctx08');p.add_argument('--out',default='reports/nanbeige_bpd_gym');a=p.parse_args()
    report(Path(a.runs),Path(a.out))


if __name__=='__main__':main()
