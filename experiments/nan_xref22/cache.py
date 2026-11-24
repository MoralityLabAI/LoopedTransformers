"""Admit only future independently audited CTX21 files; retain all closed vectors."""
from collections import defaultdict
from pathlib import Path
from .common import ROOT,RUNS,read,rows,sha,write
from experiments.nan_xref19.cache import correlations,nullable,root_mean

STRATA={'ctx21':('split','context_budget','position','arm','loop')}


def coverage(public,meta):
    expected=defaultdict(list); observed=defaultdict(list)
    for row in public:
        for loop in (1,2):
            key=tuple(row[k] for k in STRATA['ctx21'][:-1])+(loop,)
            expected[key].append((row['id'],row['root']))
    for row in meta:
        observed[tuple(row[k] for k in STRATA['ctx21'])].append((row['id'],row['root']))
    assert set(observed)<=set(expected)
    result=[]
    for key,planned in sorted(expected.items()):
        actual=observed[key]
        assert len(actual)==len(set(actual)) and set(actual)<=set(planned)
        result.append(dict(zip(STRATA['ctx21'],key),registered_vectors=len(planned),available_vectors=len(actual),
            missing_vectors=len(planned)-len(actual),registered_roots=len({v[1] for v in planned}),available_roots=len({v[1] for v in actual})))
    assert len(result)==48 and sum(g['registered_vectors'] for g in result)==1248
    return result


def groups(dataset,meta,drop_loop=False):
    keys=STRATA[dataset][:-1] if drop_loop else STRATA[dataset];result=defaultdict(list)
    for i,row in enumerate(meta):
        result[tuple(row[k] for k in keys)].append(i)
    return keys,result


def validate_prefix(public,audit,terminal):
    count=audit['contexts_checked']
    assert audit['status']=='passed' and isinstance(count,int) and 0 <= count <= 624
    assert count==terminal['completed_contexts']==len(audit['records']) and audit['registered_contexts']==len(public)==624
    assert [r['id'] for r in audit['records']]==[r['id'] for r in public[:count]]
    return count


def admit(backend):
    cfg=read(ROOT/'config.json');folder=Path('runs/nan_ctx21');auditfile=folder/'audits'/f'{backend}.json'
    if not auditfile.exists():
        result=dict(status='unrun_no_passed_terminal_context_audit',backend=backend,registered_contexts=624,files=[])
    else:
        audit=read(auditfile);assert audit['status']=='passed' and audit['source_manifest_sha256']==cfg['CTX21_seal']
        receipt=folder/backend/'receipt.json';assert sha(receipt)==audit['receipt_sha256']
        terminal=read(receipt);assert terminal['source_manifest_sha256']==cfg['CTX21_seal']
        assert any(v['event']=='worker_terminal' and v['backend']==backend for v in rows(folder/'queue_events.jsonl'))
        public=rows('experiments/nan_ctx21/fixtures/public.jsonl');count=validate_prefix(public,audit,terminal)
        files=[]
        for ordinal,graded in enumerate(audit['records']):
            recordfile=folder/backend/f'case_{ordinal:04d}.json';assert sha(recordfile)==graded['case_sha256']
            record=read(recordfile);archive=folder/backend/record['state_file']
            assert archive.name==f'case_{ordinal:04d}_states.npz' and sha(archive)==graded['state_sha256']==record['state_file_sha256']
            files.append(dict(record=str(recordfile).replace('\\','/'),record_sha256=graded['case_sha256'],
                state=str(archive).replace('\\','/'),state_sha256=graded['state_sha256']))
        result=dict(status='admitted',backend=backend,worker_status=terminal['status'],audit=str(auditfile).replace('\\','/'),audit_sha256=sha(auditfile),
            receipt=str(receipt).replace('\\','/'),receipt_sha256=sha(receipt),registered_contexts=624,available_contexts=count,
            missing_contexts=624-count,source_manifest_sha256=cfg['CTX21_seal'],files=files)
    write(RUNS/'admission'/f'{backend}.json',result)
    return result


def load(dataset,backend):
    import numpy as np
    assert dataset=='ctx21';admission=read(RUNS/'admission'/f'{backend}.json')
    assert admission['status']=='admitted'
    assert sha(admission['audit'])==admission['audit_sha256'] and sha(admission['receipt'])==admission['receipt_sha256']
    H={layer:[] for layer in (5,11,17,21)};X=[];Y=[];meta=[]
    fields=STRATA[dataset][:-1]+('world','root','id')
    for item in admission['files']:
        assert sha(item['record'])==item['record_sha256'] and sha(item['state'])==item['state_sha256']
        record=read(item['record']);by=defaultdict(dict)
        for v in record['state_index']:
            key=(v['layer_zero_based'],v['site']);assert key not in by[v['loop']]
            by[v['loop']][key]=v['array_key']
        assert set(by)=={1,2}
        with np.load(item['state'],allow_pickle=False) as arrays:
            for loop,index in sorted(by.items()):
                for layer in H:
                    h=arrays[index[layer,'residual_after_layer']];assert h.shape==(3072,) and np.isfinite(h).all();H[layer].append(h)
                x,y=arrays[index[21,'down_proj_input']],arrays[index[21,'down_proj_output']]
                assert x.shape==(10752,) and y.shape==(3072,) and np.isfinite(x).all() and np.isfinite(y).all()
                X.append(x);Y.append(y);meta.append({k:record[k] for k in fields}|dict(loop=loop,state_sha256=item['state_sha256']))
    count=2*admission['available_contexts'];assert len(meta)==count and len({(r['id'],r['loop']) for r in meta})==count
    return {l:np.asarray(v,dtype=np.float64).reshape(count,3072) for l,v in H.items()},np.asarray(X,dtype=np.float64).reshape(count,10752),np.asarray(Y,dtype=np.float64).reshape(count,3072),meta
