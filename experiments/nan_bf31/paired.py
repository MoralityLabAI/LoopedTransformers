"""Descriptive physical geometry with identical complete decoder-batch inputs."""
from collections import defaultdict
import hashlib
import json
import re
from .common import read,write,sha
from .inputs import locations,audit_records
from experiments.nan_mlp27.paired import AB_behavior


def batch_signature(rows):
    if not rows:return None
    counts={r['batch_size'] for r in rows};padded={r['padded_length'] for r in rows}
    if len(counts)!=1 or len(padded)!=1:return None
    count=next(iter(counts))
    if len(rows)!=count or {r['batch_row'] for r in rows}!=set(range(count)):return None
    fields=('source_id','mode','prefix_ids_sha256','history_sha256','batch_size','batch_row','padded_length','actual_length','padding_tokens')
    value=[[r[k] for k in fields] for r in sorted(rows,key=lambda r:r['batch_row'])]
    return hashlib.sha256(json.dumps(value,separators=(',',':')).encode()).hexdigest()


def signatures(bases,records):
    # Only audited closed episodes can supply companion rows. A missing or
    # unclosed partner excludes the batch instead of rescuing its response.
    groups=defaultdict(list);proofs={}
    for backend,r in records:
        base=bases[backend];closed=base/r['closed_file'];assert sha(closed)==r['closed_file_sha256'];episode=read(closed)
        assert episode['source_id']==r['source_id']
        closed_proofs={f['file']:f['sha256'] for f in episode['fragments']}
        for key in r['prefix_keys']:
            name=key['file'];assert name in closed_proofs and sha(base/name)==closed_proofs[name]
            p=read(base/name);assert p['source_id']==r['source_id'] and p['mode']==r['mode']
            for field in ('prefix_ids_sha256','history_sha256','batch_size','padded_length'):assert p[field]==key[field]
            match=re.fullmatch(r'(baseline|AB)_(\d+)_r[01]_p(\d+)\.json',name);assert match
            compact={k:p[k] for k in ('source_id','mode','prefix_ids_sha256','history_sha256','batch_size','batch_row','padded_length','actual_length','padding_tokens')}
            compact['file']=name;groups[backend,match[1],int(match[2]),int(match[3])].append(compact)
            proofs[backend,name]=closed_proofs[name]
    result={}
    for (backend,*_),rows in groups.items():
        sig=batch_signature(rows)
        for r in rows:result[backend,r['file']]=sig
    return result,proofs


def run(artifact):
    import numpy as np
    from .host import decode
    bases,_=locations(artifact);items=[(r['backend'],r) for r in audit_records(artifact)]
    sigs,proofs=signatures(bases,items);records={};groups=defaultdict(list);admitted=excluded=tail=missing_companion=0
    for backend,r in items:
        key=(r['source_id'],r['lane'],r['mode'],r['closed_file'] if r['lane']=='AB' else '')
        assert backend not in records.get(key,{})
        records.setdefault(key,{})[backend]=r
    unpaired=sum(set(v)!={'nf4','bf16'} for v in records.values())
    for backends in records.values():
        if set(backends)!={'nf4','bf16'}:continue
        A=backends['bf16'];B=backends['nf4'];tail+=abs(len(A['prefix_keys'])-len(B['prefix_keys']))
        for aa,bb in zip(A['prefix_keys'],B['prefix_keys']):
            sa=sigs['bf16',aa['file']];sb=sigs['nf4',bb['file']]
            if sa is None or sb is None:missing_companion+=1;continue
            if sa!=sb:excluded+=1;continue
            pairs=[]
            for backend,p in [('bf16',aa),('nf4',bb)]:
                file=bases[backend]/p['file'];assert sha(file)==proofs[backend,p['file']];fragment=read(file)
                state=bases[backend]/fragment['state_file'];assert sha(state)==fragment['state_sha256']
                with np.load(state,allow_pickle=False) as z:
                    pairs.append({(i['loop'],i['layer'],i['site']):decode(z[i['key']]).astype(float) for i in fragment['state_index']})
            for layer in (5,11,17,21):
                def cosine(x,y):return float(x@y/max(float(np.linalg.norm(x)*np.linalg.norm(y)),1e-12))
                x=[pairs[0][p,layer,'residual_after_layer'] for p in (1,2)]
                y=[pairs[1][p,layer,'residual_after_layer'] for p in (1,2)]
                dx=x[1]-x[0];dy=y[1]-y[0]
                metrics=[cosine(x[0],y[0]),cosine(x[1],y[1]),cosine(dx,dy),float(np.linalg.norm(dy-dx)/max(float(np.linalg.norm(dx)),1e-12))]
                groups[A['stage'],A['split'],A['lane'],A['mode'],json.dumps(A['source_group'],sort_keys=True),layer,A['root']].append(metrics)
            admitted+=1
    summaries=defaultdict(list)
    for key,values in groups.items():summaries[key[:-1]].append(np.mean(values,axis=0))
    output=[]
    for key,values in sorted(summaries.items()):
        output.append(dict(stage=key[0],split=key[1],lane=key[2],mode=key[3],source_group=json.loads(key[4]),layer=key[5],independent_roots=len(values),
            mean_of_root_means=np.mean(values,axis=0).tolist(),metrics=['loop1_cosine','loop2_cosine','recurrence_displacement_cosine','recurrence_displacement_relative_L2_BF16_denominator']))
    write(artifact/'host/paired_geometry.json',dict(status='completed',exact_history_and_batch_shape_prefix_pairs=admitted,
        complete_batch_signature_required=True,shared_positions_excluded_for_history_or_shape=excluded,
        shared_positions_excluded_for_missing_audited_companion=missing_companion,unshared_prefix_tails=tail,unpaired_closed_episode_keys=unpaired,
        root_level_geometry=output,descriptive_AB_behavior=AB_behavior(records),zero_norm_policy='1e-12 denominator floor; no inferred equivalence',
        fitted_alignment=False,formal_comparisons=0,sealed_test_accessed=False))
