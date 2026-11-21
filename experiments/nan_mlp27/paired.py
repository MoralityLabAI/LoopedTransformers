"""Exact-input, exact-batch-shape physical geometry, with root means only."""
from collections import defaultdict
from .common import read,sha,write

def run(artifact):
    import json
    import numpy as np
    from .host import decode
    host=artifact/'host';records={};groups=defaultdict(list);admitted=excluded=0
    for line in (host/'source_groups.jsonl').read_bytes().splitlines():
        entry=json.loads(line);meta=read(host/entry['file']);assert sha(host/entry['file'])==entry['sha256'];file=host/meta['audit_file'];assert sha(file)==meta['audit_sha256'];r=read(file);r['source_group']=meta['source_group']
        key=(r['source_id'],r['lane'],r['mode'],r['closed_file'] if r['lane']=='AB' else '')
        records.setdefault(key,{})[r['backend']]=r
    for key,backends in records.items():
        if set(backends)!=set(('nf4','bf16')):continue
        A=backends['bf16'];B=backends['nf4']
        for aa,bb in zip(A['prefix_keys'],B['prefix_keys']):
            fields=('prefix_ids_sha256','history_sha256','batch_size','padded_length')
            if any(aa[f]!=bb[f] for f in fields):excluded+=1;continue
            pairs=[]
            for backend,record,p in [('bf16',A,aa),('nf4',B,bb)]:
                file=artifact/backend/p['file'];fragment=read(file);state=artifact/backend/fragment['state_file'];assert sha(state)==fragment['state_sha256']
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
    behavior=AB_behavior(records)
    write(host/'paired_geometry.json',dict(status='completed',exact_history_and_batch_shape_prefix_pairs=admitted,
        shared_positions_excluded_for_history_or_shape=excluded,root_level_geometry=output,descriptive_AB_behavior=behavior,
        zero_norm_policy='1e-12 denominator floor; no inferred equivalence',fitted_alignment=False,formal_comparisons=0,sealed_test_accessed=False))


def AB_behavior(records):
    import json
    import numpy as np
    paired={};groups=defaultdict(list)
    for values in records.values():
        for backend,r in values.items():
            if r['lane']!='AB':continue
            stem=r['closed_file'].rsplit('_r',1)[0]
            key=(backend,stem)
            paired.setdefault(key,{})[r['mode']=='native']=r
    for (backend,stem),pair in paired.items():
        if set(pair)!={False,True}:continue
        A=pair[True];B=pair[False]
        assert A['source_id']==B['source_id'] and A['source_group']==B['source_group']
        key=(backend,A['stage'],A['split'],B['mode'],json.dumps(A['source_group'],sort_keys=True))
        groups[key].append(dict(root=A['root'],A=int(A['grade']['final_correct']),B=int(B['grade']['final_correct'])))
    result=[]
    for key,values in sorted(groups.items()):
        byroot=defaultdict(list)
        for r in values:byroot[r['root']].append(r['B']-r['A'])
        result.append(dict(backend=key[0],stage=key[1],split=key[2],route=key[3],source_group=json.loads(key[4]),
            paired_trajectories=len(values),independent_roots=len(byroot),
            A_correct=sum(r['A'] for r in values),B_correct=sum(r['B'] for r in values),
            gains=sum(r['B']>r['A'] for r in values),losses=sum(r['B']<r['A'] for r in values),
            root_macro_B_minus_A=float(np.mean([np.mean(v) for v in byroot.values()])),
            formal_comparison=False,dose_matched=False))
    return result
