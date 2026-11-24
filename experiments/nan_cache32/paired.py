"""Matched cached physical geometry; complete batch ancestry, root means first."""
from collections import defaultdict
import json
from pathlib import Path
from .common import ROOT,read,sha,write


def batches(artifact):
    from .host import journal
    base=Path(artifact)/'host';result={}
    for entry in journal(base/'events.jsonl').values():
        file=base/entry['file'];assert sha(file)==entry['sha256'];value=read(file)
        assert value['source_manifest_sha256']==sha(ROOT/'SOURCE_MANIFEST.json')
        key=value['backend'],value['lane'],value['ordinal'];assert key not in result
        result[key]=value
    return result


def run(artifact,plan,byid,budget):
    import numpy as np
    from experiments.nan_full26.host import decode
    artifact=Path(artifact);items=batches(artifact);groups=defaultdict(list);admitted=excluded=tail=unverified=0
    for job in plan['jobs']:
        budget.check();A=items.get(('bf16',job['lane'],job['ordinal']));B=items.get(('nf4',job['lane'],job['ordinal']))
        for row,member in enumerate(job['members']):
            if A is None or B is None or not (A['slots'][row]['primary_verified'] and B['slots'][row]['primary_verified']):
                unverified+=1;continue
            pa=next(p for p in A['proofs'] if p['original_row']==row);pb=next(p for p in B['proofs'] if p['original_row']==row)
            tail+=abs(len(pa['prefix_proofs'])-len(pb['prefix_proofs']))
            for aa,bb in zip(pa['prefix_proofs'],pb['prefix_proofs']):
                sa=A['complete_cache_prefixes'][aa['file']]['signature'];sb=B['complete_cache_prefixes'][bb['file']]['signature']
                if sa is None or sb is None or sa!=sb:excluded+=1;continue
                values=[]
                for backend,proof in (('bf16',aa),('nf4',bb)):
                    base=artifact/backend;file=base/proof['file'];assert sha(file)==proof['sha256'];prefix=read(file)
                    state=base/prefix['state_file'];assert sha(state)==prefix['state_sha256']
                    with np.load(state,allow_pickle=False) as z:
                        values.append({(i['loop'],i['layer']):decode(z[i['key']]).astype(float) for i in prefix['state_index'] if i['site']=='residual_after_layer'})
                slot=A['slots'][row]
                for layer in (5,11,17,21):
                    def cosine(x,y):return float(x@y/max(float(np.linalg.norm(x)*np.linalg.norm(y)),1e-12))
                    x=[values[0][p,layer] for p in (1,2)];y=[values[1][p,layer] for p in (1,2)]
                    dx=x[1]-x[0];dy=y[1]-y[0]
                    metrics=[cosine(x[0],y[0]),cosine(x[1],y[1]),cosine(dx,dy),float(np.linalg.norm(dy-dx)/max(float(np.linalg.norm(dx)),1e-12))]
                    groups[slot['stage'],slot['split'],slot['lane'],member['mode'],json.dumps(slot['source_group'],sort_keys=True),layer,slot['root']].append(metrics)
                admitted+=1
    summaries=defaultdict(list)
    for key,values in groups.items():summaries[key[:-1]].append(np.mean(values,axis=0))
    output=[]
    for key,values in sorted(summaries.items()):
        output.append(dict(stage=key[0],split=key[1],lane=key[2],mode=key[3],source_group=json.loads(key[4]),layer=key[5],independent_roots=len(values),
            mean_of_root_means=np.mean(values,axis=0).tolist(),metrics=['loop1_cosine','loop2_cosine','recurrence_displacement_cosine','recurrence_displacement_relative_L2_BF16_denominator']))
    write(artifact/'host/paired_geometry.json',dict(status='completed',exact_complete_cache_lineage_prefix_pairs=admitted,
        shared_positions_excluded_for_cache_ancestry_history_or_shape=excluded,unshared_prefix_tails=tail,
        unpaired_or_unverified_scheduled_rows=unverified,registered_row_pairs=1981,root_level_geometry=output,
        zero_norm_policy='1e-12 denominator floor; no inferred equivalence',fitted_alignment=False,formal_comparisons=0,sealed_test_accessed=False))
