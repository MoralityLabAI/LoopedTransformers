"""Exact-history backend/route geometry; no comparison after divergence."""
from collections import defaultdict
import itertools
from .common import ROOT,RUNS,public_rows,read,sha,verify,write


def run():
    from experiments.nan_ranksite13.resources import below_normal
    below_normal();verify()
    import numpy as np
    public=public_rows();byid={r['id']:r for r in public};indexes={};audits={}
    for backend in ('nf4','bf16'):
        file=RUNS/'audits'/(backend+'.json')
        if not file.exists():continue
        audit=read(file);assert audit['status']=='passed';audits[backend]=sha(file)
        index=RUNS/'audits'/(backend+'_prefixes_local.json');assert sha(index)==audit['local_index_sha256']
        grouped=defaultdict(dict)
        for p in read(index)['prefixes']:
            assert p['prefix_ids_sha256'] not in grouped[p['source_id']]
            grouped[p['source_id']][p['prefix_ids_sha256']]=p
        indexes[backend]=grouped
    jobs=[]
    if len(indexes)==2:
        for row in public:jobs.append(('nf4_vs_bf16','nf4',row,'bf16',row))
    for backend in indexes:
        source={(r['source_case_id'],r['arm'],r['mode']):r for r in public}
        for row in public:
            if row['mode']=='native':continue
            native=source[row['source_case_id'],'main','native']
            jobs.append(('native_vs_'+row['mode'],backend,native,backend,row))
    moments=defaultdict(lambda:defaultdict(list));coverage=[]
    for contrast,a,left,b,right in jobs:
        L=indexes[a].get(left['id'],{});R=indexes[b].get(right['id'],{});shared=set(L)&set(R)
        coverage.append(dict(contrast=contrast,left_backend=a,right_backend=b,root=left['root'],source_case_id=left['source_case_id'],
            arm=left['arm'],mode=right['mode'],left_prefixes=len(L),right_prefixes=len(R),exact_history_matched_prefixes=len(shared)))
        percase=defaultdict(list)
        for key in sorted(shared):
            p,q=L[key],R[key]
            x=read(RUNS/a/p['record_file']);y=read(RUNS/b/q['record_file'])
            assert x['prefix_ids_sha256']==y['prefix_ids_sha256']
            u=RUNS/a/p['state_file'];v=RUNS/b/q['state_file'];assert sha(u)==p['state_sha256'] and sha(v)==q['state_sha256']
            with np.load(u,allow_pickle=False) as U,np.load(v,allow_pickle=False) as V:
                mapx={(i['loop'],i['layer_zero_based'],i['site']):i['array_key'] for i in x['state_index']}
                mapy={(i['loop'],i['layer_zero_based'],i['site']):i['array_key'] for i in y['state_index']}
                assert mapx.keys()==mapy.keys() and len(mapx)==40
                for coordinate in mapx:
                    A=U[mapx[coordinate]].astype(np.float64);B=V[mapy[coordinate]].astype(np.float64)
                    na=float(np.linalg.norm(A));nb=float(np.linalg.norm(B));gap=float(np.linalg.norm(A-B))
                    percase[coordinate].append((gap,gap/na if na else None,float(A@B/(na*nb)) if na*nb else None))
        for coordinate,values in percase.items():
            key=(contrast,a,b,left['split'],left['query_type'],left['event_count'],left['arm'],right['mode'],*coordinate)
            means=[float(np.mean([v[i] for v in values if v[i] is not None])) if any(v[i] is not None for v in values) else None for i in range(3)]
            moments[key][left['root']].append(means)
    groups=[]
    for key,byroot in sorted(moments.items()):
        averages=[]
        for i in range(3):
            rootmeans=[float(np.mean([v[i] for v in vv if v[i] is not None])) for vv in byroot.values() if any(v[i] is not None for v in vv)]
            averages.append(float(np.mean(rootmeans)) if rootmeans else None)
        names=('contrast','left_backend','right_backend','split','query_type','event_count','arm','mode','loop','layer_zero_based','site')
        groups.append(dict(zip(names,key))|dict(independent_roots=len(byroot),physical_L2_root_mean=averages[0],
            relative_L2_root_mean=averages[1],cosine_root_mean=averages[2],zero_norm_convention='undefined values omitted and reported as null if all undefined'))
    write(RUNS/'paired_geometry.json',dict(status='completed',source_manifest_sha256=sha(ROOT/'SOURCE_MANIFEST.json'),audit_sha256=audits,
        coverage=coverage,groups=groups,exact_prompt_and_generated_history_only=True,semantic_alignment_after_divergence=False,
        formal_inferential_comparisons=0,sealed_test_accessed=False))


if __name__=='__main__':run()
