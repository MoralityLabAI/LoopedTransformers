"""Independent stdlib audit of published rankings and finite-correctness grades."""
import argparse
from itertools import product
from fractions import Fraction as Q
from collections import defaultdict
import json
import math
from pathlib import Path
from .verify_sources import execute,verify

def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment-root',default='experiments/nan_bpd01');p.add_argument('--out',required=True);a=p.parse_args()
    root=Path(a.experiment_root);verify(root/'fixtures')
    labels={r['identifier']:r for r in map(json.loads,(root/'fixtures/host_labels.jsonl').read_text().splitlines())}
    pairs=json.loads((root/'fixtures/pairs.json').read_text());donors={p['recipient']:p['donor'] for p in pairs}
    programs=list(product('INSR',repeat=3));valid={};correct={}
    for cid,c in labels.items():
        valid[cid]=[];correct[cid]=[]
        for g in programs:
            accepted=all(execute(f[:13],c['topology'],g)==y[:13] for f,y in c['demos'])
            valid[cid].append(accepted)
            correct[cid].append(accepted and execute(c['query'],c['topology'],g)==c['future'])
    scores=json.loads((root/'receipts/compact_score_tapes.json').read_text())
    grades=json.loads((root/'receipts/graded_cases.json').read_text());checked=0
    for row in grades:
        cid=row['case_id'];target=donors[cid] if row['target']=='donor' else cid
        values=scores[row['backend']][cid]['scores'][row['arm']]
        assert len(values)==64 and all(math.isfinite(v) for v in values)
        order=sorted(range(64),key=lambda i:(-values[i],i))
        found=next((i for i in order[:12] if valid[target][i]),None)
        future=int(found is not None and correct[target][found]);top1=int(correct[target][order[0]])
        assert future==row['future_correct'] and top1==row['top1_future_correct'],row
        assert [list(programs[i]) for i in order[:12]]==row['top12_programs']
        assert order.index(programs.index(tuple(labels[target]['source_graph'])))+1==row['source_rank']
        checked+=1
    results=json.loads((root/'receipts/results.json').read_text())
    for b,r in results['backend_results'].items():
        for arm,summary in r['task'].items():
            rr=[v for v in grades if v['backend']==b and v['arm']==arm and v['target']=='recipient' and arm in r['graded_cases'][v['case_id']]['arms']]
            assert len(rr)==summary['n_cases']
            assert sum(v['future_correct'] for v in rr)==summary['future12_correct']
            assert sum(v['top1_future_correct'] for v in rr)==summary['top1_future_correct']
        both=both_top=0
        for pair in pairs:
            if pair['split']!='development':continue
            rr=[v for v in grades if v['backend']==b and v['arm']=='native2' and v['target']=='recipient' and v['case_id'] in [pair['recipient'],pair['donor']]]
            assert len(rr)==2
            both+=int(all(v['future_correct'] for v in rr));both_top+=int(all(v['top1_future_correct'] for v in rr))
        assert both==r['both_native_future12_pairs'] and both_top==r['both_native_top1_pairs']
        expected=r['task']['native2']['future12_correct']>=4 and both>=2 and both_top>=1
        assert expected==r['clean_competence_gate']
    assert results['actual_multiplicity_family']==46
    initial=root/'receipts/initial_invalid_dense_analysis'
    old=json.loads((initial/'graded_cases.json').read_text())
    key=lambda r:(r['backend'],r['case_id'],r['arm'],r['target'])
    bykey={key(r):r for r in grades}
    for r in old:
        assert all(bykey[key(r)][k]==v for k,v in r.items()),'An original scalar grade changed during repair'
    assert json.loads((initial/'calibration_frozen.json').read_text())==json.loads((root/'receipts/calibration_frozen.json').read_text())
    contrasts=json.loads((root/'receipts/fsi_contrasts.json').read_text());assert len(contrasts)==46
    pvalues=[];cluster_tests=0
    for row in contrasts:
        if 'manifest' not in row:pvalues.append(Q(1));continue
        grouped=defaultdict(list)
        for c in row['manifest']['cases']:grouped[c['cluster_id']].append(Q(c['candidate']-c['baseline']))
        effects=[sum(v,Q(0))/len(v) for v in grouped.values()];obs=abs(sum(effects,Q(0)))
        extreme=sum(abs(sum((s*x for s,x in zip(signs,effects)),Q(0)))>=obs for signs in product((-1,1),repeat=len(effects)))
        value=Q(extreme,2**len(effects));assert value==Q(row['conditional_signflip_p_exact'])
        assert row['fsi']['cluster_effect_exact']==str(sum(effects,Q(0))/len(effects))
        assert row['fsi']['n_clusters']==len(effects) and row['fsi']['p_unadjusted'] is None
        pvalues.append(value);cluster_tests+=1
    adjusted=[Q(1)]*len(pvalues);running=Q(0)
    for rank,index in enumerate(sorted(range(len(pvalues)),key=lambda i:pvalues[i])):
        running=max(running,min(Q(1),(len(pvalues)-rank)*pvalues[index]));adjusted[index]=running
    assert all(Q(row['conditional_holm_p_exact'])==value for row,value in zip(contrasts,adjusted))
    result={'status':'passed','independent_audit':'stdlib finite-series checker; no Torch, model or EvalMath import',
            'graded_rows_verified':checked,'canonical_graded_rows':sum(r.get('canonical_prefix_consistent',True) for r in grades),
            'original_scalar_grades_unchanged':len(old),'CAL_choices_unchanged':True,'fresh_source_worlds':len(labels),'all_summary_counts_recomputed':True,
            'exact_root_signflip_tests_recomputed':cluster_tests,'holm_family_recomputed':len(contrasts),'sealed_test_accessed':False}
    Path(a.out).write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result))

if __name__=='__main__':main()
