"""Original full diagnostic audit with explicit rank; no criterion removed."""
from collections import defaultdict
import math
import numpy as np
from mechinterp.nanbeige_bpd_gym.audit_fit import check_phase
from mechinterp.nanbeige_bpd_gym.audit_native import final_answer

def diagnostics_audit(data,public,gold,passes,modes,rank,baseline=None,tokenizer=None):
    main={r['id']:r for r in public if not r['copy_control']};seen=set();groups=defaultdict(list)
    base={r['id']:r for r in baseline['records']} if baseline is not None else {}
    for record in data['records']:
        key=record['id'];mode=record['mode'];assert (key,mode) not in seen;seen.add((key,mode))
        source=main[key];assert record['root']==source['root'] and record['split']==source['split']
        assert mode in modes and len(record['phases'])==passes
        for t,p in enumerate(record['phases']):
            check_phase(p,t,tokenizer)
            for name in ('full_vocabulary_KL','actual_last_token_module_write_relative_l2','local_intervention_l2_at_actual_reached_input',
                         'local_intervention_relative_l2_at_actual_reached_input','reached_input_relative_l2'):
                assert math.isfinite(p[name]) and p[name]>=(-1e-6 if name=='full_vocabulary_KL' else 0)
            if mode=='all_one':
                assert p['actual_last_token_module_write_relative_l2']==p['local_intervention_l2_at_actual_reached_input']==p['reached_input_relative_l2']==0
                assert abs(p['full_vocabulary_KL'])<=1e-6
                if base:
                    assert all(p[k]==base[key]['phases'][t][k] for k in ('free_token_id','free_token_text','full_log_normalizer','free_token_logprob'))
            if mode=='reached_state':
                assert math.isfinite(p['reached_vs_clean_mask_max_abs']) and p['reached_vs_clean_mask_max_abs']>=0
                if passes==1:assert p['reached_input_relative_l2']==p['reached_vs_clean_mask_max_abs']==0
            else:assert p['reached_vs_clean_mask_max_abs'] is None
            value=final_answer(p['free_token_text'],source['schema'])
            groups[source['split'],mode,t+1].append((source['root'],int(value==gold[key]),int(value is not None),p))
    assert seen=={(key,mode) for key in main for mode in modes}
    summaries=[]
    for (split,mode,loop),records in sorted(groups.items()):
        assert len(records)==len({r[0] for r in records})==8
        row=dict(split=split,mode=mode,loop=loop,contexts=8,independent_source_roots=8,
            correct=sum(r[1] for r in records),format_valid=sum(r[2] for r in records))
        for name in ('full_vocabulary_KL','actual_last_token_module_write_relative_l2','local_intervention_l2_at_actual_reached_input',
                     'local_intervention_relative_l2_at_actual_reached_input','reached_input_relative_l2','reached_vs_clean_mask_max_abs'):
            values=[r[3][name] for r in records if r[3][name] is not None]
            row[name+'_root_mean']=float(np.mean(values)) if values else None
            row[name+'_root_max']=float(np.max(values)) if values else None
        summaries.append(row)
    static=np.asarray(data['static_CAL_mask']);assert static.shape==(passes,rank) and np.isfinite(static).all()
    assert static.min()>=0 and static.max()<=1
    return summaries
