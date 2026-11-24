"""Independent CPU checks of frozen NAN-FIT-07 evidence, without model loading.

This is a post-registration evidence auditor, not an inference or selection rule.
Detailed answers, tokens, training-root identities and checkpoints remain local.
"""
import argparse
from collections import defaultdict
import hashlib
import math
import os
from pathlib import Path
import re
import numpy as np
from experiments.nan_fit07.io import read,rows,sha,write
from experiments.nan_fit07.pointwise import masks
from .audit_native import final_answer


ROOT=Path('experiments/nan_fit07')


def dependency_matches(data,expected):
    """Allow only Git's LF/CRLF conversion for source dependency inspection.

    Frozen model workers continue requiring their original exact local bytes.
    Prompt/data/checkpoint/manifest checks below remain strictly byte-for-byte.
    """
    lf=data.replace(b'\r\n',b'\n')
    return expected in {hashlib.sha256(v).hexdigest() for v in (data,lf,lf.replace(b'\n',b'\r\n'))}


def registered_sources():
    seal=read(ROOT/'SOURCE_MANIFEST.json');cfg=read(ROOT/'config.json')
    assert sha(ROOT/'SOURCE_MANIFEST.json')=='8890f915d7b1ba05dc611dfb62e2c1d40abc2cc5bb71351e7b338f640b6f104e'
    for field in ('worker_files_sha256','host_only_files_sha256'):
        for name,h in seal[field].items():assert sha(ROOT/name)==h,name
    for name,h in seal['model_dependencies_sha256'].items():assert dependency_matches(Path(name).read_bytes(),h),name
    public=rows(ROOT/'fixtures/public.jsonl');labels=rows(ROOT/'fixtures/host_labels.jsonl')
    gold={r['id']:r['answer'] for r in labels};assert len(public)==len(gold)==32
    assert {r['id'] for r in public}==set(gold) and len({r['root'] for r in public})==16
    for row in public:
        if row['copy_control']:
            assert row['prompt'].splitlines()[-1]==gold[row['id']]
            continue
        query=row['prompt'].rsplit('Edges: ',1)[1]
        edges,start,hops=re.fullmatch(r'([^\n]+)\nStart: ([A-P])\nHops: (1)\nAnswer:',query).groups()
        graph=dict(pair.split('->') for pair in edges.split(', '))
        assert set(graph)==set(graph.values())==set('ABCDEFGHIJKLMNOP')
        assert graph[start]==gold[row['id']] and hops=='1'
    return seal,cfg,public,gold


def check_phase(phase,index,tokenizer=None):
    assert phase['pass_index_zero_based']==index
    assert type(phase['free_token_id']) is int and 0<=phase['free_token_id']<166144
    assert isinstance(phase['free_token_text'],str)
    assert math.isfinite(phase['full_log_normalizer']) and math.isfinite(phase['free_token_logprob'])
    assert phase['free_token_logprob']<=1e-8
    if tokenizer is not None:
        assert tokenizer.decode([phase['free_token_id']],skip_special_tokens=False,clean_up_tokenization_spaces=False)==phase['free_token_text']


def baseline_audit(folder,public,gold,cfg,tokenizer=None):
    baseline=read(folder/'baseline.json');by={r['id']:r for r in public};seen=set();groups=defaultdict(list)
    assert baseline['source_manifest_sha256']==sha(ROOT/'SOURCE_MANIFEST.json')
    assert baseline['public_sha256']==sha(ROOT/'fixtures/public.jsonl') and not baseline['host_gold_opened']
    assert baseline['head_parity_contexts']==32 and len(baseline['records'])==32
    assert re.fullmatch('[0-9a-f]{64}',baseline['target_parameter_sha256'])
    for record in baseline['records']:
        key=record['id'];assert key not in seen;seen.add(key);source=by[key]
        assert all(record[k]==source[k] for k in ('root','split','copy_control'))
        assert len(record['phases'])==2
        for t,p in enumerate(record['phases']):
            check_phase(p,t,tokenizer);value=final_answer(p['free_token_text'],source['schema'])
            groups[source['split'],source['copy_control'],t+1].append((source['root'],int(value==gold[key]),int(value is not None)))
    assert seen==set(by)
    summaries=[dict(split=s,copy_control=c,loop=t,contexts=len(v),source_roots=len({r[0] for r in v}),
        correct=sum(r[1] for r in v),format_valid=sum(r[2] for r in v)) for (s,c,t),v in sorted(groups.items())]
    gate=all(len(groups[s,copy,2])==8 and sum(r[1] for r in groups[s,copy,2])>=(8 if copy else cfg['minimum_clean_correct_per_split'])
        for s in ('calibration','development') for copy in (False,True))
    if (folder/'host_gate.json').exists():
        host=read(folder/'host_gate.json');assert host['baseline_sha256']==sha(folder/'baseline.json')
        assert host['public_sha256']==sha(ROOT/'fixtures/public.jsonl')
        assert host['host_labels_sha256']==sha(ROOT/'fixtures/host_labels.jsonl') and host['allow_fits']==gate
        for r in host['groups']:
            match=next(q for q in summaries if q['loop']==2 and q['split']==r['split'] and q['copy_control']==(r['condition']=='copy'))
            assert r['records']==match['contexts'] and r['correct']==match['correct']
    return baseline,dict(status='passed',baseline_sha256=sha(folder/'baseline.json'),allow_fits=gate,groups=summaries,
        global_head_parity_contexts=32,tokenizer_decoding_checked=tokenizer is not None)


def check_steps(steps,public,cfg):
    cal=[r['id'] for r in public if r['split']=='calibration' and not r['copy_control']]
    assert len(steps)==cfg['fit_steps']==len(cal)==8
    for i,r in enumerate(steps):
        assert r['step']==i and r['source_id']==cal[i] and r['mode']=='closed_loop'
        assert all(math.isfinite(v) for v in r.values() if isinstance(v,(float,int)))
        expected=cfg['coeff_faith']*r['faith_loss']+r['min_loss']+cfg['coeff_stoch']*r['stoch_loss']+cfg['coeff_adv']*r['adv_loss']
        assert math.isclose(expected,r['total_loss'],rel_tol=2e-6,abs_tol=1e-6)
    return {'optimizer_steps':len(steps),'independent_CAL_roots_used':len(cal),
        'DEV_or_copy_training_uses':0,'four_term_loss_accounting_checked':True,
        'first_step_losses':{k:steps[0][k] for k in ('total_loss','faith_loss','min_loss','stoch_loss','adv_loss')},
        'final_step_losses':{k:steps[-1][k] for k in ('total_loss','faith_loss','min_loss','stoch_loss','adv_loss')}}


def checkpoint(path):
    with np.load(path,allow_pickle=False) as z:
        values={k:z[k] for k in z.files}
    assert all(np.isfinite(v).all() for v in values.values())
    return values


def diagnostics_audit(data,public,gold,passes,modes,baseline=None,tokenizer=None):
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
    static=np.asarray(data['static_CAL_mask']);assert static.shape==(passes,8) and np.isfinite(static).all()
    assert static.min()>=0 and static.max()<=1
    return summaries


def cell_audit(folder,backend,passes,cfg,public,gold,seal,baseline,tokenizer=None):
    receipt=read(folder/'receipt.json');assert receipt['status']=='completed'
    assert receipt['backend']==backend and receipt['passes']==passes and receipt['rank']==8
    assert receipt['source_manifest_sha256']==sha(ROOT/'SOURCE_MANIFEST.json')
    assert receipt['initial_svd_file_sha256']==seal['initialization'][backend]['file_sha256']
    assert receipt['effective_operator_sha256']==seal['initialization'][backend]['effective_operator_sha256']
    assert receipt['optimizer_steps_completed']==8 and receipt['target_parameter_sha256_unchanged']==baseline['target_parameter_sha256']
    assert receipt['native_all_one_full_head_bit_exact'] and receipt['native_all_one_boundary_states_bit_exact']
    assert math.isfinite(receipt['pre_optimizer_gradient_gate_loss'])
    memory=receipt['saved_tensor_statistics']
    assert memory['maximum_added_cpu_bytes']==cfg['maximum_added_saved_tensor_CPU_bytes']
    assert 0<=memory['peak_added_cpu_bytes']<=memory['maximum_added_cpu_bytes']
    assert not receipt['BPD_donor_transfer_attempted'] and not receipt['sealed_test_accessed']
    for name,h in receipt['diagnostic_artifact_sha256'].items():assert sha(folder/name)==h
    assert set(receipt['diagnostic_artifact_sha256'])=={'calibration_local.json','steps_local.json','initial_diagnostic_local.json','diagnostic_local.json'}
    steps=check_steps(read(folder/'steps_local.json'),public,cfg)
    banks={}
    for name in ('initial_checkpoint','training_EMA_checkpoint','final_checkpoint'):
        description=receipt[name];path=folder/description['file'];assert sha(path)==description['sha256']
        bank=checkpoint(path);banks[name]=bank
        assert bank['V'].shape==(cfg['input_width'],8) and bank['U'].shape==(8,cfg['output_width'])
        for field,shape in [('mean_comp_acts',(passes,8)),('mean_delta_out',(passes,cfg['output_width'])),('mean_write_rms',(passes,)),('calibrated_passes',(passes,))]:
            assert bank[field].shape==shape
        assert bank['calibrated_passes'].all()
        if name!='initial_checkpoint':
            for key in ('optimizer__factor_V__step','optimizer__factor_U__step','optimizer__ci_proj_in.weight__step','optimizer__ci_out_proj.weight__step'):
                assert bank[key].item()==8
    initial=banks['initial_checkpoint'];final=banks['final_checkpoint']
    svd_path=Path('runs/nan_gym05')/backend/'linear_baselines/layer21_down_proj.npz'
    assert sha(svd_path)==seal['initialization'][backend]['file_sha256']
    svd=checkpoint(svd_path)
    np.testing.assert_array_equal(initial['V'],svd['V8']);np.testing.assert_array_equal(initial['U'],svd['U8'])
    for keys,field in [(('V','U'),'factor_update_l2'),([k for k in initial if k.startswith('ci__')],'controller_update_l2')]:
        delta=sum(np.sum((final[k].astype(np.float64)-initial[k])**2) for k in keys)**.5
        assert delta>0 and math.isclose(delta,receipt[field],rel_tol=2e-5,abs_tol=1e-8)
    replay_path=folder/'CI_replay_local.npz';assert sha(replay_path)==receipt['CI_replay_sha256']
    replay=checkpoint(replay_path);np.testing.assert_allclose(masks(replay['X'],final,receipt['final_checkpoint']['CI_config']),replay['masks'],rtol=2e-5,atol=1e-5)
    calibration=read(folder/'calibration_local.json');cal=[r['id'] for r in public if r['split']=='calibration' and not r['copy_control']]
    assert [r['id'] for r in calibration]==cal
    for t in range(passes):
        values=[r['same_pass_tolerances'][str(t)] for r in calibration]
        assert all(math.isfinite(v) and v>=cfg['tolerance_epsilon'] for v in values)
        assert math.isclose(receipt['training_tolerances'][str(t)],np.mean(values) if passes==2 else 1.,rel_tol=1e-12,abs_tol=1e-12)
    contract=receipt['training_contract']
    assert contract['n_passes']==passes and contract['n_steps']==8 and contract['ci']['n_blocks']==0
    assert contract['adversary']['inner_steps']==2 and contract['coeff_faith']==1e5 and contract['coeff_stoch']==contract['coeff_adv']==.5
    assert contract['ablation']['mode']==('zero' if passes==1 else 'pass_mean')
    assert contract['ablation']['sphere_rescale']==(passes==2)
    for field in ('initial_effective_operator_closure','final_effective_operator_closure'):
        assert receipt[field]['relative_l2']<=cfg['effective_operator_closure_relative_l2_tolerance']
    initial_groups=diagnostics_audit(read(folder/'initial_diagnostic_local.json'),public,gold,passes,['component_only'],tokenizer=tokenizer)
    final_groups=diagnostics_audit(read(folder/'diagnostic_local.json'),public,gold,passes,cfg['diagnostic_modes'],baseline,tokenizer)
    return dict(status='passed',backend=backend,passes=passes,fit_receipt_sha256=sha(folder/'receipt.json'),
        checkpoint_sha256=receipt['final_checkpoint']['sha256'],training=steps,factor_update_l2=receipt['factor_update_l2'],
        controller_update_l2=receipt['controller_update_l2'],training_tolerances=receipt['training_tolerances'],
        gauge_diagnostic=receipt['gauge_diagnostic'],initial_effective_operator_closure=receipt['initial_effective_operator_closure'],
        final_effective_operator_closure=receipt['final_effective_operator_closure'],initial_groups=initial_groups,final_groups=final_groups,
        CUDA_peak_allocated_bytes=receipt['peak_cuda_allocated_bytes'],CUDA_peak_reserved_bytes=receipt['peak_cuda_reserved_bytes'],
        elapsed_seconds=receipt['elapsed_seconds'],saved_tensor_statistics=receipt['saved_tensor_statistics'],
        limitation='No independent full-vocabulary forward/gradient replay or CAL-mean recomputation; source-bound worker checks supply those measurements.')


def audit(runs,backend,tokenizer=None):
    seal,cfg,public,gold=registered_sources();folder=Path(runs)/backend
    baseline,result=baseline_audit(folder,public,gold,cfg,tokenizer)
    cells=[]
    for passes in (1,2):
        cell=folder/f't{passes}_layer21_down_proj_rank8';path=cell/'receipt.json'
        if not path.exists():cells.append(dict(passes=passes,status='pending'));continue
        r=read(path)
        if r['status']!='completed':
            cells.append(dict(passes=passes,status=r['status'],optimizer_steps_completed=r['optimizer_steps_completed'],
                failure_type=r.get('failure_type'),fit_receipt_sha256=sha(path)))
        else:
            assert result['allow_fits'];cells.append(cell_audit(cell,backend,passes,cfg,public,gold,seal,baseline,tokenizer))
    return dict(status='passed',backend=backend,source_manifest_sha256=sha(ROOT/'SOURCE_MANIFEST.json'),auditor_source_sha256=sha(__file__),baseline=result,cells=cells,
        original_algebraic_gate_unchanged=True,formal_inferential_comparisons=0,BPD_donor_transfer_enabled=False,sealed_test_accessed=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',default='runs/nan_fit07');p.add_argument('--backend',choices=('bf16','nf4'),required=True)
    p.add_argument('--snapshot');p.add_argument('--out',required=True);a=p.parse_args();tokenizer=None
    if a.snapshot:
        cache=Path(a.runs)/'tokenizer_cache';cache.mkdir(parents=True,exist_ok=True)
        os.environ['HF_HOME']=str(cache.resolve());os.environ['HF_HUB_OFFLINE']='1';os.environ.pop('TRANSFORMERS_CACHE',None)
        from transformers import PreTrainedTokenizerFast
        snapshot=Path(a.snapshot);assert sha(snapshot/'tokenizer_config.json')==read(ROOT/'SOURCE_MANIFEST.json')['tokenizer_config_sha256']
        tokenizer=PreTrainedTokenizerFast(tokenizer_file=str(snapshot/'tokenizer.json'))
    result=audit(a.runs,a.backend,tokenizer);write(a.out,result)
    print({'status':result['status'],'backend':a.backend,'clean_gate_passed':result['baseline']['allow_fits'],
        'cells':[{k:r[k] for k in ('passes','status')} for r in result['cells']]})


if __name__=='__main__':main()
