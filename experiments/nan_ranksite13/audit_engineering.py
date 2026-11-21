"""Independent CPU replay of all closed random rank/site pipeline artifacts."""
import argparse
import math
from pathlib import Path
import numpy as np
from experiments.nan_fit07.io import read,rows,sha,write
from experiments.nan_fit07.pointwise import masks
from mechinterp.nanbeige_bpd_gym.audit_fit import checkpoint,check_steps


def audit(runs):
    root=Path(__file__).parent;proof=read(runs/'engineering_receipt.json');cfg=read(root/'config.json')
    assert proof['status']=='passed' and not proof['pretrained_weights_loaded'] and not proof['sealed_test_accessed']
    for name,h in proof['source_sha256'].items():assert sha(root/name)==h,name
    assert proof['recomputation_source_sha256']==sha('mechinterp/nanbeige_bpd_gym/recomputed_native.py')
    public=rows('experiments/nan_fit07/fixtures/public.jsonl');main={r['id']:r for r in public if not r['copy_control']}
    results=[];targets=set()
    assert len(proof['results'])==len(cfg['conditions'])==6
    for condition,summary in zip(cfg['conditions'],proof['results']):
        assert summary['condition']==condition and summary['status']=='passed'
        rank=condition['rank'];passes=condition['passes'];name=f"t{passes}_layer21_{condition['module']}_rank{rank}"
        folder=runs/name;receipt=read(folder/'receipt.json');assert receipt['status']=='completed'
        assert receipt['rank']==rank and receipt['passes']==passes and receipt['module']==condition['module']
        assert receipt['optimizer_steps_completed']==8 and receipt['native_all_one_full_head_bit_exact'] and receipt['native_all_one_boundary_states_bit_exact']
        assert math.isfinite(receipt['pre_optimizer_gradient_gate_loss']) and receipt['gauge_diagnostic']['stable_at_1e_minus5']
        assert not receipt['BPD_donor_transfer_attempted'] and not receipt['sealed_test_accessed']
        targets.add(receipt['target_parameter_sha256_unchanged'])
        for artifact,h in receipt['diagnostic_artifact_sha256'].items():assert sha(folder/artifact)==h
        training=check_steps(read(folder/'steps_local.json'),public,cfg)
        banks={}
        for field in ('initial_checkpoint','training_EMA_checkpoint','final_checkpoint'):
            description=receipt[field];path=folder/description['file'];assert sha(path)==description['sha256']
            bank=checkpoint(path);banks[field]=bank
            assert bank['V'].shape==(128,rank) and bank['U'].shape==(rank,64)
            for key,shape in (('mean_comp_acts',(passes,rank)),('mean_delta_out',(passes,64)),('mean_write_rms',(passes,)),('calibrated_passes',(passes,))):
                assert bank[key].shape==shape
            assert bank['calibrated_passes'].all()
            if field!='initial_checkpoint':
                for key in ('optimizer__factor_V__step','optimizer__factor_U__step','optimizer__ci_proj_in.weight__step','optimizer__ci_out_proj.weight__step'):
                    assert bank[key].item()==8
        initial=banks['initial_checkpoint'];final=banks['final_checkpoint']
        for keys,field in ((('V','U'),'factor_update_l2'),([k for k in initial if k.startswith('ci__')],'controller_update_l2')):
            change=sum(np.sum((final[k].astype(np.float64)-initial[k])**2) for k in keys)**.5
            assert change>0 and math.isclose(change,receipt[field],rel_tol=2e-5,abs_tol=1e-8)
        path=folder/'CI_replay_local.npz';assert sha(path)==receipt['CI_replay_sha256'];replay=checkpoint(path)
        np.testing.assert_allclose(masks(replay['X'],final,receipt['final_checkpoint']['CI_config']),replay['masks'],rtol=2e-5,atol=1e-5)
        diagnostics=[]
        for artifact,modes in (('initial_diagnostic_local.json',['component_only']),('diagnostic_local.json',cfg['diagnostic_modes'])):
            data=read(folder/artifact);static=np.asarray(data['static_CAL_mask']);assert static.shape==(passes,rank) and np.isfinite(static).all() and static.min()>=0 and static.max()<=1
            seen=set()
            for record in data['records']:
                pair=record['id'],record['mode'];assert pair not in seen;seen.add(pair)
                source=main[record['id']];assert source['root']==record['root'] and source['split']==record['split']
                assert record['mode'] in modes and len(record['phases'])==passes
                for t,phase in enumerate(record['phases']):
                    assert phase['pass_index_zero_based']==t
                    assert all(math.isfinite(v) for v in phase.values() if isinstance(v,(float,int)))
                    if record['mode']=='all_one':
                        assert phase['actual_last_token_module_write_relative_l2']==phase['local_intervention_l2_at_actual_reached_input']==phase['reached_input_relative_l2']==0
                        assert abs(phase['full_vocabulary_KL'])<=1e-6
                    if record['mode']=='reached_state' and passes==1:assert phase['reached_vs_clean_mask_max_abs']==0
            assert seen=={(key,mode) for key in main for mode in modes}
            diagnostics.append(dict(artifact=artifact,records=len(seen),phase_readouts=len(seen)*passes,static_mask_shape=list(static.shape)))
        memory=receipt['saved_tensor_statistics'];assert 0<=memory['peak_added_cpu_bytes']<=memory['maximum_added_cpu_bytes']==64*1024**2
        if passes==2:assert memory['checkpointed_frozen_blocks']>0 and memory['recomputed_frozen_blocks']>0
        results.append(dict(module=condition['module'],rank=rank,passes=passes,status='passed',training=training,
            receipt_sha256=sha(folder/'receipt.json'),checkpoint_sha256=receipt['final_checkpoint']['sha256'],
            factor_update_l2=receipt['factor_update_l2'],controller_update_l2=receipt['controller_update_l2'],
            saved_tensor_CPU_peak_bytes=memory['peak_added_cpu_bytes'],diagnostic_coverage=diagnostics,pointwise_controller_replay_passed=True))
    assert len(targets)==1
    return dict(status='passed',evidence_class='independent closed-artifact random integration audit; not a pretrained/model-accuracy result',
        source_sha256=sha(Path(__file__)),engineering_receipt_sha256=sha(runs/'engineering_receipt.json'),cells=results,
        optimizer_steps_verified=sum(r['training']['optimizer_steps'] for r in results),pretrained_weights_loaded=False,
        target_digest_shared_across_all_cells=True,sealed_test_accessed=False,BPD_donor_transfer_attempted=False,formal_inferential_comparisons=0,
        limitations='No independent all-resident gradient/mean replay in this audit; worker assertions and prior gradient fixtures provide that engineering evidence. No task accuracy is graded.')


def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    result=audit(Path(a.runs));write(a.out,result);print({'status':result['status'],'optimizer_steps_verified':result['optimizer_steps_verified']})


if __name__=='__main__':main()
