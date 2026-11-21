"""Original FIT07 independent cell checks with the RECOMP09 source root.

No check is relaxed; resident-reference/recomputation assertions are additional.
"""
import math
from pathlib import Path
import numpy as np
from experiments.nan_fit07.io import read,sha
from mechinterp.nanbeige_bpd_gym.audit_fit import checkpoint,check_steps
from .diagnostic_audit import diagnostics_audit
from experiments.nan_fit07.pointwise import masks
ROOT=Path(__file__).parent


def cell_audit(folder,backend,passes,cfg,public,gold,seal,baseline,tokenizer=None):
    receipt=read(folder/'receipt.json');assert receipt['status']=='completed'
    assert receipt['gauge_diagnostic']['stable_at_1e_minus5']
    assert receipt['backend']==backend and receipt['passes']==passes and receipt['rank']==cfg['rank'] and receipt['module']==cfg['module']
    assert receipt['source_manifest_sha256']==sha(ROOT/'SOURCE_MANIFEST.json')
    assert receipt['initial_svd_file_sha256']==seal['initialization'][cfg['module']]['file_sha256']
    assert receipt['effective_operator_sha256']==seal['initialization'][cfg['module']]['effective_operator_sha256']
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
        assert bank['V'].shape==(cfg['input_width'],cfg['rank']) and bank['U'].shape==(cfg['rank'],cfg['output_width'])
        for field,shape in [('mean_comp_acts',(passes,cfg['rank'])),('mean_delta_out',(passes,cfg['output_width'])),('mean_write_rms',(passes,)),('calibrated_passes',(passes,))]:
            assert bank[field].shape==shape
        assert bank['calibrated_passes'].all()
        if name!='initial_checkpoint':
            for key in ('optimizer__factor_V__step','optimizer__factor_U__step','optimizer__ci_proj_in.weight__step','optimizer__ci_out_proj.weight__step'):
                assert bank[key].item()==8
    initial=banks['initial_checkpoint'];final=banks['final_checkpoint']
    svd_path=Path('runs/nan_gym05')/backend/'linear_baselines'/('layer21_'+cfg['module']+'.npz')
    assert sha(svd_path)==seal['initialization'][cfg['module']]['file_sha256']
    svd=checkpoint(svd_path)
    np.testing.assert_array_equal(initial['V'],svd['V'+str(cfg['rank'])]);np.testing.assert_array_equal(initial['U'],svd['U'+str(cfg['rank'])])
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
    contract=receipt['training_contract'];assert contract['c_per_module']=={cfg['site']:cfg['rank']}
    assert contract['n_passes']==passes and contract['n_steps']==8 and contract['ci']['n_blocks']==0
    assert contract['adversary']['inner_steps']==2 and contract['coeff_faith']==1e5 and contract['coeff_stoch']==contract['coeff_adv']==.5
    assert contract['ablation']['mode']==('zero' if passes==1 else 'pass_mean')
    assert contract['ablation']['sphere_rescale']==(passes==2)
    for field in ('initial_effective_operator_closure','final_effective_operator_closure'):
        assert receipt[field]['relative_l2']<=cfg['effective_operator_closure_relative_l2_tolerance']
    initial_groups=diagnostics_audit(read(folder/'initial_diagnostic_local.json'),public,gold,passes,['component_only'],cfg['rank'],tokenizer=tokenizer)
    final_groups=diagnostics_audit(read(folder/'diagnostic_local.json'),public,gold,passes,cfg['diagnostic_modes'],cfg['rank'],baseline,tokenizer)
    return dict(status='passed',backend=backend,passes=passes,fit_receipt_sha256=sha(folder/'receipt.json'),
        checkpoint_sha256=receipt['final_checkpoint']['sha256'],training=steps,factor_update_l2=receipt['factor_update_l2'],
        controller_update_l2=receipt['controller_update_l2'],training_tolerances=receipt['training_tolerances'],
        gauge_diagnostic=receipt['gauge_diagnostic'],initial_effective_operator_closure=receipt['initial_effective_operator_closure'],
        final_effective_operator_closure=receipt['final_effective_operator_closure'],initial_groups=initial_groups,final_groups=final_groups,
        CUDA_peak_allocated_bytes=receipt['peak_cuda_allocated_bytes'],CUDA_peak_reserved_bytes=receipt['peak_cuda_reserved_bytes'],
        elapsed_seconds=receipt['elapsed_seconds'],saved_tensor_statistics=receipt['saved_tensor_statistics'],
        limitation='No independent full-vocabulary forward/gradient replay or CAL-mean recomputation; source-bound worker checks supply those measurements.')
