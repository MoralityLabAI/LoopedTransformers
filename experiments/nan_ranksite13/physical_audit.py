"""Independent closed-tape finite-write audit/SAE response; no model loading."""
from collections import defaultdict
import math
import numpy as np
from mechinterp.nanbeige_bpd_gym.probe_towers import encoding,load_npz
from mechinterp.sae_causal_transfer.adapter import to_physical_dictionary
from .common import read,sha
from .basis import control_factors


def bf16(values):
    """Round finite float32 to BF16, ties to even, retaining float32 storage."""
    a=np.asarray(values,dtype=np.float32)
    if not np.isfinite(a).all():raise ValueError('Finite input required')
    bits=a.view(np.uint32)
    rounded=(bits+np.uint32(0x7fff)+((bits>>np.uint32(16))&np.uint32(1)))&np.uint32(0xffff0000)
    return rounded.view(np.float32)


def physical_audit(folder,public,cfg,fit_receipt,sae):
    out=folder/'physical_bridge';receipt=read(out/'receipt.json')
    assert receipt['status']=='completed' and receipt['checkpoint_sha256']==fit_receipt['final_checkpoint']['sha256']
    assert receipt['source_manifest_sha256']==fit_receipt['source_manifest_sha256']
    assert not receipt['derivative_finite_equivalence_asserted'] and not receipt['BPD_donor_transfer_attempted'] and not receipt['sealed_test_accessed']
    sources=[r for r in public if not r['copy_control']];assert len(sources)==receipt['contexts']==cfg['physical_probe_main_contexts']
    final=load_npz(folder/fit_receipt['final_checkpoint']['file']);initial=load_npz(folder/fit_receipt['initial_checkpoint']['file'])
    path=out/'Haar_local.npz';assert sha(path)==receipt['Haar_file_sha256'];haar=load_npz(path)
    seed=cfg['Haar_seed_base']+cfg['rank']+100*cfg['passes']+(1000 if cfg['module']=='o_proj' else 0)
    rv,ru=control_factors(final['V'],final['U'],seed)
    np.testing.assert_allclose(haar['V'],rv,rtol=1e-5,atol=1e-7);np.testing.assert_allclose(haar['U'],ru,rtol=1e-5,atol=1e-7)
    bases={'learned':final['U'],'initial_SVD':initial['U'],'matched_norm_Haar':haar['U']}
    E,_,D=to_physical_dictionary(sae['encoder'],sae['bias'],sae['decoder'],sae['mean'],sae['scale'])
    assert E.shape==(128,cfg['output_width'])
    encoder_norm=np.linalg.norm(E,axis=1);decoder_norm=np.linalg.norm(D,axis=1)
    groups=defaultdict(list);centers=defaultdict(list);checked=0
    for ordinal,source in enumerate(sources):
        record_path=out/f'case_{ordinal:03d}.json';r=read(record_path);binding=receipt['records'][ordinal]
        assert sha(record_path)==binding['record_sha256'] and all(r[k]==source[k] for k in ('id','root','split'))
        path=out/r['state_file'];assert sha(path)==r['state_file_sha256']==binding['state_sha256']
        assert r['all_baseline_native_additions_bit_exact'] and r['prefix_tokens']>0
        expected={(t,b,c,d,s) for t in (1,2) for b in bases for c in range(cfg['rank']) for d in cfg['relative_module_output_L2_doses'] for s in cfg['physical_probe_signs']}
        seen=set();arrays=load_npz(path);pairs=defaultdict(dict);allkeys=set()
        for t in (0,1):
            for name in ('attention_residual','o_proj_output','pre_MLP','down_proj_output','residual_after_layer'):
                k=f'base_{t}_{name}';allkeys.add(k);a=arrays[k]
                assert a.shape==(cfg['output_width'],) and a.dtype==np.float32 and np.isfinite(a).all()
            np.testing.assert_array_equal(bf16(arrays[f'base_{t}_attention_residual']+arrays[f'base_{t}_o_proj_output']),arrays[f'base_{t}_pre_MLP'])
            np.testing.assert_array_equal(bf16(arrays[f'base_{t}_pre_MLP']+arrays[f'base_{t}_down_proj_output']),arrays[f'base_{t}_residual_after_layer'])
            for basis,U in bases.items():
                for field in ('JVP','native_tangents'):
                    k=f'{t}_{basis}_{field}';allkeys.add(k)
                    assert arrays[k].shape==(cfg['rank'],cfg['output_width']) and np.isfinite(arrays[k]).all()
                np.testing.assert_array_equal(arrays[f'{t}_{basis}_native_tangents'],bf16(U))
                if cfg['module']=='down_proj':np.testing.assert_array_equal(arrays[f'{t}_{basis}_JVP'],bf16(U))
        changed=np.stack([arrays[item['prefix']+'_layer_residual'] for item in r['write_index']])
        base=np.stack([arrays[f'base_{t}_residual_after_layer'] for t in (0,1)])
        z_changed=encoding(changed,sae,4)[0];z_base=encoding(base,sae,4)[0]
        displacement=changed.astype(np.float64)-base[[item['loop']-1 for item in r['write_index']]]
        effect_norm=np.linalg.norm(displacement,axis=1)
        sensitivity=displacement@E.T;decoder_dot=displacement@D.T
        denominator=effect_norm[:,None]*encoder_norm[None,:]
        cosine_e=np.divide(sensitivity,denominator,out=np.zeros_like(sensitivity),where=denominator>0)
        denominator=effect_norm[:,None]*decoder_norm[None,:]
        cosine_d=np.divide(decoder_dot,denominator,out=np.zeros_like(decoder_dot),where=denominator>0)
        for write_number,item in enumerate(r['write_index']):
            tag=tuple(item[k] for k in ('loop','basis','component','relative_dose','sign'));assert tag in expected and tag not in seen;seen.add(tag)
            t=item['loop']-1;prefix=item['prefix'];v={}
            for field in ('delta','module_output','pre_MLP','MLP_output','layer_residual'):
                k=prefix+'_'+field;allkeys.add(k);v[field]=arrays[k]
                assert v[field].shape==(cfg['output_width'],) and v[field].dtype==np.float32 and np.isfinite(v[field]).all()
            original=arrays[f'base_{t}_{cfg["module"]}_output'];base_h=arrays[f'base_{t}_residual_after_layer']
            np.testing.assert_array_equal(bf16(original+v['delta']),v['module_output'])
            if cfg['module']=='o_proj':np.testing.assert_array_equal(bf16(arrays[f'base_{t}_attention_residual']+v['module_output']),v['pre_MLP'])
            else:
                np.testing.assert_array_equal(v['pre_MLP'],arrays[f'base_{t}_pre_MLP'])
                np.testing.assert_array_equal(v['MLP_output'],v['module_output'])
            np.testing.assert_array_equal(bf16(v['pre_MLP']+v['MLP_output']),v['layer_residual'])
            delta=v['layer_residual'].astype(np.float64)-base_h
            dose=float(np.linalg.norm(v['module_output'].astype(np.float64)-original));effect=float(np.linalg.norm(delta))
            assert math.isclose(dose,item['actual_module_input_write_L2'],rel_tol=1e-6,abs_tol=1e-8)
            assert math.isclose(effect,item['actual_layer_residual_write_L2'],rel_tol=1e-6,abs_tol=1e-8)
            assert math.isfinite(item['scale']) and item['scale']>0
            pairkey=tag[:-1];pairs[pairkey][item['sign']]=(v['layer_residual'].astype(np.float64),item['scale'])
            groups[source['split'],item['loop'],item['basis'],item['relative_dose'],item['sign']].append(dict(root=source['root'],
                component=item['component'],actual_module_write_L2=dose,actual_layer_write_L2=effect,zero_actual_module_write=int(dose==0),
                zero_actual_layer_write=int(effect==0),encoder_cosine=cosine_e[write_number],decoder_cosine=cosine_d[write_number],
                encoder_preactivation_delta=sensitivity[write_number],SAE_code_delta=z_changed[write_number]-z_base[t]))
            checked+=1
        assert seen==expected and set(arrays)==allkeys
        for (t,b,c,d),signed in pairs.items():
            assert set(signed)=={-1,1};plus,scale=signed[1];minus,other=signed[-1];assert scale==other
            j=arrays[f'{t-1}_{b}_JVP'][c].astype(np.float64);central=(plus-minus)/(2*scale)
            error=float(np.linalg.norm(central-j)/max(np.linalg.norm(j),1e-12))
            centers[source['split'],t,b,d].append(dict(root=source['root'],component=c,relative_l2=error))
    expected_count=len(sources)*2*3*cfg['rank']*3*2
    assert checked==receipt['local_signed_finite_writes']==expected_count
    summary=[]
    for (split,t,b,d,s),values in sorted(groups.items()):
        byroot=defaultdict(list)
        for r in values:byroot[r['root']].append(r)
        macro=lambda field:np.mean([np.mean([r[field] for r in rr],axis=0) for rr in byroot.values()],axis=0)
        stats={field:(macro(field).tolist() if isinstance(values[0][field],np.ndarray) else float(macro(field))) for field in values[0] if field not in ('root','component')}
        summary.append(dict(split=split,loop=t,basis=b,relative_dose=d,sign=s,source_roots=len(byroot),component_writes=len(values),
            out_of_fit_depth=t>cfg['passes'],root_macro_over_all_components=stats))
    jacobian=[]
    for (split,t,b,d),values in sorted(centers.items()):
        byroot=defaultdict(list)
        for r in values:byroot[r['root']].append(r['relative_l2'])
        jacobian.append(dict(split=split,loop=t,basis=b,relative_dose=d,source_roots=len(byroot),
            centered_finite_vs_JVP_relative_l2_root_macro=float(np.mean([np.mean(v) for v in byroot.values()])),
            centered_finite_vs_JVP_relative_l2_max=max(r['relative_l2'] for r in values)))
    return dict(status='passed',physical_receipt_sha256=sha(out/'receipt.json'),signed_finite_writes_verified=checked,
        contexts=len(sources),geometry_groups=summary,derivative_diagnostics=jacobian,
        privacy='all source IDs, per-record vectors/codes and component-wise tapes stay local; every feature and group retained in root/component macro summaries',
        limitation='CPU verifies all archived BF16 additions/doses/coverage. Native MLP outputs/JVPs are source-bound worker measurements; no independent 3B MLP replay here.',
        causal_claim=False,derivative_finite_equivalence_asserted=False)
