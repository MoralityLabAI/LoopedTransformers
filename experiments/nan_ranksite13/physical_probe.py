"""Native local finite-write tapes; no labels, next-loop edits or donor transfer."""
from contextlib import contextmanager
import hashlib
from pathlib import Path
import numpy as np
import torch
from mechinterp.nanbeige_bpd_gym.o_proj_bridge import bridge_o_proj
from mechinterp.nanbeige_bpd_gym.recomputed_native import rollout_adapter
from experiments.nan_fit07.worker import save_arrays
from .common import read,sha,write
from .basis import control_factors


@contextmanager
def capture_clean(loop):
    data={};handles=[]
    def put(name,value):
        if loop.active_layer==21:
            entry=data.setdefault(loop.phase,{})
            if name in entry:raise RuntimeError('Duplicate physical capture')
            entry[name]=value.detach().clone()
    layer=loop.layers[21] if 21<loop.n_gpu_layers else loop.stage_layer
    handles.append(layer.input_layernorm.register_forward_pre_hook(lambda m,a:put('attention_residual',a[0])))
    handles.append(layer.self_attn.o_proj.register_forward_hook(lambda m,a,y:put('o_proj_output',y)))
    handles.append(layer.post_attention_layernorm.register_forward_pre_hook(lambda m,a:put('pre_MLP',a[0])))
    handles.append(layer.mlp.down_proj.register_forward_hook(lambda m,a,y:put('down_proj_output',y)))
    handles.append(layer.register_forward_hook(lambda m,a,y:put('residual_after_layer',y[0])))
    try:yield data
    finally:
        for h in handles:h.remove()


def physical_probe(loop,public,seq,cfg,folder,receipt,check_budget):
    """All main roots/bases/components/doses/signs, with full-prefix arithmetic."""
    out=folder/'physical_bridge';out.mkdir(exist_ok=False)
    with np.load(folder/receipt['final_checkpoint']['file'],allow_pickle=False) as z:V,U=z['V'],z['U']
    with np.load(folder/receipt['initial_checkpoint']['file'],allow_pickle=False) as z:initial=z['U']
    seed=cfg['Haar_seed_base']+cfg['rank']+100*cfg['passes']+(1000 if cfg['module']=='o_proj' else 0)
    rv,ru=control_factors(V,U,seed);save_arrays(out/'Haar_local.npz',dict(V=rv,U=ru))
    bases={'learned':U,'initial_SVD':initial,'matched_norm_Haar':ru}
    roots=[r for r in public if not r['copy_control']];assert len(roots)==cfg['physical_probe_main_contexts']
    adapter=rollout_adapter(loop,None,{cfg['site']:cfg['rank']},recompute=False)
    records=[]
    for ordinal,row in enumerate(roots):
        check_budget()
        with torch.no_grad(),capture_clean(loop) as c:adapter(seq[row['id']],2,0,None)
        assert set(c)=={0,1}
        arrays={};index=[];source_hashes={}
        for t in (0,1):
            native=c[t];layer=loop.layers[21] if 21<loop.n_gpu_layers else loop._load_stage(loop.layers[21])
            assert torch.equal(native['attention_residual']+native['o_proj_output'],native['pre_MLP'])
            assert torch.equal(native['pre_MLP']+native['down_proj_output'],native['residual_after_layer'])
            for name,value in native.items():
                arrays[f'base_{t}_{name}']=value[:,-1:].flatten().float().cpu().numpy()
                source_hashes[f'{t}_{name}']=hashlib.sha256(memoryview(value.float().cpu().contiguous().numpy())).hexdigest()
            native_output=native[cfg['module']+'_output'];norm=float(native_output[:,-1:].float().norm())
            for basis,values in bases.items():
                check_budget();directions=torch.as_tensor(values,device=loop.device,dtype=torch.float32)
                if cfg['module']=='o_proj':
                    bridge=bridge_o_proj(layer,native['attention_residual'],native['o_proj_output'],directions,
                        actual_pre_mlp_state=native['pre_MLP'],relative_doses=())
                    assert torch.equal(bridge['native_remainder'],native['residual_after_layer'])
                    jvp=bridge['physical_JVP_rows'];del bridge
                else:jvp=directions.to(native_output)
                arrays[f'{t}_{basis}_JVP']=jvp.float().cpu().numpy()
                arrays[f'{t}_{basis}_native_tangents']=directions.to(native_output).float().cpu().numpy()
                for component,direction in enumerate(directions):
                    tangent=torch.zeros_like(native_output);tangent[:,-1:]=direction.to(tangent).reshape(1,1,-1)
                    tangent_norm=float(tangent.float().norm())
                    for dose in cfg['relative_module_output_L2_doses']:
                        check_budget();scale=dose*max(norm,1e-12)/max(tangent_norm,1e-12)
                        for sign in cfg['physical_probe_signs']:
                            with torch.no_grad():
                                delta=sign*scale*tangent
                                changed_output=native_output+delta
                                if cfg['module']=='o_proj':
                                    reached=native['attention_residual']+changed_output
                                    mlp=layer.mlp(layer.post_attention_layernorm(reached))
                                    h=reached+mlp
                                else:
                                    reached=native['pre_MLP'];mlp=changed_output;h=reached+mlp
                            prefix=f'w{len(index):05d}'
                            for name,value in (('delta',delta),('module_output',changed_output),('pre_MLP',reached),('MLP_output',mlp),('layer_residual',h)):
                                arrays[f'{prefix}_{name}']=value[:,-1:].flatten().float().cpu().numpy()
                            index.append(dict(prefix=prefix,loop=t+1,basis=basis,component=component,relative_dose=dose,sign=sign,
                                scale=scale,zero_direction=tangent_norm==0,native_module_output_last_L2=norm,
                                actual_module_input_write_L2=float((changed_output[:,-1:].double()-native_output[:,-1:].double()).norm()),
                                actual_layer_residual_write_L2=float((h[:,-1:].double()-native['residual_after_layer'][:,-1:].double()).norm())))
                del jvp,directions,tangent,delta,changed_output,reached,mlp,h
            del native,layer,native_output
        expected=2*3*cfg['rank']*len(cfg['relative_module_output_L2_doses'])*2
        assert len(index)==expected
        path=out/f'case_{ordinal:03d}.npz';save_arrays(path,arrays)
        record={k:row[k] for k in ('id','root','split')}
        record.update(state_file=path.name,state_file_sha256=sha(path),prefix_tokens=len(seq[row['id']]),full_prefix_capture_sha256=source_hashes,
            all_baseline_native_additions_bit_exact=True,write_index=index)
        write(out/f'case_{ordinal:03d}.json',record);records.append(dict(record_sha256=sha(out/f'case_{ordinal:03d}.json'),state_sha256=sha(path)))
        with (out/'events.jsonl').open('a',encoding='utf-8') as f:f.write(__import__('json').dumps(dict(closed_contexts=ordinal+1))+'\n')
        print({'physical_bridge_closed_contexts':ordinal+1,'module':cfg['module'],'rank':cfg['rank'],'passes':cfg['passes']},flush=True)
        del c,arrays,index;torch.cuda.empty_cache()
    result=dict(status='completed',contexts=len(roots),native_loops=2,bases=list(bases),rank=cfg['rank'],
        local_signed_finite_writes=len(roots)*2*3*cfg['rank']*len(cfg['relative_module_output_L2_doses'])*2,
        Haar_file_sha256=sha(out/'Haar_local.npz'),records=records,
        source_manifest_sha256=receipt['source_manifest_sha256'],checkpoint_sha256=receipt['final_checkpoint']['sha256'],
        derivative_finite_equivalence_asserted=False,BPD_donor_transfer_attempted=False,sealed_test_accessed=False,
        scope='Full-prefix native local physical response before final norm; no future recurrent or task outcome measured')
    write(out/'receipt.json',result)
    return result
