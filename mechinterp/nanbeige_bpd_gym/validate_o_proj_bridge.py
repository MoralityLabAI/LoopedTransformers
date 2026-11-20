"""Random native full-forward injection check of the local o_proj bridge."""
import argparse
import os
from pathlib import Path
import time
import traceback
import torch
from mechinterp.dark_fiber_gyms import arg03_nanbeige as NB
from experiments.nan_bpd01.backend import NativeLoop
from experiments.nan_recomp09.common import sha,write,target_digest
from .o_proj_bridge import bridge_o_proj,local_remainder


def main():
    p=argparse.ArgumentParser();p.add_argument('--snapshot',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False);cache=out/'HF_cache';cache.mkdir()
    os.environ['HF_HOME']=str(cache.resolve());os.environ['HF_HUB_OFFLINE']='1';os.environ.pop('TRANSFORMERS_CACHE',None)
    if os.name=='nt':
        import ctypes
        k=ctypes.WinDLL('kernel32',use_last_error=True);k.GetCurrentProcess.restype=ctypes.c_void_p;k.SetPriorityClass.argtypes=[ctypes.c_void_p,ctypes.c_uint]
        if not k.SetPriorityClass(k.GetCurrentProcess(),0x4000):raise ctypes.WinError(ctypes.get_last_error())
    torch.set_num_threads(1);torch.manual_seed(2026112013);torch.cuda.set_per_process_memory_fraction(.05);torch.cuda.reset_peak_memory_stats()
    started=time.monotonic();results=[];finite=[];status='failed';failure=None
    try:
        model,_=NB.build_nanbeige_model_and_loop(Path(a.snapshot),exec_device=torch.device('cpu'),dtype=torch.bfloat16,n_gpu_layers=22,random_init=True,
            config_overrides={'vocab_size':4096,'num_hidden_layers':22,'hidden_size':64,'intermediate_size':128,'num_attention_heads':8,'num_key_value_heads':2,'head_dim':16,'kv_channels':16})
        loop=NativeLoop(model,'cuda:0',quantized=False);loop.stage_layer.requires_grad_(False).eval();before=target_digest(loop)
        rng=torch.Generator().manual_seed(2026112013);ids=torch.randint(0,100,(1,19),generator=rng)
        directions=torch.randn((16,64),generator=rng).to(loop.device)
        def budget():
            if time.monotonic()-started>600:raise RuntimeError('Registered finite random bridge budget exceeded')
        def forward(*,phase=None,delta=None):
            records={};handles=[];stage=loop.stage_layer
            def save(field,value):
                if loop.active_layer==21:records.setdefault(loop.phase,{})[field]=value.detach().clone()
            handles.append(stage.input_layernorm.register_forward_pre_hook(lambda m,args:save('attention_residual',args[0])))
            def inject(m,args,value):
                if loop.active_layer==21 and loop.phase==phase:
                    changed=value.clone();changed[:,-1:]=changed[:,-1:]+delta;return changed
            handles.append(stage.self_attn.o_proj.register_forward_hook(inject))
            handles.append(stage.self_attn.o_proj.register_forward_hook(lambda m,args,value:save('attention_output',value)))
            handles.append(stage.post_attention_layernorm.register_forward_pre_hook(lambda m,args:save('pre_mlp',args[0])))
            handles.append(stage.register_forward_hook(lambda m,args,value:save('layer_residual',value[0])))
            try:
                with torch.inference_mode():
                    h,ctx=loop.prepare(ids)
                    for t in (1,2):h=loop.step(h,ctx,t)
            finally:
                for handle in handles:handle.remove()
            return records
        clean=forward()
        for phase in (1,2):
            budget();c=clean[phase];layer=loop._load_stage(loop.layers[21])
            bridge=bridge_o_proj(layer,c['attention_residual'],c['attention_output'],directions,actual_pre_mlp_state=c['pre_mlp'])
            assert torch.equal(bridge['native_remainder'],c['layer_residual'])
            for item in bridge['finite_writes']:
                index=item['direction'];tangent=torch.zeros_like(c['attention_output']);tangent[:,-1:]=directions[index].to(tangent)
                for sign in (-1,1):
                    budget();delta=sign*item['scale']*tangent[:,-1:]
                    layer=loop._load_stage(loop.layers[21])
                    # Preserve the physical operation order: alter o_proj output, then add its residual.
                    with torch.no_grad():expected=local_remainder(layer,c['attention_residual']+(c['attention_output']+sign*item['scale']*tangent))
                    actual=forward(phase=phase,delta=delta)[phase]['layer_residual'];exact=torch.equal(actual,expected)
                    results.append(dict(loop=phase,direction=index,relative_dose=item['requested_relative_input_dose'],sign=sign,
                        full_native_layer_output_bit_exact=exact,full_native_layer_output_max_abs=float((actual.float()-expected.float()).abs().max())))
                    assert exact,'The local bridge did not reproduce physical native o_proj injection'
                finite.append(dict(loop=phase,**item))
            jv=bridge['physical_JVP_rows'].float();tangent=bridge['native_tangent_rows'].float()
            write(out/f'phase{phase}_summary.json',dict(prefill_remainder_bit_exact=True,JVP_rows=len(jv),
                identity_bridge_relative_l2=float((jv-tangent).norm()/tangent.norm().clamp_min(1e-12)),finite_native_comparisons=len(results)))
        assert len(results)==192 and len(finite)==96 and target_digest(loop)==before
        status='passed'
    except Exception as error:
        failure=dict(type=type(error).__name__,message=str(error));traceback.print_exc()
    root=Path(__file__).parent
    receipt=dict(status=status,failure=failure,full_forward_signed_comparisons=results,centered_finite_write_diagnostics=finite,
        source_sha256={name:sha(root/name) for name in ('o_proj_bridge.py','validate_o_proj_bridge.py','O_PROJ_BRIDGE_ENGINEERING_PLAN.md')},
        pretrained_weights_loaded=False,target_parameters_unchanged=status=='passed',sealed_test_accessed=False,BPD_donor_transfer_attempted=False,
        evidence_class='random native local coordinate/JVP/finite-write engineering; no pretrained or causal evidence',
        CPU_threads=torch.get_num_threads(),formal_inferential_comparisons=0,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),elapsed_seconds=time.monotonic()-started)
    write(out/'receipt.json',receipt);print({'status':status,'full_forward_signed_comparisons':len(results),'failure':failure},flush=True)
    if status!='passed':raise SystemExit(2)


if __name__=='__main__':main()
