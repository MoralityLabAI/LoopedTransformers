"""Synthetic CPU engineering of the actual pinned Nanbeige MLP class."""
import argparse
import ast
import os
from pathlib import Path
from types import SimpleNamespace
from experiments.nan_full26.common import THREADS,sha,write


def run(snapshot,out):
    for key in THREADS:os.environ[key]='1'
    from experiments.nan_ranksite13.resources import below_normal
    below_normal()
    import torch
    from torch import nn
    torch.set_num_threads(1);torch.manual_seed(2026112127)
    from .mlp_workspace27 import BoundedMLPWorkspace
    source=Path(snapshot)/'modeling_nanbeige.py';tree=ast.parse(source.read_text(encoding='utf-8'))
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='NanbeigeMLP')
    namespace={'nn':nn,'torch':torch,'F':torch.nn.functional,'ACT2FN':{'silu':torch.nn.SiLU()}}
    exec(compile(ast.Module(body=[cls],type_ignores=[]),'pinned_synthetic_MLP','exec'),namespace)
    MLP=namespace['NanbeigeMLP'];cases=[]
    for dtype in (torch.float32,torch.bfloat16):
        for batch,length,chunk in ((1,7,32),(2,65,32),(2,513,32),(1,7,256),(2,65,256),(2,513,256)):
            config=SimpleNamespace(hidden_size=11,intermediate_size=17,hidden_act='silu',pretraining_tp=1,mlp_bias=False)
            model=MLP(config).to(dtype).eval();model.requires_grad_(False)
            x=torch.randn(batch,length,11,dtype=dtype);native=model.forward
            gates=[];ups=[];down=[]
            handles=[model.gate_proj.register_forward_hook(lambda m,a,o:gates.append(tuple(o.shape))),
                model.up_proj.register_forward_hook(lambda m,a,o:ups.append(tuple(o.shape))),
                model.down_proj.register_forward_hook(lambda m,a,o:down.append(a[0].detach().clone()))]
            with torch.inference_mode():
                expected=native(x);reference_middle=down[0];gates.clear();ups.clear();down.clear()
                adapter=BoundedMLPWorkspace([model],chunk);actual=model(x)
                assert len(down)==1 and tuple(down[0].shape)==(batch,length,17)
                assert len(gates)==len(ups)==(length+chunk-1)//chunk
                assert max(v[1] for v in gates)<=chunk and max(v[1] for v in ups)<=chunk
                torch.testing.assert_close(down[0],reference_middle,rtol=.02,atol=.01)
                torch.testing.assert_close(actual,expected,rtol=.02,atol=.01)
                difference=float((actual.float()-expected.float()).abs().max())
                middle_difference=float((down[0].float()-reference_middle.float()).abs().max())
                # A down-projection intervention still sees one full [B,P,I]
                # tensor. This checks the actual hook boundary, not a proxy.
                count=[0]
                def edit(m,args,output):
                    count[0]+=1;result=output.clone();result[-1]+=1;return result
                h=model.down_proj.register_forward_hook(edit);edited=model(x);h.remove()
                assert count[0]==1
                if batch==2:assert torch.equal(edited[0],actual[0])
                assert torch.equal(edited[-1],actual[-1]+1)
                adapter.enabled=False;assert torch.equal(model(x),expected)
                adapter.close();assert torch.equal(model(x),expected)
            for h in handles:h.remove()
            cases.append(dict(dtype=str(dtype),batch=batch,tokens=length,gate_up_token_chunk=chunk,
                down_projection_calls=1,full_intermediate_shape=[batch,length,17],
                maximum_output_abs_difference=difference,maximum_intermediate_abs_difference=middle_difference,
                native_bit_exact=torch.equal(actual,expected),restoration_bit_exact=True,down_hook_reached_full_tensor=True))
    # Refuse gradients and tensor-parallel branches rather than silently
    # changing a training or unsupported numerical condition.
    model=MLP(SimpleNamespace(hidden_size=11,intermediate_size=17,hidden_act='silu',pretraining_tp=1,mlp_bias=False))
    a=BoundedMLPWorkspace([model],4)
    try:model(torch.randn(1,7,11))
    except RuntimeError as e:assert 'inference-only' in str(e)
    else:raise AssertionError('Gradient execution was accepted')
    a.close();model.config.pretraining_tp=2
    try:BoundedMLPWorkspace([model],4)
    except ValueError:pass
    else:raise AssertionError('Unsupported TP branch was accepted')
    result=dict(status='passed',synthetic_only=True,pretrained_model_forwards=0,GPU_allocations=0,
        cases=cases,physical_down_hook_once_per_MLP=True,gradient_and_unsupported_TP_rejected=True,
        source_model_file_sha256=sha(source),adapter_sha256=sha(Path(__file__).with_name('mlp_workspace27.py')),
        fixture_sha256=sha(__file__),CPU_threads=1,priority='BelowNormal',
        actual_GPU_capacity_unmeasured=True,new_execution_condition_required=True,
        frozen_CTX21_STORY23_FULL26_sources_or_criteria_changed=False,sealed_test_accessed=False)
    write(out,result);print(dict(status='passed',cases=len(cases),maximum_output_difference=max(c['maximum_output_abs_difference'] for c in cases),physical_down_hook_once=True,pretrained_forwards=0))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--snapshot',required=True);p.add_argument('--out',required=True);a=p.parse_args();run(a.snapshot,a.out)
