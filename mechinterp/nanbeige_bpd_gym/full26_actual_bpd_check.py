"""Post-freeze synthetic check of FULL26 dispatch using the real pinned BPD.

This adds evidence without changing any frozen execution source or criterion.
No Nanbeige weights, source gold, original test or CUDA device is opened.
"""
import argparse
import ast
from dataclasses import asdict
import os
from pathlib import Path
from types import MethodType,SimpleNamespace
from experiments.nan_full26.common import ROOT,THREADS,read,sha,verify,write


def run(bpd_root,out):
    for key in THREADS:os.environ[key]='1'
    from experiments.nan_ranksite13.resources import below_normal
    below_normal();seal=verify()
    import numpy as np
    import torch
    from torch import nn
    from mechinterp.nanbeige_bpd_gym.native_components import component_class,load_bpd
    from experiments.nan_story23.adapters import bound_components
    from experiments.nan_story23.instrument import Instrument
    from experiments.nan_story23.host import local_replay
    from experiments.nan_fit07.pointwise import masks as independent_masks
    from experiments.nan_full26.batch import RowInstrument
    torch.set_num_threads(1);torch.manual_seed(2026112126)
    framework=load_bpd(bpd_root)
    from mechinterp.bpd.ci import TiedCINetwork
    config=framework['config'];Kernel=component_class(bpd_root)
    ablation=config.AblationConfig(mode='pass_mean',sphere_rescale=True)
    core=Kernel(nn.Linear(11,7,bias=False,dtype=torch.bfloat16),8,max_passes=2,ablation=ablation,init_svd=False)
    with torch.no_grad():
        core.V.normal_(0,.1);core.U.normal_(0,.1);core.mean_comp_acts.normal_();core.mean_delta_out.normal_()
        core.mean_write_rms.copy_(torch.tensor([.7,1.1]));core.calibrated_passes.fill_(True)
    controller=TiedCINetwork({'x':11},{'x':8},config.CIConfig(variant='state_only',n_blocks=0,d_model=32)).eval()
    class Model(framework['components'].BPDModel):
        def __init__(self):
            nn.Module.__init__(self);self.module_order=['x'];self.wrappers={'x':core};self.c_per_module={'x':8}
    base=Instrument.__new__(Instrument);base.core=core;base.model=Model();base.loop=SimpleNamespace(device=torch.device('cpu'))
    base.cfg={'learned_site':'x'};base.ci=controller;base.Schedule=framework['components'].PassMaskSchedule
    base.static=torch.linspace(.05,.95,16).reshape(2,8)
    row=RowInstrument.__new__(RowInstrument);row.base=base;row.core=core
    bound_components(core,4);row.original=core.forward_kernel
    # Execute the exact frozen dispatch closure, binding only its owner. This
    # avoids the real constructor's model/same-target checkpoint requirements.
    tree=ast.parse((ROOT/'batch.py').read_text(encoding='utf-8'))
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='RowInstrument')
    constructor=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
    fn=next(n for n in constructor.body if isinstance(n,ast.FunctionDef) and n.name=='forward')
    namespace={'owner':row};exec(compile(ast.Module(body=[fn],type_ignores=[]),'frozen_full26_row_dispatch','exec'),namespace)
    core.forward_kernel=MethodType(namespace['forward'],core)
    bank={name:getattr(core,name).detach().numpy() for name in ('V','U','mean_comp_acts','mean_delta_out','mean_write_rms')}
    bank.update({'ci__'+k:v.detach().numpy() for k,v in controller.state_dict().items()})
    rows=[];mask_max=write_max=0.;nonzero_edits=0
    with torch.inference_mode():
        for lengths in ([13,9],[3,13],[13]):
            B=len(lengths);P=max(lengths)+3;edited=B-1
            for route in ('all_one','component_only','static_CAL_mean','reached_state'):
                modes=['native']*(B-1)+[route]
                with row.context(modes,lengths):
                    for phase in range(2):
                        x=torch.randn(B,P,11,dtype=torch.bfloat16)
                        native=torch.nn.functional.linear(x,core.W_target)
                        actual=core.forward_kernel(x,native);n=lengths[edited]
                        if B==2:assert torch.equal(actual[0],native[0]),'Clean A row changed'
                        assert torch.equal(actual[edited,n:],native[edited,n:]),'Right padding changed'
                        assert core.calls==phase+1,'Token chunks advanced the physical pass counter'
                        if route=='all_one':assert torch.equal(actual,native),'All-one closure failed'
                        else:
                            mask=np.asarray(row.mask_by_phase[-1])
                            expected=independent_masks(x[edited,n-1].float().numpy()[None],bank,asdict(controller.cfg))[0] if route=='reached_state' else base.static[phase].numpy() if route=='static_CAL_mean' else np.ones(8)
                            gap=float(np.max(np.abs(mask-expected)));mask_max=max(mask_max,gap)
                            np.testing.assert_allclose(mask,expected,rtol=2e-5,atol=1e-5)
                            predicted=local_replay(x[edited,n-1].float().numpy(),native[edited,n-1].float().numpy(),mask,bank,asdict(ablation),phase)
                            y=actual[edited,n-1].float().numpy();error=float(np.linalg.norm(predicted-y)/max(float(np.linalg.norm(y)),1e-12))
                            assert error<=.03;write_max=max(write_max,error)
                            nonzero_edits+=int(not torch.equal(actual[edited,:n],native[edited,:n]))
                        assert len(row.raw_by_phase)==phase+1
                        assert torch.equal(row.raw_by_phase[-1][edited],native[edited,n-1])
                        rows.append(dict(lengths=lengths,padded_length=P,route=route,phase=phase+1,
                            clean_row_unchanged=True,padding_unchanged=True,physical_counter=core.calls))
                assert core.mode=='target' and core.schedule is None
    result=dict(status='passed',scope='synthetic CPU tensors through exact frozen row dispatch and real pinned BPD/CI/context/chunk adapters',
        frozen_source_manifest_sha256=sha(ROOT/'SOURCE_MANIFEST.json'),source_bindings_verified=len(seal['files']),
        checked_physical_invocations=len(rows),rows=rows,nonzero_reduced_route_edits=nonzero_edits,
        maximum_independent_mask_abs_error=mask_max,maximum_independent_local_write_relative_L2_error=write_max,
        independent_mask_rtol=2e-5,independent_mask_atol=1e-5,independent_write_relative_L2_tolerance=.03,
        all_one_native_bit_exact=True,unchanged_clean_A=True,unchanged_right_padding=True,
        B1_edited_tail_tested=True,one_counter_increment_per_physical_call=True,
        CPU_threads=1,priority='BelowNormal',pretrained_model_forwards=0,GPU_allocations=0,sealed_test_accessed=False,
        source_sha256={str(ROOT/'batch.py'):sha(ROOT/'batch.py'),__file__:sha(__file__)})
    write(out,result)
    print({k:result[k] for k in ('status','checked_physical_invocations','nonzero_reduced_route_edits','maximum_independent_mask_abs_error','maximum_independent_local_write_relative_L2_error')})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bpd-root',required=True);p.add_argument('--out',required=True);a=p.parse_args();run(a.bpd_root,a.out)
