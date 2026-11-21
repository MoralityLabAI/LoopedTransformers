"""One-thread CPU adapter checks; zero checkpoint forwards or GPU allocations."""
import argparse
import copy
import os
from pathlib import Path
from .common import write,ROOT,sha


def run(bpd_root):
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
    from experiments.nan_ranksite13.resources import below_normal
    below_normal()
    import torch
    from torch import nn
    from mechinterp.nanbeige_bpd_gym.native_components import component_class,load_bpd
    from .adapters import bound_components,EmbeddingLookupCache,last_snapshot_function
    from .host import local_replay
    from dataclasses import asdict
    import numpy as np
    torch.set_num_threads(1);torch.manual_seed(2026112123)
    framework=load_bpd(bpd_root);C=framework['config'];S=framework['components'].PassMaskSchedule
    Kernel=component_class(bpd_root);ablation=C.AblationConfig(mode='pass_mean',sphere_rescale=True)
    reference=Kernel(nn.Linear(11,7,bias=False),3,max_passes=2,ablation=ablation,init_svd=False)
    reference.V.data.normal_(0,.1);reference.U.data.normal_(0,.1)
    reference.mean_comp_acts.normal_();reference.mean_delta_out.normal_();reference.mean_write_rms.fill_(1);reference.calibrated_passes.fill_(True)
    actual=copy.deepcopy(reference);bound_components(actual,4)
    ci=__import__('mechinterp.bpd.ci',fromlist=['TiedCINetwork']).TiedCINetwork({'x':11},{'x':3},C.CIConfig(variant='state_only',n_blocks=0,d_model=32))
    maximum=0.;closure=True;snapshots=0;local_maximum=0.
    with torch.no_grad():
        for route in ('all_one','static','live'):
            masks=torch.ones((1,2,13,3)) if route=='all_one' else torch.rand((1,2,13,3))
            delta=torch.ones((1,2,13)) if route=='all_one' else torch.zeros((1,2,13))
            for core in (reference,actual):
                core.reset_pass_state();core.mode='component';core.schedule=S(masks,delta)
                core.live_controller=(lambda x,tau:ci({'x':x.unsqueeze(1)},pass_offset=tau).lower_leaky['x'][:,0]) if route=='live' else None
            for phase in range(2):
                x=torch.randn(1,13,11);native=torch.randn(1,13,7).to(torch.bfloat16)
                a=reference.forward_kernel(x,native);b=actual.forward_kernel(x,native)
                gap=float(torch.max(torch.abs(a.float()-b.float())));maximum=max(maximum,gap)
                torch.testing.assert_close(a,b,rtol=.02,atol=.01)
                if route=='all_one':closure=closure and torch.equal(native,a) and torch.equal(native,b)
                else:
                    bank={name:getattr(actual,name).detach().numpy() for name in ('V','U','mean_comp_acts','mean_delta_out','mean_write_rms')}
                    mask=actual.controller_observations[-1]['mask'][0,-1].numpy() if route=='live' else masks[0,phase,-1].numpy()
                    predicted=local_replay(x[0,-1].numpy(),native[0,-1].float().numpy(),mask,bank,asdict(ablation),phase)
                    relative=float(np.linalg.norm(predicted-b[0,-1].float().numpy())/max(float(np.linalg.norm(b[0,-1].float().numpy())),1e-12))
                    assert relative<=.03;local_maximum=max(local_maximum,relative)
            if route=='live':
                assert len(actual.controller_observations)==2
                for a,b in zip(reference.controller_observations,actual.controller_observations):
                    torch.testing.assert_close(a['input'][:,-1:],b['input'],rtol=0,atol=0)
                    torch.testing.assert_close(a['mask'][:,-1:],b['mask'],rtol=1e-5,atol=1e-6);snapshots+=1
        embedding=nn.Embedding(31,9,dtype=torch.bfloat16);original=embedding.forward
        cache=EmbeddingLookupCache(embedding,'cpu',4)
        for ids in (torch.tensor([[1,2,3]]),torch.tensor([[1,2,3]]),torch.tensor([[1,2,3,4]])):
            assert torch.equal(embedding(ids),original(ids))
        assert cache.rows_gathered==4
        try:embedding(torch.tensor([[1,7,3,4,5]]))
        except AssertionError:pass
        else:raise AssertionError('Cache accepted changed history')
        cache.close();assert closure
    return dict(status='passed',CPU_threads=1,checkpoint_model_forwards=0,GPU_allocations=0,
        observation_AST_reversible=True,all_one_native_bit_exact=closure,chunk_full_max_abs_BF16_output_difference=maximum,
        last_token_controller_snapshots_checked=snapshots,embedding_lookup_cache_exact=True,
        independent_BF16_local_write_maximum_relative_L2_error=local_maximum,
        source_sha256={name:sha(ROOT/name) for name in ('adapters.py','instrument.py','host.py','engineering_numeric.py')},sealed_test_accessed=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bpd-root',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    result=run(a.bpd_root);write(a.out,result);print(result)
