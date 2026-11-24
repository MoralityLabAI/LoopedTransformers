"""Random-weight observation closure before native inference; no pretrained load."""
import argparse
import copy
from pathlib import Path
import time
import torch
from mechinterp.dark_fiber_gyms import arg03_nanbeige as NB
from experiments.nan_bpd01.backend import NativeLoop
from experiments.nan_comp02.head import OperatorHead
from experiments.nan_comp03.decode import grouped_states
from .common import read,write,sha
from .observe import TowerObserver
from .linear import decompose

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--snapshot',required=True);p.add_argument('--out',required=True);a=p.parse_args();start=time.perf_counter()
    torch.manual_seed(2026112005);torch.cuda.set_per_process_memory_fraction(.75);torch.cuda.reset_peak_memory_stats()
    model,_=NB.build_nanbeige_model_and_loop(Path(a.snapshot),exec_device=torch.device('cpu'),dtype=torch.bfloat16,n_gpu_layers=2,random_init=True,
        config_overrides={'num_hidden_layers':2,'hidden_size':64,'intermediate_size':96,'num_attention_heads':4,'num_key_value_heads':2,'head_dim':16,'kv_channels':16})
    alias={'I':[1,7],'N':[117,118],'S':[83000,83001],'R':[166142,166143]};metrics={}
    old=read(Path('experiments/nan_diag04/receipts/gpu_fixture.json'))
    assert old['status']=='passed' and old['reused_exact_dimension_head_fixture']['full_vocabulary_argmax_exact']
    for backend in ('bf16','nf4'):
        candidate=copy.deepcopy(model);loop=NativeLoop(candidate,'cuda',quantized=backend=='nf4');observer=TowerObserver(loop,[0,1]);head=OperatorHead(candidate.lm_head.weight,alias)
        seq={str(i):torch.randint(0,100,(length,)).tolist() for i,length in enumerate([19,21,17,25,23,15,20,24])}
        error=score_error=0.;same=True;counts=[]
        for step in range(3):
            expected=grouped_states(loop,list(seq.values()));actual,observed,one=observer.forward(seq,capture=True)
            counts.append(len(observed));assert len(observed)==8*2*2*5
            for ids,a1,b1 in zip(seq.values(),expected,actual):
                error=max(error,float((a1.float()-b1.float()).abs().max()))
                x,y=head.score(a1),head.score(b1);same&=x['free_top_token_id']==y['free_top_token_id']
                score_error=max(score_error,max(abs(x['selected_logits'][k]-y['selected_logits'][k]) for k in x['selected_logits']))
                ids.append(x['free_top_token_id'])
        assert error==score_error==0 and same
        metrics[backend]={'B1_contexts':8,'growing_prefix_steps':3,'hook_observations_per_step':counts,'hidden_max_error':error,'selected_logit_max_error':score_error,'full_vocab_argmax_exact':same}
        observer.close();del candidate,loop,observer,head,expected,actual,observed,one,a1,b1;torch.cuda.empty_cache()
    _,closure=decompose(torch.randn(64,96,device='cuda'),16,2026112005)
    root=Path(__file__).parent
    result={'status':'passed','pretrained_weights_loaded':False,'sealed_test_accessed':False,'observer_vs_uninstrumented':metrics,
        'small_effective_operator_closure':closure,'reused_full_head_fixture_sha256':sha(Path('experiments/nan_comp02/receipts/gpu_fixture.json')),
        'source_sha256':{n:sha(root/n) for n in ('observe.py','linear.py','validate_gpu.py')},
        'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved(),'elapsed_seconds':round(time.perf_counter()-start,3)}
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);write(out/'gpu_fixture.json',result);print(result)

if __name__=='__main__':main()
