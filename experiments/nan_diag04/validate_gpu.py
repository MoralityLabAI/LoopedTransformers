"""Random-weight grouped B1 streaming parity before pretrained inference."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time
import torch
from experiments.nan_bpd01.backend import NativeLoop
from experiments.nan_comp02.head import OperatorHead
from mechinterp.dark_fiber_gyms import arg03_nanbeige as NB
from .decode import grouped_states

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--snapshot',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    start=time.perf_counter();torch.manual_seed(2026111803);torch.cuda.set_per_process_memory_fraction(.75);torch.cuda.reset_peak_memory_stats()
    old=json.loads(Path('experiments/nan_comp02/receipts/gpu_fixture.json').read_text())
    assert old['status']=='passed' and old['full_vocabulary_argmax_exact'] and old['selected_logit_max_error']==0
    assert old['source_sha256']['head.py']==hashlib.sha256(Path('experiments/nan_comp02/head.py').read_bytes()).hexdigest()
    model,_=NB.build_nanbeige_model_and_loop(Path(a.snapshot),exec_device=torch.device('cpu'),dtype=torch.bfloat16,n_gpu_layers=2,random_init=True,
        config_overrides={'num_hidden_layers':2,'hidden_size':64,'intermediate_size':96,'num_attention_heads':4,'num_key_value_heads':2,'head_dim':16,'kv_channels':16})
    alias={'I':[1,7],'N':[117,118],'S':[83000,83001],'R':[166142,166143]};metrics={}
    for backend in ('bf16','nf4'):
        candidate=copy.deepcopy(model);loop=NativeLoop(candidate,'cuda',quantized=backend=='nf4');head=OperatorHead(candidate.lm_head.weight,alias)
        sequences=[torch.randint(0,100,(n,)).tolist() for n in ([19,21,17,25,23,15]*4)];hidden_error=score_error=0.;argmax_same=True
        for step in range(3):
            grouped=grouped_states(loop,sequences);next_ids=[]
            for ids,h in zip(sequences,grouped):
                e,ctx=loop.prepare(torch.tensor([ids]));h1=loop.step(e,ctx,1);serial=loop.step(h1,ctx,2)
                hidden_error=max(hidden_error,float((h.float()-serial.float()).abs().max()))
                x,y=head.score(h),head.score(serial);argmax_same&=x['free_top_token_id']==y['free_top_token_id']
                score_error=max(score_error,max(abs(x['selected_logits'][k]-y['selected_logits'][k]) for k in x['selected_logits']))
                next_ids.append(x['free_top_token_id'])
            sequences=[ids+[token] for ids,token in zip(sequences,next_ids)]
        assert hidden_error==score_error==0 and argmax_same
        metrics[backend]={'B1_prefixes':24,'autoregressive_steps':3,'hidden_max_error':hidden_error,'selected_logit_max_error':score_error,'full_vocab_argmax_exact':argmax_same}
        del loop,head,candidate,grouped,serial,h,h1,e;torch.cuda.empty_cache()
    result={'status':'passed','pretrained_weights_loaded':False,'sealed_test_accessed':False,'grouped_vs_serial':metrics,
        'reused_exact_dimension_head_fixture':old,'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved(),
        'elapsed_seconds':round(time.perf_counter()-start,3),'source_sha256':{n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in ('validate_gpu.py','decode.py','parser.py')}}
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);(out/'gpu_fixture.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n');print(json.dumps(result),flush=True)

if __name__=='__main__':main()
