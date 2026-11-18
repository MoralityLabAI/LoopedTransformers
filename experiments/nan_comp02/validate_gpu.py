"""Exact-dimension, single-row head and native B1 loop fixtures before 3B use."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time
import torch
from .head import OperatorHead
from experiments.nan_bpd01.backend import NativeLoop
from mechinterp.dark_fiber_gyms import arg03_nanbeige as NB

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--snapshot',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    start=time.perf_counter();torch.manual_seed(733);torch.cuda.set_per_process_memory_fraction(.75);torch.cuda.reset_peak_memory_stats()
    w=torch.randn(166144,3072,dtype=torch.bfloat16)*.01;h=torch.randn(1,3,3072,device='cuda',dtype=torch.bfloat16)
    alias={'I':[1,7],'N':[117,118],'S':[83000,83001],'R':[166142,166143]}
    fw=w.cuda();raw=torch.nn.functional.linear(h[:,-1],fw).float().flatten();reference=torch.log_softmax(raw,0)
    rec=OperatorHead(w,alias).score(h)
    selected_error=max(abs(rec['selected_logits'][str(i)]-float(raw[i])) for ids in alias.values() for i in ids)
    lp_error=max(abs((rec['selected_logits'][str(i)]-rec['full_log_normalizer'])-float(reference[i])) for ids in alias.values() for i in ids)
    assert selected_error==0 and lp_error<=2e-5 and rec['free_top_token_id']==int(raw.argmax()),(selected_error,lp_error)
    del fw,raw,w,h;torch.cuda.empty_cache()
    torch.manual_seed(314159)
    model,_=NB.build_nanbeige_model_and_loop(Path(a.snapshot),exec_device=torch.device('cpu'),dtype=torch.bfloat16,n_gpu_layers=2,random_init=True,
        config_overrides={'num_hidden_layers':2,'hidden_size':64,'intermediate_size':96,'num_attention_heads':4,'num_key_value_heads':2,'head_dim':16,'kv_channels':16})
    resident=copy.deepcopy(model).cuda();loop=NativeLoop(model,'cuda');ids=torch.randint(0,100,(1,19))
    e,ctx=loop.prepare(ids);one=loop.step(e,ctx,1);two=loop.step(one,ctx,2)
    expected=resident.model(input_ids=ids.cuda(),use_cache=False,return_dict=True).last_hidden_state
    bf_error=float((two.float()-expected.float()).abs().max());assert bf_error==0,bf_error
    quant=NativeLoop(copy.deepcopy(resident).cpu(),'cuda',quantized=True);quant.base.embed_tokens.cuda()
    e,ctx=quant.prepare(ids);one=quant.step(e,ctx,1);two=quant.step(one,ctx,2)
    expected=quant.base(input_ids=ids.cuda(),use_cache=False,return_dict=True).last_hidden_state
    nf_error=float((two.float()-expected.float()).abs().max());assert nf_error==0 and torch.isfinite(two).all(),nf_error
    result={'status':'passed','synthetic_vocabulary':166144,'synthetic_width':3072,'single_prompt_batch':1,'selected_logit_max_error':selected_error,
        'full_vocabulary_logprob_max_error':lp_error,'full_vocabulary_argmax_exact':True,'BF16_native_B1_two_loop_hidden_max_error':bf_error,
        'NF4_native_B1_two_loop_hidden_max_error':nf_error,'pretrained_weights_loaded':False,'sealed_test_accessed':False,
        'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved(),
        'elapsed_seconds':round(time.perf_counter()-start,3),'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_name('head.py')]}}
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);(out/'gpu_fixture.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result),flush=True)

if __name__=='__main__':main()
