"""Synthetic CUDA head and native-recurrence gates, before pretrained loading."""
import argparse
import copy
from hashlib import sha256
import json
from pathlib import Path
import sys
import time
import torch
from .head import ChunkedHead,program_scores,scores_at_temperatures
from .backend import NativeLoop
from mechinterp.dark_fiber_gyms import arg03_nanbeige as NB

def run(snapshot,out):
    snapshot=Path(snapshot);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    config=json.loads((Path(__file__).parent/'config.json').read_text(encoding='utf-8'))
    torch.cuda.set_per_process_memory_fraction(config['cuda_memory_fraction'])
    torch.manual_seed(511)
    start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    w=torch.randn((166144,3072),dtype=torch.bfloat16)*.01
    h=torch.randn((16,3,3072),dtype=torch.bfloat16,device='cuda:0')
    ids=dict(zip('INSR',(7,117,83000,166143)))
    full_w=w.to('cuda:0')
    raw=torch.nn.functional.linear(h,full_w).float()
    rec=ChunkedHead(w,ids,chunk_rows=config['vocabulary_chunk_rows']).score([h,h.clone()])
    op=torch.tensor(rec[0]['operator_logits'],device='cuda:0')
    op_error=float((op-raw[:,:,[ids[c] for c in 'INSR']]).abs().max())
    lp_error=0.;score_error=0.;z_error=0.
    for t in (.5,1.,2.):
        full_lp=torch.log_softmax(raw/t,-1)
        selected=torch.tensor(rec[0]['operator_logits'],device='cuda:0')/t-torch.tensor(rec[0]['log_normalizers'][f'{t:g}'],device='cuda:0',dtype=torch.float32)[...,None]
        lp_error=max(lp_error,float((selected-full_lp[:,:,[ids[c] for c in 'INSR']]).abs().max()))
        expected=program_scores(full_lp[:,:,[ids[c] for c in 'INSR']].cpu())
        got=scores_at_temperatures(rec[0],(t,)*3)
        score_error=max(score_error,max(abs(a-b) for a,b in zip(expected,got)))
        z_error=max(z_error,float((torch.tensor(rec[0]['log_normalizers'][f'{t:g}'],device='cuda:0')-torch.logsumexp(raw/t,-1)).abs().max()))
    assert op_error==0.,f'Operator logits not exact: {op_error}'
    assert lp_error<=config['chunk_logprob_tolerance'],lp_error
    assert score_error<=config['chunk_program_score_tolerance'],score_error
    assert rec[1]['full_vocab_max_logprob_error_vs_reference']==0.
    head_result={'fixture':'random BF16 head, exact checkpoint dimensions; no pretrained weights loaded',
                 'vocabulary':166144,'hidden':3072,'prefix_rows':16,'scored_positions':3,
                 'max_operator_logit_error':op_error,'max_operator_logprob_error':lp_error,
                 'max_log_normalizer_error':z_error,'max_64_program_score_error':score_error,
                 'identical_input_full_vocab_error':rec[1]['full_vocab_max_logprob_error_vs_reference']}
    del w,h,full_w,raw,op,rec;torch.cuda.empty_cache()
    # Actual pinned remote model code, but small RANDOM fixture weights.
    torch.manual_seed(271828)
    model,_=NB.build_nanbeige_model_and_loop(snapshot,exec_device=torch.device('cpu'),dtype=torch.bfloat16,
        n_gpu_layers=2,random_init=True,config_overrides={'num_hidden_layers':2,'hidden_size':64,
        'intermediate_size':96,'num_attention_heads':4,'num_key_value_heads':2,'head_dim':16,'kv_channels':16})
    original=copy.deepcopy(model).to('cuda:0')
    loop=NativeLoop(model,'cuda:0',quantized=False)
    tokens=torch.randint(0,100,(16,19))
    a,ctx=loop.prepare(tokens)
    a=loop.step(a,ctx,1);a=loop.step(a,ctx,2)
    expected=[]
    with torch.inference_mode():
        for row in tokens:
            expected.append(original.model(input_ids=row[None].to('cuda:0'),use_cache=False,return_dict=True).last_hidden_state)
    expected=torch.cat(expected)
    hidden_error=float((a.float()-expected.float()).abs().max())
    assert hidden_error==0.,hidden_error
    assert loop.model.lm_head.weight.device.type=='cpu'
    # A genuine NF4 effective-operator fixture, not a fake quantizer.
    import bitsandbytes as bnb
    from mechinterp.dark_fiber_gyms.q4_3050 import replace_linear4bit
    q=torch.nn.Linear(64,64,bias=False,dtype=torch.bfloat16)
    replace_linear4bit(torch.nn.ModuleList([q]),torch.device('cuda:0')) # CUDA path availability.
    nf=copy.deepcopy(original).cpu()
    nf_loop=NativeLoop(nf,'cuda:0',quantized=True)
    x,c=nf_loop.prepare(tokens);x=nf_loop.step(x,c,1);x=nf_loop.step(x,c,2)
    assert torch.isfinite(x).all() and nf_loop.model.lm_head.weight.device.type=='cpu'
    result={'status':'passed','head':head_result,'native_two_loop_hidden_error':hidden_error,
            'native_loop_fixture':'pinned remote code; random two-layer model; resident native vs streamed microbatch1',
            'real_nf4_decoder_fixture_finite':True,'all_output_heads_remained_cpu':True,
            'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),
            'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved(),
            'elapsed_seconds':round(time.perf_counter()-start,3),
            'torch':torch.__version__,'bitsandbytes':bnb.__version__,
            'source_sha256':{p.name:sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),Path(__file__).with_name('head.py'),Path(__file__).with_name('backend.py'))},
            'pretrained_checkpoint_loaded':False,'sealed_test_accessed':False}
    (out/'gpu_fixture.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--snapshot',required=True);p.add_argument('--out',required=True);a=p.parse_args();run(a.snapshot,a.out)
