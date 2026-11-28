"""Small actual NF4 dequantized-target and BPD residual check; no checkpoint."""
import argparse
import json
from pathlib import Path
import torch
from .components import effective_weight

def main():
    p=argparse.ArgumentParser();p.add_argument('--bpd-root',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    import bitsandbytes as bnb
    import mechinterp
    mechinterp.__path__.append(str(Path(a.bpd_root).resolve()/'mechinterp'))
    from mechinterp.bpd.components import TiedComponentLinear,PassMaskSchedule
    torch.manual_seed(1981)
    original=torch.nn.Linear(64,48,bias=False).bfloat16()
    quant=bnb.nn.Linear4bit(64,48,bias=False,compute_dtype=torch.bfloat16,compress_statistics=True,quant_type='nf4')
    quant.weight=bnb.nn.Params4bit(original.weight.detach().clone(),requires_grad=False,compress_statistics=True,quant_type='nf4')
    quant=quant.to('cuda');target=effective_weight(quant)
    x=torch.randn(16,3,64,device='cuda',dtype=torch.bfloat16)
    with torch.inference_mode():
        actual=quant(x);direct=torch.nn.functional.linear(x,target.to(x.dtype))
    assert torch.isfinite(actual).all()
    delta=float((target.cpu()-original.weight.float()).norm());assert delta>0
    residual={}
    for rank in [8,16]:
        linear=torch.nn.Linear(64,48,bias=False);linear.weight=torch.nn.Parameter(target.cpu(),requires_grad=False)
        wrapped=TiedComponentLinear(linear,rank,max_passes=2,init_svd=True);wrapped.mode='component';wrapped.schedule=PassMaskSchedule(torch.ones(16,2,3,rank))
        cpu_x=x.cpu().float();error=float((wrapped(cpu_x)-linear(cpu_x)).abs().max());assert error<1e-5
        residual[rank]=error
    result={'status':'passed','evidence_class':'random small NF4 engineering fixture; no pretrained dictionary recovery',
        'nf4_kernel_vs_effective_bf16_linear_max_error':float((actual.float()-direct.float()).abs().max()),
        'effective_vs_original_weight_l2':delta,'full_product_plus_residual_max_error_by_rank':residual,
        'uses_actual_quant_state':quant.weight.quant_state is not None,'sealed_test_accessed':False}
    Path(a.out).write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result))

if __name__=='__main__':main()
