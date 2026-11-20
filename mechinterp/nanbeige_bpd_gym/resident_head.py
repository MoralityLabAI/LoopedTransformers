"""Place the unchanged BF16 vocabulary head on CUDA for new conditions only.

The existing chunked scorer and GEMM shapes are retained. Weight chunks copy
GPU-to-GPU instead of CPU-to-GPU. No head quantization or candidate restriction.
Never import this as a patch to a frozen CPU-head worker.
"""
import torch
import torch.nn.functional as F
from experiments.nan_comp02.head import OperatorHead


class ResidentHead(OperatorHead):
    def __init__(self,cpu_head):
        super().__init__(cpu_head.weight,cpu_head.aliases,device=str(cpu_head.device),chunk_rows=cpu_head.chunk)
        if self.device.type!='cuda':raise ValueError('A resident-head condition requires CUDA')
        self.weight=self.weight.to(self.device)
        if self.weight.dtype!=torch.bfloat16:raise ValueError('The registered resident head retains BF16 precision')


def mount(loop,cpu_head):
    resident=ResidentHead(cpu_head)
    if loop.model.lm_head.bias is not None:raise ValueError('Only the pinned bias-free head is supported')
    loop.model.lm_head.weight=torch.nn.Parameter(resident.weight,requires_grad=False)
    assert loop.model.lm_head.weight.data_ptr()==resident.weight.data_ptr()
    return resident


def retain_decoder_layers(loop,count):
    """New BF16 placement only; retain a fixed prefix plus the existing stage."""
    if not loop.n_gpu_layers<=count<=loop.n:raise ValueError('Invalid registered resident decoder prefix')
    for i in range(loop.n_gpu_layers,count):loop.layers[i].to(loop.device)
    loop.n_gpu_layers=count
    if loop.stage_layer is not None:loop.stage_layer.requires_grad_(False).eval()


@torch.inference_mode()
def full_logits(weight,h,device,chunk_rows):
    """Bounded parity check only: retain one 166144-float readout on CPU."""
    tail=h[:,-1,:].to(device);parts=[]
    staging=torch.empty((min(chunk_rows,weight.shape[0]),weight.shape[1]),device=device,dtype=weight.dtype)
    for start in range(0,weight.shape[0],chunk_rows):
        end=min(start+chunk_rows,weight.shape[0]);n=end-start;staging[:n].copy_(weight[start:end])
        parts.append(F.linear(tail,staging[:n]).flatten().float().cpu())
    return torch.cat(parts)


def check_parity(cpu_head,resident,h):
    a=full_logits(cpu_head.weight,h,cpu_head.device,cpu_head.chunk)
    b=full_logits(resident.weight,h,resident.device,resident.chunk)
    if not torch.equal(a,b):raise AssertionError('Resident placement changed vocabulary logits')
    original=cpu_head.score(h);changed=resident.score(h)
    if original!=changed:raise AssertionError('Resident placement changed full-vocabulary scoring')
    return {'full_vocabulary_logits_checked':len(a),'vocabulary_logits_bit_exact':True,'all_scorer_fields_exact':True,
        'head_dtype':'BF16','head_quantized':False,'scorer_chunk_rows':resident.chunk}
