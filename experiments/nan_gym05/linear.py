"""Weight-only rank-8/16 baselines, explicitly not learned VPD/BPD factors."""
import hashlib
from pathlib import Path
import numpy as np
import torch
from .common import write,sha

@torch.inference_mode()
def decompose(matrix,rank=16,seed=0):
    torch.manual_seed(seed);W=matrix.detach().float()
    u,s,v=torch.svd_lowrank(W,q=rank,niter=2)
    arrays={};checks={}
    for r in (8,16):
        if r>rank:continue
        V=v[:,:r]*s[:r].sqrt();U=s[:r].sqrt()[:,None]*u[:,:r].T
        residual=W-(V@U).T
        x=torch.randn(4,W.shape[1],device=W.device)
        direct=x@W.T;closed=(x@V)@U+x@residual.T
        absolute=float((direct-closed).abs().max());relative=float((direct-closed).norm()/direct.norm().clamp_min(1e-12))
        if relative>1e-5:raise RuntimeError('Effective-operator float32 closure failed')
        arrays['V'+str(r)]=V.cpu().numpy();arrays['U'+str(r)]=U.cpu().numpy();arrays['singular_values'+str(r)]=s[:r].cpu().numpy()
        checks[str(r)]={'closure_max_abs':absolute,'closure_relative_l2':relative,
            'factor_relative_frobenius_error':float(residual.norm()/W.norm().clamp_min(1e-12)),
            'scope':'float32 effective operator plus residual, not native BF16/NF4 kernel parity'}
    payload=W.cpu().contiguous().numpy().tobytes()
    return arrays,{'effective_float32_operator_sha256':hashlib.sha256(payload).hexdigest(),'shape':list(W.shape),
        'ranks':checks,'factorization':'randomized rank-16 SVD baseline; rank-8 nested prefix','trained_VPD':False,'trained_BPD':False}

@torch.inference_mode()
def extract(loop,out,backend,cfg):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);receipts=[]
    for layer in cfg['layers_zero_based']:
        block=loop.layers[layer]
        for name,module in [('down_proj',block.mlp.down_proj),('o_proj',block.self_attn.o_proj)]:
            print(f'{backend} algebraic baseline layer {layer} {name}',flush=True)
            weight=module.weight
            if getattr(weight,'quant_state',None) is not None:
                from bitsandbytes.functional import dequantize_4bit
                matrix=dequantize_4bit(weight.data,quant_state=weight.quant_state).float()
            else:matrix=weight.detach().to(loop.device,dtype=torch.float32)
            arrays,receipt=decompose(matrix,cfg['svd_baseline_rank'],cfg['source_seed']+layer)
            file=out/f'layer{layer}_{name}.npz';np.savez_compressed(file,**arrays)
            receipt|={'backend':backend,'checkpoint_revision':cfg['checkpoint_revision'],'layer_zero_based':layer,'module':name,
                'native_loop_coverage':[1,2],'file':file.name,'file_sha256':sha(file),
                'output_site':name+'_output','bridge_to_residual_after_layer':'identity addition for down_proj output' if name=='down_proj' else 'requires intervening MLP Jacobian; no identity bridge claimed'}
            receipts.append(receipt);del matrix,arrays;torch.cuda.empty_cache()
    write(out/'manifest.json',{'evidence_class':'weight-only algebraic baseline','receipts':receipts,'BPD_transfer_attempted':False})
