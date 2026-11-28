"""Metadata-only Q4 storage scenarios; does not claim numerical parity or runtime fit."""
import argparse, json, math
from pathlib import Path
parser=argparse.ArgumentParser()
parser.add_argument('--inventory',type=Path,default=Path('preflight/inventory.json'))
parser.add_argument('--out',type=Path,default=Path('pilot/backbone_estimate.json'))
a=parser.parse_args()
p=a.inventory
inv=json.loads(p.read_text())
sensitive=('mlp.gate.','message','sconv','convolution','attn_res','mlp_res')
items={k:v for k,v in inv['tensors'].items() if k.startswith('model.llm.') and '.experts.' not in k}

def is_eligible(k,v,head=True,embed=True):
 if any(part in k for part in sensitive):return False
 if '.unembed.' in k and not head:return False
 if '.embed.' in k and not embed:return False
 return v['dtype']=='BF16' and len(v['shape'])==2 and v['bytes']>=262144 and v['shape'][-1]>=64

def q4_bytes(shape):
 rows=math.prod(shape[:-1]);n=shape[-1];blocks=math.ceil(n/64)
 return rows*blocks*32+rows*blocks*2

scenarios={}
for name,head,embed in [('Q4_all_eligible',True,True),('Q4_keep_head_BF16',False,True),
                        ('Q4_keep_embed_BF16',True,False),('Q4_keep_both_BF16',False,False)]:
 selected={k:v for k,v in items.items() if is_eligible(k,v,head,embed)}
 q4=sum(q4_bytes(v['shape']) for v in selected.values())
 unchanged=sum(v['bytes'] for k,v in items.items() if k not in selected)
 scenarios[name]={'selected_tensors':len(selected),'Q4_code_plus_FP16_scale_bytes':q4,
                  'higher_precision_bytes':unchanged,'total_weight_bytes_estimate':q4+unchanged,
                  'GiB_estimate':(q4+unchanged)/2**30,
                  'RTX3050_4GiB_headroom_before_cache_kernel_buffers_and_experts':4*2**30-q4-unchanged}
ret={'source_revision':inv['revision'],'original_text_other_bytes':sum(v['bytes'] for v in items.values()),
     'excluded_other_review_required_bytes':inv['bucket_bytes']['other_review_required'],
     'q4_method':'signed block64 per last axis, 4-bit codes + one FP16 scale per block; excludes alignment/headers',
     'preserved':'router, message/recurrent/convolution, norms, 1D and small tensors; head/embedding by scenario',
     'scenarios':scenarios,
     'limits':'No numerical INT4 backbone conversion, model inference, transfer or 3050 VRAM measurement performed; estimate excludes CPU expert bank, resident experts, caches, activations, kernels, fragmentation and vision dependency validation.'}
out=a.out;out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps(ret,indent=2)+'\n')
print(json.dumps({k:round(v['GiB_estimate'],3) for k,v in scenarios.items()}))
