"""Synthetic CPU execution of real batch/observer/embedding code, no model weights."""
import os
from types import SimpleNamespace
from .common import ROOT,THREADS,write,sha

def run():
    for key in THREADS:os.environ[key]='1'
    from experiments.nan_ranksite13.resources import below_normal
    below_normal()
    import torch
    torch.set_num_threads(1)
    from .batch import install,BatchObserver,Embeddings,RowInstrument
    torch.manual_seed(26)
    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__();self.mlp=SimpleNamespace(down_proj=torch.nn.Linear(4,4,bias=False,dtype=torch.bfloat16));self.self_attn=SimpleNamespace(o_proj=torch.nn.Linear(4,4,bias=False,dtype=torch.bfloat16));self.calls=[]
        def forward(self,h,**kwargs):
            self.calls.append(tuple(h.shape));return (h+self.self_attn.o_proj(h)+self.mlp.down_proj(h),)
    class Base:
        def __init__(self):self.embed_tokens=torch.nn.Embedding(64,4,dtype=torch.bfloat16);self.norm=torch.nn.Identity()
        def _update_causal_mask(self,attention,h,cache,*args):
            assert attention.shape==h.shape[:2];return None
    layers=[Layer().to(torch.bfloat16) for _ in range(4)]
    loop=SimpleNamespace(layers=layers,n=4,n_gpu_layers=4,stage_layer=None,base=Base(),device=torch.device('cpu'),run_pairs=lambda x:None)
    install(loop);observe=BatchObserver(loop,[0,1,2,3]);embed=Embeddings(loop)
    a=[1,2,3];b=[4,5];h,ctx=embed.prepare([a,b]);expected=loop.base.embed_tokens(torch.tensor([a]))
    assert torch.equal(h[0,:3],expected[0]);assert torch.equal(h[1,2],torch.zeros(4,dtype=torch.bfloat16))
    # Test actual unpadded observation and genuinely one B2 layer call.
    observe.enabled=True;observe.lengths=[3,2];observe.phase=1;observe.values={}
    result=loop.run_pairs([(h,ctx)])[0];assert all(layer.calls==[(2,3,4)] for layer in layers)
    assert len(observe.values)==40
    assert torch.equal(observe.values[1,1,3,'residual_after_layer'],result[1,1])
    assert not torch.equal(result[1,1],result[1,2])
    before=embed.gathered;h2,_=embed.prepare([a+[6],b+[7]])
    assert embed.gathered==before+2
    assert torch.equal(h2[0,:3],h[0,:3])
    # Real RowInstrument dispatch, with the algebra replaced by an explicit
    # independently known per-row output. Constructor never loads real BPD.
    row=RowInstrument.__new__(RowInstrument);row.lengths=[3,2];row.modes=['native','component_only'];row.raw_by_phase=[];row.mask_by_phase=[]
    core=SimpleNamespace(mode='components',calls=0,live_controller=None)
    row.base=SimpleNamespace(last_masks=torch.ones(1,2,8).numpy());row.core=core
    def known(x,native):core.calls+=1;return native+2
    row.original=known
    # Extract the exact nested dispatch from source, without framework loading.
    import ast
    tree=ast.parse((ROOT/'batch.py').read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='RowInstrument')
    init=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
    forward=next(n for n in init.body if isinstance(n,ast.FunctionDef) and n.name=='forward')
    namespace={'owner':row};exec(compile(ast.Module(body=[forward],type_ignores=[]),'row_dispatch_fixture','exec'),namespace)
    native=torch.randn(2,3,4,dtype=torch.bfloat16);x=torch.randn_like(native)
    actual=namespace['forward'](core,x,native)
    assert torch.equal(actual[0],native[0]);assert torch.equal(actual[1,:2],native[1,:2]+2);assert torch.equal(actual[1,2],native[1,2])
    assert core.calls==1 and len(row.raw_by_phase)==1 and len(row.mask_by_phase)==1
    observe.close();embed.reset()
    write(ROOT/'engineering_receipt.json',dict(status='passed',synthetic_only=True,pretrained_model_calls=0,
        actual_B2_decoder_call=True,last_valid_token_not_padding=True,embedding_extension_exact=True,clean_row_untouched=True,
        edited_row_valid_tokens_only=True,one_physical_pass_increment=True,CPU_threads=1,
        source_sha256={name:sha(ROOT/name) for name in ('batch.py','engineering.py')},sealed_test_accessed=False))
    print('Synthetic CPU batch/observer/embedding/row-dispatch engineering passed; no pretrained weights.')

if __name__=='__main__':run()
