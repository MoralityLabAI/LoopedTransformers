"""Frozen native two-loop decoder paths, with the LM head permanently on CPU."""
from __future__ import annotations
import copy
import gc
import json
from pathlib import Path
import sys
import torch
from mechinterp.dark_fiber_gyms import arg03_nanbeige as NB
from mechinterp.dark_fiber_gyms.q4_3050 import replace_linear4bit,_slice_context
from .head import ChunkedHead

class NativeLoop(NB.StreamedNanbeigeLoop):
    def __init__(self,model,device,*,quantized=False,channels=(2279,2278,2280,173,1531)):
        dev=torch.device(device)
        n=len(model.model.layers)
        stage=None
        if quantized:
            count=0
            for layer in model.model.layers:
                count+=replace_linear4bit(layer,dev)
                layer.to(dev);gc.collect()
            keep=n
            self.quantized_linears=count
        else:
            keep=1
            stage=copy.deepcopy(model.model.layers[keep]).to(dev)
            model.model.layers[0].to(dev)
            self.quantized_linears=0
        model.model.norm.to(dev)
        model.eval();model.requires_grad_(False)
        # Deliberately no model.lm_head.to(dev).
        assert model.lm_head.weight.device.type=='cpu'
        super().__init__(model,dev,keep,stage)
        self.channels=tuple(c for c in channels if c<model.config.hidden_size)
        self.collect=False
        self.observations=[]
        self.phase=0
        self.active_layer=-1

    def observe(self,h,site):
        if not self.collect:
            return
        a=h.detach().float()
        self.observations.append({'phase':self.phase,'layer':self.active_layer,'site':site,
            'channels':list(self.channels),'values':a[...,list(self.channels)].cpu(),
            'token_l2':a.norm(dim=-1).cpu(),'token_rms':a.square().mean(-1).sqrt().cpu(),
            'token_max_abs':a.abs().amax(-1).cpu()})

    @torch.inference_mode()
    def run_pairs(self,items):
        cur=list(items)
        for idx in range(self.n):
            self.active_layer=idx
            active=self.layers[idx] if idx<self.n_gpu_layers else self._load_stage(self.layers[idx])
            for k,(states,context) in enumerate(cur):
                chunks=[]
                for begin in range(states.shape[0]):
                    ctx=_slice_context(context,begin,begin+1,states.shape[0])
                    h=active(states[begin:begin+1],attention_mask=ctx['mask'],position_ids=ctx['pos'],
                             past_key_value=None,output_attentions=False,use_cache=False,
                             cache_position=ctx['cache_position'],loop_idx=0)[0]
                    if idx in (self.n-2,self.n-1):self.observe(h,'decoder_output_pre_final_norm')
                    chunks.append(h)
                cur[k]=(torch.cat(chunks),context)
        result=[]
        for h,_ in cur:
            normalized=torch.cat([self.base.norm(h[b:b+1]) for b in range(h.shape[0])])
            self.active_layer=self.n-1;self.observe(normalized,'loop_boundary_post_norm')
            result.append(normalized)
        return result

    @torch.inference_mode()
    def step(self,h,ctx,phase,*,observe=False):
        self.phase=phase;self.collect=observe;self.observations=[]
        result=self.run_pairs([(h,ctx)])[0]
        self.collect=False
        return result

def load_backend(compact_root,snapshot,backend,*,device='cuda:0',chunk_rows=8192):
    NB.assert_oracle_isolated()
    sys.path.insert(0,str(Path(compact_root).resolve()))
    from research.algernon.algebraic_reasoning_gym.arg02 import runner,nanbeige
    from research.algernon.algebraic_reasoning_gym.arg02.settings import load_lock
    import os
    os.environ['ARG02_SNAPSHOT_NANBEIGE']=str(Path(snapshot).resolve())
    entry=load_lock()['nanbeige']
    if entry['revision']!='451ed48c3273ecef7ea8faaa43c31ce529763bb1':raise ValueError('Checkpoint pin changed')
    snap=runner.verify_snapshot(entry) # All locked safetensors and code hashes.
    model,_=NB.build_nanbeige_model_and_loop(snap,exec_device=torch.device('cpu'),
        dtype=torch.bfloat16,attn='sdpa',n_gpu_layers=22)
    nanbeige.check_supported(model.config)
    loop=NativeLoop(model,device,quantized=backend=='nf4')
    tok=nanbeige.load_tokenizer(snap,entry['bos_token_id'])
    op_ids=runner.op_token_ids(tok)
    head=ChunkedHead(model.lm_head.weight,op_ids,device=device,chunk_rows=chunk_rows)
    NB.assert_oracle_isolated()
    return loop,head,tok,entry

def prompt_ids(tokenizer,snapshot,row,*,native_chat=True,variant='canonical'):
    from research.algernon.algebraic_reasoning_gym.arg02.prompts import render,TAIL
    text=render(row,shots=[],variant=variant)
    if not native_chat:return tokenizer.encode_prompt(text)
    cfg=json.loads((Path(snapshot)/'tokenizer_config.json').read_text(encoding='utf-8'))
    t=tokenizer.tok
    t.chat_template=cfg['chat_template'];t.bos_token=cfg['bos_token'];t.eos_token=cfg['eos_token']
    body=text[:-len(TAIL)]
    rendered=t.apply_chat_template([{'role':'user','content':body}],tokenize=False,
                                   add_generation_prompt=True,enable_thinking=False)
    return t.encode(rendered+TAIL,add_special_tokens=False)

def candidate_inputs(tokenizer,op_ids,snapshot,row,*,native_chat=True,variant='canonical'):
    from research.algernon.algebraic_reasoning_gym.arg02.recurrence import candidate_rows
    ids=prompt_ids(tokenizer,snapshot,row,native_chat=native_chat,variant=variant)
    return candidate_rows(ids,op_ids),len(ids)
