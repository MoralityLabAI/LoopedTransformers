"""Self-contained FP64 causal fixture: padding holes, semantic positions and pruning."""
import unittest
import torch
from torch import nn
from mechinterp.nanbeige_bpd_gym.chunked_cache import ChunkedTwoLoopCache


class Cache:
    def __init__(self):self.key_cache=[];self.value_cache=[]
    def update(self,k,v,index,kwargs):
        if index==len(self.key_cache):self.key_cache.append(k);self.value_cache.append(v)
        else:self.key_cache[index]=torch.cat([self.key_cache[index],k],dim=-2);self.value_cache[index]=torch.cat([self.value_cache[index],v],dim=-2)
        return self.key_cache[index],self.value_cache[index]
    def get_seq_length(self,index=0):return self.key_cache[index].shape[-2] if len(self.key_cache)>index else 0


class Layer(nn.Module):
    def __init__(self,index):
        super().__init__();self.scale=nn.Parameter(torch.tensor(1+index/7,dtype=torch.float64),requires_grad=False)
        self.self_attn=nn.Module();self.self_attn.layer_idx=index;self.self_attn.o_proj=nn.Identity()
        self.mlp=nn.Module();self.mlp.down_proj=nn.Identity()
    def forward(self,h,*,attention_mask,position_ids,past_key_value,cache_position,**kwargs):
        projected=h+position_ids.unsqueeze(-1)*.01
        k=projected.unsqueeze(1).clone();v=k.clone()
        if past_key_value is not None:k,v=past_key_value.update(k,v,self.self_attn.layer_idx,{})
        values=v.squeeze(1);end=values.shape[1]
        allowed=(torch.arange(end)[None,None,:]<=cache_position[None,:,None]) & attention_mask[:,:end].bool()[:,None,:]
        average=(allowed.to(h.dtype)@values)/allowed.sum(-1,keepdim=True).clamp_min(1)
        return (h+self.mlp.down_proj(self.self_attn.o_proj(average*self.scale)),)


class Loop:
    def __init__(self):
        torch.manual_seed(7);self.base=nn.Module();self.base.embed_tokens=nn.Embedding(64,4,dtype=torch.float64);self.base.norm=nn.Identity()
        self.layers=nn.ModuleList([Layer(0),Layer(1)]);self.base.layers=self.layers;self.model=self.base.requires_grad_(False)
        self.base._update_causal_mask=lambda attention,*args:attention
        self.device=torch.device('cpu');self.n=2;self.n_gpu_layers=1;self.stage_layer=Layer(1);self.phase=0;self.active_layer=-1
    def _load_stage(self,source):
        self.stage_layer.scale.copy_(source.scale);self.stage_layer.self_attn.layer_idx=source.self_attn.layer_idx;return self.stage_layer
    def full(self,sequences):
        P=max(map(len,sequences));h=torch.zeros((len(sequences),P,4),dtype=torch.float64);mask=torch.zeros(len(sequences),P,dtype=torch.long)
        for b,ids in enumerate(sequences):h[b,:len(ids)]=self.base.embed_tokens(torch.tensor(ids));mask[b,:len(ids)]=1
        positions=torch.arange(P);bounds=[]
        for phase in (1,2):
            for layer in self.layers:h=layer(h,attention_mask=mask,position_ids=positions[None].expand(len(sequences),-1),past_key_value=None,cache_position=positions)[0]
            bounds.append(torch.stack([h[b,len(ids)-1] for b,ids in enumerate(sequences)]))
        return bounds


class ChunkedCacheTests(unittest.TestCase):
    def test_ragged_semantic_positions_and_row_pruning_match_full_causal_inputs(self):
        for inplace in (False,True):
            loop=Loop();cache=ChunkedTwoLoopCache(loop,prefill_chunk=2,observe_layers=(0,1),cache_factory=Cache,inplace=inplace)
            sequences=[[1,3,7,9],[2,5]]
            try:
                value=cache.prefill(sequences)
                self.assertEqual(len(value['states']),40)
                for x,y in zip(value['boundaries'],loop.full(sequences)):torch.testing.assert_close(x,y,rtol=1e-12,atol=1e-12)
                for tokens in ([13,17],[19,23]):
                    for ids,t in zip(sequences,tokens):ids.append(t)
                    value=cache.step(tokens)
                    for x,y in zip(value['boundaries'],loop.full(sequences)):torch.testing.assert_close(x,y,rtol=1e-12,atol=1e-12)
                cache.select_rows([1]);sequences=[sequences[1]];sequences[0].append(29);value=cache.step([29])
                self.assertEqual(cache.physical_length,7);self.assertEqual(cache.semantic_lengths,[5])
                self.assertEqual(len(value['states']),20)
                for x,y in zip(value['boundaries'],loop.full(sequences)):torch.testing.assert_close(x,y,rtol=1e-12,atol=1e-12)
                self.assertEqual(cache.stage_materializations,8)
            finally:cache.close()
    def test_reset_and_invalid_inputs_or_shared_phase_cache_are_refused(self):
        loop=Loop();cache=ChunkedTwoLoopCache(loop,prefill_chunk=2,observe_layers=(),cache_factory=Cache)
        try:
            with self.assertRaises(RuntimeError):cache.step([1])
            for ids in ([],[[]],[[True]],[[64]]):
                with self.assertRaises(ValueError):cache.prefill(ids)
            with self.assertRaisesRegex(ValueError,'BPD instrument'):cache.prefill([[1]],modes=['reached_state'])
            with self.assertRaises(ValueError):cache.prefill([[1]],modes=[])
            with self.assertRaisesRegex(ValueError,'Unknown'):cache.prefill([[1]],modes=['unregistered_route'])
            cache.prefill([[1,2,3]]);cache.prefill([[5]])
            self.assertEqual(cache.physical_length,1)
            for indices in ([],[0,0],[1]):
                with self.assertRaises(ValueError):cache.select_rows(indices)
        finally:cache.close()
        shared=Cache();cache=ChunkedTwoLoopCache(Loop(),observe_layers=(),cache_factory=lambda:shared)
        try:
            with self.assertRaisesRegex(ValueError,'Independent'):cache.prefill([[1]])
        finally:cache.close()


if __name__=='__main__':unittest.main()
