from types import SimpleNamespace
import unittest
import torch
from torch import nn
from mechinterp.nanbeige_bpd_gym.native_cached import NativeTwoLoopCache


class Cache:
    def __init__(self):self.key_cache=[];self.value_cache=[]
    def update(self,k,v,index,kwargs):
        if index==len(self.key_cache):self.key_cache.append(k);self.value_cache.append(v)
        else:self.key_cache[index]=torch.cat([self.key_cache[index],k],dim=-2);self.value_cache[index]=torch.cat([self.value_cache[index],v],dim=-2)
        return self.key_cache[index],self.value_cache[index]
    def get_seq_length(self,index=0):return self.key_cache[index].shape[-2] if len(self.key_cache)>index else 0


class Layer(nn.Module):
    def __init__(self,index):
        super().__init__();self.scale=nn.Parameter(torch.tensor(1.+index/7,dtype=torch.float64),requires_grad=False)
        self.self_attn=nn.Module();self.self_attn.layer_idx=index;self.self_attn.o_proj=nn.Identity();self.mlp=nn.Module();self.mlp.down_proj=nn.Identity()
    def forward(self,h,*,past_key_value,cache_position,**kwargs):
        if past_key_value is not None:k,_=past_key_value.update(h.unsqueeze(1),h.unsqueeze(1),self.self_attn.layer_idx,{})
        else:k=h.unsqueeze(1)
        whole=k.squeeze(1);pos=cache_position
        result=whole.cumsum(1).index_select(1,pos)/(pos+1).reshape(1,-1,1)
        return (h+self.mlp.down_proj(self.self_attn.o_proj(result*self.scale)),)


class Loop:
    def __init__(self):
        torch.manual_seed(7);self.base=nn.Module();self.base.embed_tokens=nn.Embedding(32,4,dtype=torch.float64);self.base.norm=nn.Identity()
        self.layers=nn.ModuleList([Layer(0),Layer(1)]);self.base.layers=self.layers;self.model=self.base.requires_grad_(False)
        self.base._update_causal_mask=lambda *args:None
        self.device=torch.device('cpu');self.n=2;self.n_gpu_layers=1;self.stage_layer=Layer(1);self.phase=0;self.active_layer=-1
    def _load_stage(self,source):
        self.stage_layer.scale.copy_(source.scale);self.stage_layer.self_attn.layer_idx=source.self_attn.layer_idx;return self.stage_layer
    def full(self,ids):
        h=self.base.embed_tokens(torch.tensor([ids]));pos=torch.arange(len(ids));result=[]
        for phase in range(2):
            for layer in self.layers:h=layer(h,past_key_value=None,cache_position=pos)[0]
            h=self.base.norm(h);result.append(h[:,-1:])
        return result


class TwoPassCacheTests(unittest.TestCase):
    def test_separate_phase_caches_match_full_causal_prefixes_and_capture_sites(self):
        loop=Loop();cached=NativeTwoLoopCache(loop,observe_layers=(0,1),cache_factory=Cache);ids=[1,3,7]
        try:
            states,observed=cached.prefill(ids,capture=True)
            self.assertEqual(len(observed),20)
            for a,b in zip(states,loop.full(ids)):torch.testing.assert_close(a,b,rtol=1e-12,atol=1e-12)
            self.assertNotEqual(cached.caches[0].key_cache[0].data_ptr(),cached.caches[1].key_cache[0].data_ptr())
            for token in (13,5,9):
                ids.append(token);states,observed=cached.step(token,capture=True)
                for a,b in zip(states,loop.full(ids)):torch.testing.assert_close(a,b,rtol=1e-12,atol=1e-12)
                self.assertEqual(len(observed),20);self.assertEqual(cached.length,len(ids))
            self.assertGreater(cached.cache_bytes(),0)
        finally:cached.close()

    def test_new_prefix_resets_every_layer_and_rejects_missing_or_batched_input(self):
        c=NativeTwoLoopCache(Loop(),observe_layers=(),cache_factory=Cache)
        try:
            with self.assertRaises(RuntimeError):c.step(1)
            with self.assertRaises(ValueError):c.prefill([])
            with self.assertRaises(ValueError):c.prefill([[1],[2]])
            c.prefill([1,2,3,4]);c.prefill([2,5]);self.assertEqual(c.length,2)
            self.assertTrue(all(cache.get_seq_length(i)==2 for cache in c.caches for i in range(2)))
        finally:c.close()


if __name__=='__main__':unittest.main()
