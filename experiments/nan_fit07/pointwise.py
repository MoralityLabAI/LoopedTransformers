"""Independent CPU replay of the frozen zero-block state-only CI network."""
import numpy as np


def masks(x,checkpoint,ci_config):
    if ci_config['variant']!='state_only' or ci_config['n_blocks']!=0:raise ValueError('Only the registered pointwise controller may be replayed')
    x=np.asarray(x,dtype=np.float64)
    def rms(v):return v/np.sqrt(np.mean(v*v,axis=-1,keepdims=True)+1e-6)
    h=rms(x)@checkpoint['ci__proj_in.weight'].T+checkpoint['ci__proj_in.bias']
    logits=rms(h)@checkpoint['ci__out_proj.weight'].T+checkpoint['ci__out_proj.bias']
    return np.clip(logits,0,1)
