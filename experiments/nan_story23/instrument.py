"""Qualified IO18 T2 physical online edits; no host answer access."""
import importlib
from pathlib import Path
from .common import fit_asset
from .adapters import bound_components


class Instrument:
    def __init__(self,loop,backend,bpd_root,cfg):
        import numpy as np
        import torch
        from mechinterp.nanbeige_bpd_gym.native_components import load_bpd
        from mechinterp.nanbeige_bpd_gym.recomputed_native import attach_resident_components
        from experiments.nan_recomp09.common import target_digest
        self.asset=fit_asset(backend,2);assert self.asset is not None
        assert target_digest(loop)==self.asset['receipt']['target_parameter_sha256_unchanged'], 'Qualified factors require the exact same native target weights'
        framework=load_bpd(bpd_root);self.cfg=cfg;self.loop=loop;self.raw=[];self.handles=[]
        # Snapshot native physical outputs BEFORE the real edit hook.
        def raw(module,args,output):
            if loop.active_layer==21:
                self.raw.append(output[0,-1].detach().float().cpu().clone())
        self.handles.append(loop.layers[21].mlp.down_proj.register_forward_hook(raw))
        if loop.stage_layer is not None and 21>=loop.n_gpu_layers:
            self.handles.append(loop.stage_layer.mlp.down_proj.register_forward_hook(raw))
        receipt=self.asset['receipt'];ablation=framework['config'].AblationConfig(**receipt['training_contract']['ablation'])
        self.model=attach_resident_components(loop,{cfg['learned_site']:8},bpd_root=bpd_root,ablation=ablation,max_passes=2)
        self.core=self.model.wrappers[cfg['learned_site']]
        with np.load(self.asset['checkpoint'],allow_pickle=False) as data:
            for name in ('V','U','mean_comp_acts','mean_delta_out','mean_write_rms','calibrated_passes'):
                target=getattr(self.core,name)
                with torch.no_grad():target.copy_(torch.as_tensor(data[name],device=target.device,dtype=target.dtype))
            ci_config=receipt['final_checkpoint']['CI_config'];assert ci_config['variant']=='state_only' and ci_config['n_blocks']==0
            module=importlib.import_module('mechinterp.bpd.ci');assert Path(module.__file__).resolve().is_relative_to(Path(bpd_root).resolve())
            self.ci=module.TiedCINetwork({cfg['learned_site']:self.core.d_in},{cfg['learned_site']:8},cfg=framework['config'].CIConfig(**ci_config)).to(loop.device).eval()
            state={k[4:]:torch.as_tensor(data[k],device=loop.device) for k in data.files if k.startswith('ci__')}
            self.ci.load_state_dict(state,strict=True)
        for p in self.model.component_parameters():p.requires_grad_(False)
        for p in self.ci.parameters():p.requires_grad_(False)
        self.static=torch.as_tensor(self.asset['static_CAL_mask'],device=loop.device,dtype=torch.float32)
        assert tuple(self.static.shape)==(2,8)
        bound_components(self.core,cfg['component_token_chunk_rows'])
        self.Schedule=framework['components'].PassMaskSchedule

    def context(self,mode,length):
        import torch
        self.raw=[];self.core.live_controller=None
        if mode not in ('all_one','component_only','static_CAL_mean','reached_state'):raise ValueError(mode)
        masks=torch.ones((1,2,length,8),device=self.loop.device)
        if mode=='static_CAL_mean':masks[:]=self.static[None,:,None,:]
        delta=torch.ones((1,2,length),device=self.loop.device) if mode=='all_one' else torch.zeros((1,2,length),device=self.loop.device)
        if mode=='reached_state':
            def live(x,tau):
                return self.ci({self.cfg['learned_site']:x.unsqueeze(1)},pass_offset=tau).lower_leaky[self.cfg['learned_site']][:,0]
            self.core.live_controller=live
        self.last_masks=masks[:,:,-1,:].detach().cpu().numpy().copy()
        return self.model.masked_components({self.cfg['learned_site']:self.Schedule(masks,delta)})

    def snapshots(self):
        import numpy as np
        assert len(self.raw)==2
        masks=self.last_masks[0]
        if self.core.live_controller is not None:
            assert len(self.core.controller_observations)==2
            masks=np.asarray([v['mask'][0,-1].detach().float().cpu().numpy() for v in self.core.controller_observations])
        self.core.live_controller=None
        return masks,np.asarray([v.numpy() for v in self.raw])

    def close(self):
        self.core.live_controller=None;self.model.close()
        for handle in self.handles:handle.remove()
