"""Finite positive diagonal/permutation gauge checks at ranks 8 and 16."""
import torch

@torch.no_grad()
def gauge_diagnostic(core,trainer):
    # Exact positive power-of-two scaling and coordinated component relabeling.
    # Test physical writes in float32, not unidentifiable hard-mask indices.
    x=torch.linspace(-1,1,core.d_in,device=core.V.device)[None,None,:]
    mask=torch.linspace(.1,.9,core.C,device=core.V.device)[None,None,:]
    initial=(x@core.V)*mask@core.U
    site=trainer.bpd_model.module_order[0]
    g=trainer.ci_net({site:x.unsqueeze(1)}).lower_leaky[site][:,0]
    controlled=(x@core.V*g)@core.U
    saved_v=core.V.clone();saved_u=core.U.clone();saved_mean=core.mean_comp_acts.clone()
    output_weight=trainer.ci_net.out_proj.weight.clone();output_bias=trainer.ci_net.out_proj.bias.clone()
    scales=torch.tensor([.5,1.,2.,4.,.25,8.,2.,1.],device=core.V.device).repeat((core.C+7)//8)[:core.C];permutation=torch.arange(core.C-1,-1,-1,device=core.V.device)
    try:
        core.V.copy_((saved_v*scales)[:,permutation]);core.U.copy_((saved_u/scales[:,None])[permutation])
        core.mean_comp_acts.copy_((saved_mean*scales)[...,permutation])
        trainer.ci_net.out_proj.weight.copy_(output_weight[permutation]);trainer.ci_net.out_proj.bias.copy_(output_bias[permutation])
        changed=((x@core.V)*mask[...,permutation])@core.U
        relative=float((changed-initial).norm()/initial.norm().clamp_min(1e-12))
        changed_g=trainer.ci_net({site:x.unsqueeze(1)}).lower_leaky[site][:,0]
        control_error=float((changed_g-g[...,permutation]).abs().max())
        controlled_changed=(x@core.V*changed_g)@core.U
        controlled_error=float((controlled_changed-controlled).norm()/controlled.norm().clamp_min(1e-12))
    finally:
        core.V.copy_(saved_v);core.U.copy_(saved_u);core.mean_comp_acts.copy_(saved_mean)
        trainer.ci_net.out_proj.weight.copy_(output_weight);trainer.ci_net.out_proj.bias.copy_(output_bias)
    return {'positive_diagonal_and_permutation_physical_write_relative_l2':relative,
        'coordinated_controller_mask_max_abs':control_error,'controlled_write_relative_l2':controlled_error,
        'stable_at_1e_minus5':max(relative,control_error,controlled_error)<=1e-5,
        'scope':'finite float32 operator diagnostic; general rotations/hard-mask component identities are not certified'}
