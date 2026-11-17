"""Actual BPD API finite engineering receipts; no pretrained behavioral claims."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from .components import canonical_factors,spectral_groups,basis
from .edits import routed_mask

def main():
    p=argparse.ArgumentParser();p.add_argument('--bpd-root',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    import mechinterp
    mechinterp.__path__.append(str(Path(a.bpd_root).resolve()/'mechinterp'))
    from mechinterp.bpd.components import TiedComponentLinear,PassMaskSchedule
    torch.manual_seed(871);result={'evidence_class':'finite engineering fixture, random linear maps; no pretrained inference','ranks':{}}
    for rank in [8,16]:
        linear=torch.nn.Linear(32,24);x=torch.randn(2,5,32)
        wrapped=TiedComponentLinear(linear,rank,max_passes=2,init_svd=True)
        wrapped.mode='component';wrapped.schedule=PassMaskSchedule(torch.ones(2,2,5,rank));wrapped.record_inputs=True
        err=max(float((wrapped(x)-linear(x)).abs().max()),float((wrapped(x)-linear(x)).abs().max()))
        assert err<1e-5 and len(wrapped.recorded_inputs)==2
        v=wrapped.V.detach().double();u=wrapped.U.detach().double()
        gauge=torch.eye(rank,dtype=torch.float64);gauge[0,1]=1.7;gauge[2,3]=-.8
        vg=v@gauge;ug=torch.linalg.solve(gauge,u)
        rawdrift=float((v[:,:rank//2]@u[:rank//2]-vg[:,:rank//2]@ug[:rank//2]).norm())
        cv,cu,s=canonical_factors(v,u);gv,gu,_=canonical_factors(vg,ug)
        stable=float((cv[:,:rank//2]@cu[:rank//2]-gv[:,:rank//2]@gu[:rank//2]).abs().max())
        assert stable<1e-10
        # Shear crosses the hard-mask boundary to expose raw-gauge fragility.
        cross=torch.eye(rank,dtype=torch.float64);cross[0,rank-1]=2
        xv=v@cross;xu=torch.linalg.solve(cross,u)
        rawdrift=float((v[:,:rank//2]@u[:rank//2]-xv[:,:rank//2]@xu[:rank//2]).norm())
        assert rawdrift>1e-5
        # Equal singular values must be grouped, never assigned semantic labels.
        assert spectral_groups(torch.tensor([3.,3.,1.],dtype=torch.float64))==[[0,1],[2]]
        checks={}
        for kind in ['svd','pca','gradient','gramian','random']:
            b=basis(torch.randn(24,32),rank,kind,activations=torch.randn(60,24),gradients=torch.randn(60,24),delayed_jacobian=torch.randn(28,24))
            orth=float((b.T@b-torch.eye(rank)).abs().max());assert orth<1e-10;checks[kind]={'orthogonality_error':orth,'shape':list(b.shape)}
        clean=torch.tensor([[9.,1.,0.]]);reached=torch.tensor([[0.,1.,9.]])
        static=routed_mask(clean,torch.eye(3),1);online=routed_mask(reached,torch.eye(3),1)
        assert not torch.equal(static,online)
        result['ranks'][rank]={'full_product_plus_residual_max_error':err,'raw_GL_masked_operator_drift_l2':rawdrift,
            'canonical_masked_operator_max_error':stable,'spectral_groups':spectral_groups(s),'basis_fixtures':checks,
            'static_or_clean_oracle_mask':static.tolist(),'actual_reached_state_mask':online.tolist(),'passes_recorded':2}
    result['sealed_test_accessed']=False;result['status']='passed'
    result['bpd_components_sha256']=hashlib.sha256(Path(__import__('mechinterp.bpd.components',fromlist=['x']).__file__).read_bytes()).hexdigest()
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result))

if __name__=='__main__':main()
