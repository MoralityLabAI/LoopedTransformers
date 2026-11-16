"""LS-SAE02 calibration-only PCA/SAE fit for native LoopedTRM step-one states."""
import hashlib
import json
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from mechinterp.loopscope02.scripts import trm_sudoku as TS
from mechinterp.sae_causal_transfer.adapter import (
    fingerprint_arrays,to_physical_dictionary,
)

SAE_SEED=71
DATA_SEED=20261971
FIT_STEPS=200
FIT_BATCH=256
PCA_RANK=48
SAE_WIDTH=96
TOP_K=4


def question(row):
    return "".join("." if v==0 else str(v) for v in row["x"])


def stage_one(model,rows,batch_size=2,retain_zl=False):
    """Native exact ACT step 1, all board-token zh, optional saved zl for continuation."""
    hidden=[]
    anchors=[]
    model.eval()
    device=next(model.parameters()).device
    with torch.no_grad():
        for i in range(0,len(rows),batch_size):
            inp=TS.encode_questions([question(row) for row in rows[i:i+batch_size]],device)
            xemb=model.encode_inputs(inp)
            zh,zl=model.init_states(len(inp))
            zh,zl=model.act_step(zh,zl,xemb)
            hidden.append(zh[:,model.puzzle_emb_len:,:].detach().float().cpu().numpy())
            if retain_zl:
                for p in range(len(inp)):
                    anchors.append({
                        "zh":zh[p:p+1].detach().clone(),
                        "zl":zl[p:p+1].detach().clone(),
                        "x_emb":xemb[p:p+1].detach().clone(),
                    })
    return np.concatenate(hidden,axis=0),anchors


class SparseTopFour(nn.Module):
    def __init__(self,d=48,width=96):
        super().__init__()
        self.encoder=nn.Linear(d,width)
        self.decoder=nn.Linear(width,d)
        with torch.no_grad():
            self.encoder.bias.fill_(0.5)

    def encode(self,x):
        z=F.relu(self.encoder(x))
        _,idx=torch.topk(z,TOP_K,dim=-1)
        return torch.zeros_like(z).scatter(-1,idx,z.gather(-1,idx))

    def forward(self,x):
        code=self.encode(x)
        return self.decoder(code),code


@dataclass
class Fitted:
    mean512:np.ndarray
    basis:np.ndarray
    mu:np.ndarray
    scale:np.ndarray
    physical_rows:np.ndarray
    physical_bias:np.ndarray
    physical_decoder:np.ndarray
    sae:SparseTopFour
    fingerprint:str
    final_loss:float

    def encode(self,hidden):
        y=(np.asarray(hidden,dtype=np.float64)-self.mean512)@self.basis
        return y


def fit_only(states,*,steps=FIT_STEPS,rank=PCA_RANK,seed=SAE_SEED):
    """Fit source-only normalizer, PCA and top4 SAE, never a test state."""
    if steps!=FIT_STEPS or rank!=PCA_RANK or seed!=SAE_SEED:
        raise ValueError("registration forbids changed fit budget or seed")
    X=np.asarray(states,dtype=np.float64)
    if X.ndim!=3 or X.shape[1]!=81 or X.shape[-1]!=512 or len(X)<4:
        raise ValueError("native z_H must be N×81×512 on fit cohort")
    if not np.all(np.isfinite(X)):
        raise ValueError("nonfinite hidden fit state")
    x=X.reshape(-1,512)
    mean512=x.mean(0)
    centered=x-mean512
    # Sign-fix each principal direction to make replay stable across SVD conventions.
    _,sv,vh=np.linalg.svd(centered,full_matrices=False)
    basis=vh[:rank].T.copy()
    for j in range(rank):
        pivot=int(np.abs(basis[:,j]).argmax())
        if basis[pivot,j]<0:
            basis[:,j]*=-1
    if np.linalg.norm(basis.T@basis-np.eye(rank),ord=np.inf)>1e-7:
        raise ArithmeticError("nonorthogonal fit-only PCA")
    projected=centered@basis
    mu=projected.mean(0)
    scale=np.maximum(projected.std(0),1e-6)
    z=torch.tensor((projected-mu)/scale,dtype=torch.float32)
    torch.manual_seed(seed)
    sae=SparseTopFour(rank,SAE_WIDTH)
    opt=torch.optim.AdamW(sae.parameters(),lr=.003,weight_decay=.0001)
    rng=torch.Generator().manual_seed(DATA_SEED)
    for _ in range(steps):
        indices=torch.randint(len(z),(FIT_BATCH,),generator=rng)
        xbatch=z[indices]
        reconstructed,code=sae(xbatch)
        loss=F.mse_loss(reconstructed,xbatch)+.001*code.mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    sae.eval()
    final=float(loss.detach())
    if not np.isfinite(final):
        raise ArithmeticError("nonfinite fit-only SAE loss")
    E=sae.encoder.weight.detach().numpy().copy()
    bias=sae.encoder.bias.detach().numpy().copy()
    D=sae.decoder.weight.detach().numpy().copy()
    phys_E,phys_b,phys_D=to_physical_dictionary(E,bias,D,mu,scale)
    fingerprint=fingerprint_arrays(mean512,basis,mu,scale,E,bias,D)
    return Fitted(mean512,basis,mu,scale,phys_E,phys_b,phys_D,
                  sae,fingerprint,final)


def frozen_fit_receipt(fit,fit_roster_digest,checkpoint_sha):
    return {"schema":"LS-SAE02/calibration-locked-native-PCA-SAE-v1",
            "checkpoint_file_sha256":checkpoint_sha,
            "source_fit_roster_sha256":fit_roster_digest,
            "sae_seed":SAE_SEED,"sampling_seed":DATA_SEED,
            "optimizer_steps":FIT_STEPS,"batch_size":FIT_BATCH,
            "pca_rank":PCA_RANK,"sae_width":SAE_WIDTH,"topk":TOP_K,
            "physical_dictionary_sha256":fingerprint_arrays(
                fit.physical_rows,fit.physical_bias,fit.physical_decoder),
            "calibration_fit_fingerprint_sha256":fit.fingerprint,
            "final_fit_loss":fit.final_loss}


def fit_support_diagnostics(fit,states):
    hidden=np.asarray(states,dtype=np.float64).reshape(-1,512)
    reduced=fit.encode(hidden)
    from mechinterp.sae_causal_transfer.observer import encode_top4
    supported=np.zeros(96,dtype=np.int64)
    for x in reduced:
        code,_=encode_top4(x,fit.physical_rows,fit.physical_bias)
        supported+=(code>0)
    return {"n_observed_board_tokens":len(hidden),
            "feature_activities":supported.tolist(),
            "never_active_atoms":int((supported==0).sum()),
            "n_positive_total":int(supported.sum())}
