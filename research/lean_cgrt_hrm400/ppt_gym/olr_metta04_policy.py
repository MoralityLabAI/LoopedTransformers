"""OLR-METTA-04: trained stage-by-stage STOP/restore controller.

A decision consumes only public MeTTa evidence, native stage, budget and
already observed feedback. The tied attention block carries a recurrent
summary between actual proof decisions. Labels come from registered source
Lean-proof trajectories; query outcomes never enter the optimizer.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

PHASES = ("root", "open", "failed", "guard_ok", "retry", "closed")
ACTIONS = (
    "root_r0","root_r1","probe_guard","work_first",
    "restore","metta_repair","manual","stop",
)
LEGAL = {
    "root": (0,1),
    "open": (2,3,7),
    "failed": (4,7),
    "guard_ok": (5,6,7),
    "retry": (0,1),
    "closed": (7,),
}


def features(claims: list[list[float]], *, phase: str, remaining: int,
             branch: int|None, macro_allowed: bool) -> torch.Tensor:
    if phase not in PHASES or not 0<=remaining<=6 or branch not in (None,0,1):
        raise ValueError("unregistered proof-control state")
    if len(claims)!=6 or any(len(r)!=6 for r in claims):
        raise ValueError("expected six MeTTa evidence rows")
    rows=[list(map(float,claim))+[0.,0.,0.] for claim in claims]
    phase_bits=[float(phase==v) for v in PHASES]
    status=phase_bits+[remaining/6.,float(macro_allowed),
                       0. if branch is None else (1. if branch==0 else -1.)]
    rows.append(status)
    return torch.tensor(rows,dtype=torch.float32)


def mask(phase:str, device:torch.device):
    if phase not in LEGAL:
        raise ValueError("unregistered current stage")
    allowed=torch.full((len(ACTIONS),),float("-inf"),device=device)
    allowed[list(LEGAL[phase])]=0.
    return allowed


class TiedStagePolicy(nn.Module):
    """One attention block applied repeatedly per decision, across decisions."""

    def __init__(self,width:int=24):
        super().__init__()
        if width%4:
            raise ValueError("attention width must support four heads")
        self.width=width
        self.embed=nn.Linear(9,width)
        self.positions=nn.Parameter(torch.zeros(1,7,width))
        self.attention=nn.MultiheadAttention(width,4,batch_first=True)
        self.ff=nn.Sequential(nn.Linear(width,2*width),nn.GELU(),
                              nn.Linear(2*width,width))
        self.norm_a=nn.LayerNorm(width)
        self.norm_b=nn.LayerNorm(width)
        self.norm_c=nn.LayerNorm(width)
        self.head=nn.Linear(width,len(ACTIONS))

    def forward(self,obs:torch.Tensor,prev:torch.Tensor|None=None,
                *,depth:int=3):
        if obs.ndim!=3 or obs.shape[1:]!=(7,9) or depth not in (1,3):
            raise ValueError("expected Bx7x9 and one/three recurrent visits")
        if prev is not None and prev.shape!=(len(obs),self.width):
            raise ValueError("recurrent inter-decision memory shape changed")
        hidden=self.embed(obs)+self.positions
        if prev is not None:
            # Carry from the previous *actual decision*, with bounded gate.
            hidden=hidden.clone()
            hidden[:,6,:]=hidden[:,6,:]+0.2*prev
        for _ in range(depth):
            attended,_=self.attention(hidden,hidden,hidden,need_weights=False)
            hidden=self.norm_a(hidden+0.25*attended)
            hidden=self.norm_b(hidden+0.25*self.ff(hidden))
        summary=hidden[:,6,:]
        carry=self.norm_c(summary if prev is None else 0.65*prev+0.35*summary)
        return self.head(summary),carry


class DirectStagePolicy(nn.Module):
    """Same visible packet and status, flattened one-pass MLP control."""

    def __init__(self,width:int=24):
        super().__init__()
        self.head=nn.Sequential(nn.Linear(63,4*width),nn.GELU(),
                                nn.Linear(4*width,width),nn.GELU(),
                                nn.Linear(width,len(ACTIONS)))

    def forward(self,obs:torch.Tensor,prev=None,*,depth:int=3):
        if obs.ndim!=3 or obs.shape[1:]!=(7,9):
            raise ValueError("direct policy input packet changed")
        return self.head(obs.reshape(len(obs),-1)),None


def training_trace(claims:list[list[float]],*,winner:int,
                   macro_allowed:bool, wrong_start:bool):
    """Source-only teacher sequence with independently verified native branches.

    Examples cover correct starts and errors requiring a restore decision.
    The public-observation features exclude the source winner.
    """
    if winner not in (0,1):
        raise ValueError("source witness must be native-verified")
    chosen=1-winner if wrong_start else winner
    seq=[]
    def add(phase,remaining,branch,action):
        row=features(claims,phase=phase,remaining=remaining,
                     branch=branch,macro_allowed=macro_allowed)
        seq.append({"phase":phase,"obs":row,
                    "target":ACTIONS.index(action)})
    add("root",6,None,"root_r"+str(winner))
    add("open",5,chosen,"probe_guard")
    if wrong_start:
        add("failed",4,chosen,"restore" if macro_allowed else "stop")
        if not macro_allowed:
            return seq
        add("retry",4,chosen,"root_r"+str(winner))
        add("open",3,winner,"probe_guard")
        add("guard_ok",2,winner,"metta_repair")
        add("closed",0,winner,"stop")
    else:
        add("guard_ok",4,winner,"metta_repair" if macro_allowed else "manual")
        add("closed",2 if macro_allowed else 1,winner,"stop")
    return seq


def fit(traces:list[list[dict]],*,seed:int,arm:str,steps:int=80,
        batch_size:int=12, width:int=24):
    if arm not in ("loop3","loop1","direct") or not traces:
        raise ValueError("unregistered trained stage controller")
    if not 1<=steps<=500 or len(traces)>512:
        raise ValueError("training workload outside bounded CPU contract")
    torch.manual_seed(seed)
    model=DirectStagePolicy(width) if arm=="direct" else TiedStagePolicy(width)
    model.train()
    optimizer=torch.optim.AdamW(model.parameters(),lr=0.002,weight_decay=.005)
    rng=torch.Generator().manual_seed(seed+9600)
    indices=torch.randint(len(traces),(steps,min(batch_size,len(traces))),
                          generator=rng)
    depth=1 if arm=="loop1" else 3
    first_loss=last_loss=0.
    for i,batch in enumerate(indices):
        optimizer.zero_grad(set_to_none=True)
        losses=[]
        for idx in batch.tolist():
            trace=traces[idx]
            carry=None
            for step in trace:
                logits,carry=model(step["obs"].unsqueeze(0),carry,depth=depth)
                z=logits[0]+mask(step["phase"],logits.device)
                target=torch.tensor([step["target"]],dtype=torch.long)
                losses.append(F.cross_entropy(z.unsqueeze(0),target))
        loss=torch.stack(losses).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite native-controller source imitation loss")
        if i==0:first_loss=float(loss.detach())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
        last_loss=float(loss.detach())
    model.eval()
    return model,{"seed":seed,"arm":arm,"parameters":sum(p.numel() for p in model.parameters()),
                  "steps":steps,"depth":depth,"source_traces":len(traces),
                  "first_loss":first_loss,"last_loss":last_loss}


@torch.no_grad()
def act(model:nn.Module,claims:list[list[float]],*,phase:str,
        remaining:int,branch:int|None,macro_allowed:bool,
        carry:torch.Tensor|None,arm:str):
    if arm not in ("loop3","loop1","direct"):
        raise ValueError("invalid learned policy architecture")
    packet=features(claims,phase=phase,remaining=remaining,
                    branch=branch,macro_allowed=macro_allowed).unsqueeze(0)
    logits,next_carry=model(packet,carry,depth=(1 if arm=="loop1" else 3))
    z=logits[0]+mask(phase,logits.device)
    return ACTIONS[int(z.argmax().item())],next_carry,tuple(float(v) for v in z[LEGAL[phase]].tolist())
