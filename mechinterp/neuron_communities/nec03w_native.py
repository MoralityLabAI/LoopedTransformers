"""NEC-03W optional CPU Torch recurrent intervention hooks.

A single native original recurrent visit receives the source-only frozen
full-affine donor with A, B or A+B removed. After the NEXT whole block visit,
a 32D memory-state residual is injected either along the removed target's
frozen physical direction or along a matched disjoint sham direction.

This delayed rescue is a new causal question about information transport.
An EXACT SAME-VISIT addback reproduces the original donor by construction;
it is solely an implementation positive control, never empirical proof of
feature necessity.

This module has no checkpoint loading, source-root discovery or task labels.
Callers must pass the original SHA-verified source-only model and original
source-fit geometry. All operations stay on CPU; weights remain frozen.
"""
from __future__ import annotations

from dataclasses import dataclass

D=32
ACTIVATION=64
TIED_PASSES=12
MODES=(
    "baseline", "remove_A","remove_B","remove_AB",
    "sham_C","sham_D","sham_CD",
    "late_target_rescue","late_matched_sham_rescue",
    "same_visit_exact_addback",
)


@dataclass(frozen=True)
class DelayedRescue:
    visit: int
    direction32: object


def validate_rescue(rescue, batch):
    if rescue is None:
        return
    import torch
    if not isinstance(rescue,DelayedRescue) or not 1<=rescue.visit<TIED_PASSES:
        raise ValueError("rescue must target the immediately subsequent original recurrent visit")
    if (not isinstance(rescue.direction32,torch.Tensor)
            or rescue.direction32.device.type!="cpu"
            or tuple(rescue.direction32.shape)!=(batch,D)
            or not bool(torch.isfinite(rescue.direction32).all())):
        raise ValueError("late source-rescue directions must be finite CPU [batch,32]")


def native_forward(model,bits,*,donors=None,gates=None,rescue=None):
    """Run exactly 12 passes with independently countable source interventions.

    donors[v] has shape [batch,64]; gates[v] is a binary [64] *original hub*
    mask, and rescue is an optional 32D correction AFTER the next block
    iteration. This never evaluates or selects a source candidate itself.
    """
    import torch
    if (not isinstance(bits,torch.Tensor) or bits.device.type!="cpu"
            or bits.ndim!=2 or bits.shape[1]!=TIED_PASSES
            or any(p.device.type!="cpu" for p in model.parameters())):
        raise ValueError("original NEC03W recurrent inference must be CPU only")
    donors={} if donors is None else donors
    gates={} if gates is None else gates
    if set(donors)!=set(gates) or any(
        type(k) is not int or not 0<=k<TIED_PASSES for k in donors
    ):
        raise ValueError("source donor and original hub gate visits differ")
    for visit,d in donors.items():
        gate=gates[visit]
        if (not isinstance(d,torch.Tensor) or not isinstance(gate,torch.Tensor)
                or d.device.type!="cpu" or gate.device.type!="cpu"
                or tuple(d.shape)!=(len(bits),ACTIVATION)
                or tuple(gate.shape)!=(ACTIVATION,)
                or not bool(torch.isfinite(d).all())
                or not bool(torch.isfinite(gate).all())
                or not bool(((gate==0)|(gate==1)).all())):
            raise ValueError("original source 64D donor/gate geometry invalid")
    validate_rescue(rescue,len(bits))
    pre=[0]
    post=[0]

    def on_down(_layer,args):
        v=pre[0]
        pre[0]+=1
        if v not in donors:
            return None
        x=args[0]
        if tuple(x.shape)!=(len(bits),2,ACTIVATION):
            raise ValueError("NEC03W MLP-down source activation geometry changed")
        y=x.clone()
        y[:,1,:]=x[:,1,:]*gates[v][None,:]+donors[v]
        return (y,)

    def on_block(_layer,_args,outputs):
        v=post[0]
        post[0]+=1
        if tuple(outputs.shape)!=(len(bits),2,D):
            raise ValueError("original recurrent block output must be memory/token [batch,2,32]")
        if rescue is None or v!=rescue.visit:
            return None
        corrected=outputs.clone()
        corrected[:,1,:]+=rescue.direction32
        return corrected

    handles=[
        model.block.mlp_down.register_forward_pre_hook(on_down),
        model.block.register_forward_hook(on_block),
    ]
    try:
        with torch.no_grad():
            logits=model.logits(bits)
    finally:
        for hook in handles:
            hook.remove()
    if pre!=[TIED_PASSES] or post!=[TIED_PASSES] or tuple(logits.shape)!=(len(bits),2):
        raise ValueError("source recurrent trajectory omitted or duplicated one of twelve passes")
    return logits.detach().clone()


def source_donor_modes(frozen,z_per_visit,*,target_visit,matched_geometry):
    """Construct paired original SAE/full-affine donors for a ONE-visit test.

    The data caller must already have authenticated the 16-positive-code
    TopK record for each case and EACH historical selected source visit.
    Any invalid case passes the entire original full-affine donor through
    every arm and receives no late rescue. Single-visit variants avoid
    confounding different recurrence-step site counts.
    """
    import torch
    if (target_visit not in frozen["source_sites"] or
            type(target_visit) is not int or target_visit not in range(TIED_PASSES)):
        raise ValueError("only one originally registered recurrent source visit can be manipulated")
    model=frozen["model"]
    sae=frozen["sae"]
    if sae is None:
        raise ValueError("source-ineligible original arm cannot synthesize SAE donor")
    scale=frozen["scale"]
    mean=frozen["mean"]
    down=model.block.mlp_down.weight.detach()
    batch=len(matched_geometry)
    if len(z_per_visit)==0 or any(v not in frozen["source_sites"] for v in z_per_visit):
        raise ValueError("missing original source codes")
    donors={}
    gates={}
    source_motion=torch.zeros((batch,32),dtype=down.dtype)
    sham_motion=torch.zeros_like(source_motion)
    for v in sorted(frozen["source_sites"]):
        if v not in z_per_visit:
            raise ValueError("all other selected original source visits need full-affine background")
        z=z_per_visit[v]
        if tuple(z.shape)!=(batch,128) or not bool(torch.isfinite(z).all()):
            raise ValueError("original source SAE code matrix must be [batch,128]")
        gate=torch.zeros(64,dtype=down.dtype)
        gate[sorted(frozen["source_hubs"].get(v,()))]=1.
        nonhub=1-gate
        full=(sae.decoder(z)*scale+mean)*nonhub
        donors[v]={"baseline":full}
        gates[v]=gate
        if v!=target_visit:
            continue
        parts={key:torch.zeros_like(full) for key in ("A","B","C","D")}
        for i,row in enumerate(matched_geometry):
            if not row["admissible"]:
                continue
            if row["visit"]!=v:
                raise ValueError("matched original causal quartet belongs to a different recurrence visit")
            coeffs={
                "A":(row["pair_A"],row["source_gain"]),
                "B":(row["pair_B"],row["source_gain"]),
                "C":(row["matched_control_C"],row["sham_gain"]),
                "D":(row["matched_control_D"],row["sham_gain"]),
            }
            for key,(feature_ids,gain) in coeffs.items():
                native=z[i,feature_ids]@sae.decoder.weight[:,feature_ids].T
                native=native*scale*nonhub*float(gain)
                physical=native@down.T
                expected=torch.tensor(row["projected32"][key],dtype=down.dtype)
                if not torch.allclose(physical,expected,atol=2e-4,rtol=2e-5):
                    raise ValueError("real source 64D pair removal disagrees with independently checked 32D")
                parts[key][i]=native
            source_motion[i]=torch.tensor(row["projected32"]["AB"],dtype=down.dtype)
            sham_motion[i]=torch.tensor(row["projected32"]["CD"],dtype=down.dtype)
        for key in parts:
            donors[v][key]=full-parts[key]
        donors[v]["AB"]=full-parts["A"]-parts["B"]
        donors[v]["CD"]=full-parts["C"]-parts["D"]
        donors[v]["exact_addback"]=donors[v]["AB"]+parts["A"]+parts["B"]
        if not torch.allclose(donors[v]["exact_addback"],full,atol=5e-6,rtol=1e-6):
            raise ValueError("same-visit algebraic rescue failed to reproduce full-affine donor")

    plans={}
    for mode in MODES:
        selected={}
        for v,variant in donors.items():
            if mode=="baseline":
                key="baseline"
            elif mode=="same_visit_exact_addback":
                key="exact_addback" if v==target_visit else "baseline"
            else:
                short={
                    "remove_A":"A","remove_B":"B","remove_AB":"AB",
                    "sham_C":"C","sham_D":"D","sham_CD":"CD",
                    "late_target_rescue":"AB","late_matched_sham_rescue":"AB",
                }[mode]
                key=short if v==target_visit else "baseline"
            selected[v]=variant[key]
        rescue=None
        if mode.startswith("late_") and target_visit+1<TIED_PASSES:
            source=source_motion if mode=="late_target_rescue" else sham_motion
            rescue=DelayedRescue(visit=target_visit+1,direction32=source)
        plans[mode]={"donors":selected,"gates":gates,"rescue":rescue}
    return plans
