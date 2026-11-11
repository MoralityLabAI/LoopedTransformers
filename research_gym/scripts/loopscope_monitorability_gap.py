"""RMP-MG-01 local readout assay using the existing semantic-echo native adapter.

Run from the LoopedTransformers checkout after applying this additive bundle.
Default --smoke uses an untrained model. A checkpoint run requires --checkpoint.
This assay does not supply a natural-language CoT monitor or infer deception.
"""
from __future__ import annotations
import argparse
import json
import inspect
import platform
from pathlib import Path
import numpy as np
from research_gym.analysis.monitorability_gap import (
    strongest_null_direction, null_project, torch_jacobians, finite_effects,
)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument("--smoke", action="store_true")
    source.add_argument("--checkpoint",type=Path)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--items",type=int,default=4)
    p.add_argument("--visits",type=int,nargs="+",default=[1,2,4,8])
    p.add_argument("--horizon",type=int,default=4)
    p.add_argument("--relative-radii",type=float,nargs="+",default=[1e-4,1e-3,1e-2])
    p.add_argument("--rank-rtol",type=float,default=1e-9)
    p.add_argument("--rank-atol",type=float,default=1e-11)
    p.add_argument("--device",default="cpu")
    p.add_argument("--dtype",choices=["float32","float64"],default="float64")
    p.add_argument("--seed",type=int,default=941101)
    p.add_argument("--observation",choices=["cursor_logits","all_token_logits"],default="cursor_logits")
    args=p.parse_args()
    if args.seed < 0 or not np.isfinite([args.rank_rtol, args.rank_atol]).all() or min(args.rank_rtol, args.rank_atol) < 0:
        p.error("seed and finite rank tolerances must be nonnegative")
    if args.items<1 or args.horizon<1 or min(args.visits)<1:
        p.error("items, horizon, and visits must be positive")
    if any(not np.isfinite(r) or r<=0 for r in args.relative_radii):
        p.error("relative radii must be finite and positive")
    if args.checkpoint is not None and not args.checkpoint.is_file():
        p.error(f"checkpoint does not exist: {args.checkpoint}")
    import torch
    try:
        from research_gym.scripts.loopscope_semantic_echo import (
            load_model, native_trace, native_step, verify_native_trace, sha256_file,
        )
        from research_gym.scripts.loopscope_toy import make_batch
    except ImportError as e:
        raise SystemExit("Apply this bundle to LoopedTransformers rmp-v0-phase2 with the semantic_echo adapter present.") from e
    torch.set_num_threads(min(4,torch.get_num_threads()))
    model,cfg=load_model(args.checkpoint,smoke=args.smoke,seed=args.seed,device=args.device,dtype=args.dtype)
    for param in model.parameters(): param.requires_grad_(False)
    max_loop=max(args.visits)+args.horizon
    tokens, hops=make_batch(model.n_nodes,args.items,args.seed,max_loop)
    tokens=tokens.to(args.device)
    tolerance=1e-8 if args.dtype=="float64" else 2e-5
    with torch.no_grad():
        embedding,states=native_trace(model,tokens,max_loop)
        parity=verify_native_trace(model,tokens,states,rtol=tolerance,atol=tolerance)
    rows=[]
    for item in range(args.items):
        for t in sorted(set(args.visits)):
            state=states[t,item:item+1].detach()
            fixed_embedding=embedding[item:item+1].detach()
            def center(logits): return logits-logits.mean(dim=-1,keepdim=True)
            def observation(h):
                z=h[:,-1] if args.observation=="cursor_logits" else h
                return center(model.readout(z))
            def future(h):
                for _ in range(args.horizon): h=native_step(model,h,fixed_embedding)
                return center(model.readout(h[:,-1]))
            c,b=torch_jacobians(state,observation,future)
            direction=strongest_null_direction(c,b,rtol=args.rank_rtol,atol=args.rank_atol)
            rng=np.random.default_rng(args.seed+100000*item+t)
            noise=rng.standard_normal(state.numel())
            null_noise,_=null_project(noise[None],c,rtol=args.rank_rtol,atol=args.rank_atol)
            gauge=np.broadcast_to(rng.standard_normal((*state.shape[:-1],1)),tuple(state.shape)).reshape(-1).copy()
            dirs={"optimized_observation_null":direction.direction,
                  "random_full":noise,"random_observation_null":null_noise[0],
                  "per_token_constant_shift":gauge,"zero_write":np.zeros(state.numel())}
            o0=observation(state).detach().cpu().numpy()
            f0=future(state).detach().cpu().numpy()
            baseline_label=int(f0.reshape(-1).argmax())
            expected=int(hops[item,t+args.horizon-1])
            scale=float(torch.linalg.vector_norm(state))
            effects=[]
            for kind,vector in dirs.items():
                norm=float(np.linalg.norm(vector))
                for relative in args.relative_radii:
                    radius=relative*scale
                    for sign in [-1,1]:
                        delta=np.zeros_like(vector) if norm<=1e-14 else sign*radius*vector/norm
                        patched=state+torch.as_tensor(delta.reshape(state.shape),dtype=state.dtype,device=state.device)
                        with torch.no_grad():
                            oo=observation(patched).cpu().numpy()
                            ff=future(patched).cpu().numpy()
                        effects.append({"kind":kind,"relative_radius":relative,"sign":sign,
                            "requested_l2":radius,"constructed_delta_l2":float(np.linalg.norm(delta)),
                            "actual_l2":float(torch.linalg.vector_norm(patched-state)),
                            "zero_direction":bool(norm<=1e-14),
                            **finite_effects(o0,oo,f0,ff),
                            "observation_labels_unchanged":bool(np.array_equal(o0.argmax(-1),oo.argmax(-1))),
                            "future_prediction":int(ff.reshape(-1).argmax()),
                            "future_prediction_changed":bool(ff.reshape(-1).argmax()!=baseline_label),
                            "future_oracle_correct":bool(ff.reshape(-1).argmax()==expected)})
            receipt=direction.receipt()
            # Direction vectors can be large; exact Jacobian diagnostics are retained.
            receipt.pop("direction")
            rows.append({"item":item,"visit":t,"horizon":args.horizon,
                         "oracle_future":expected,"baseline_future_prediction":baseline_label,
                         "baseline_oracle_correct":baseline_label==expected,"state_l2":scale,
                         "local_direction":receipt,"finite_interventions":effects})
    dependency_paths = {"model": Path(inspect.getfile(type(model))),
                        "native_adapter": Path(inspect.getfile(native_trace)),
                        "assay": Path(__file__),
                        "analysis": Path(inspect.getfile(strongest_null_direction))}
    result={"experiment":"RMP-MG-01","schema_version":2,
            "dependencies":{key:{"path":str(path),"sha256":sha256_file(path)}
                            for key,path in dependency_paths.items()},
            "status":"UNTRAINED_SMOKE" if args.smoke else "CHECKPOINT_ASSAY",
            "observation_scope":args.observation,
            "natural_language_monitor_evaluated":False,
            "global_indistinguishability_claim":False,
            "cfg":cfg,"arguments":{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
            "checkpoint_sha256":sha256_file(args.checkpoint) if args.checkpoint else None,
            "native_parity":parity,"python":platform.python_version(),"torch":torch.__version__,
            "rows":rows}
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/"results.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"status":result["status"],"items":args.items,"visit_items":len(rows),
                      "native_parity":parity["passed"],"output":str(args.out/"results.json")},indent=2))


if __name__=="__main__": main()
