"""OLR-METTA-02 native multi-witness proof-tactic learning.

Integrates DataGoblin's DG-METTA-12 A/B/C controls with a lightweight OLR
tied recurrent scorer and independently certified Lean proof actions.
Native Hyperon curates TRAIN source roots. Held roots are scored only after
fitted policy decisions are frozen; every successful script compiles in Lean.
This CPU study is a reference pilot, not native courier transfer.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
import subprocess
import tempfile

import torch
from torch import nn

from lcgrt import config
from lcgrt.lean_repl import LeanRepl
from .data import canonical, digest, sha256_file
from .live_lean import candidate_menu, public_state
from .metta_action_sets import objective, teacher_actions_from_verified_graph
from .metta_obligations import VerifiedRollout, curate_verified

SCHEMA = "ppt.olr-metta-multi-witness.v1"
ACTIONS=("apply r0","apply r1","apply r2")
TRAIN_MODES=("canonical","hashed","set")
SEEDS=(17,29,43)


@dataclass(frozen=True)
class Root:
    split: str
    index: int
    a: int
    b: int
    targets: tuple[int,int,int]
    mode: str

    @property
    def root_id(self):
        return digest([SCHEMA,self.split,self.index,self.a,self.b,
                       self.targets,self.mode])

    def statement(self):
        a,b=self.a,self.b
        parts=" ".join(
            f"(r{i} : ({a} + {b} = {value}) → P → Q → R)"
            for i,value in enumerate(self.targets)
        )
        return f"∀ (P Q R : Prop) (h0 : P) (h1 : Q) {parts}, R"


def generate_roots(split:str,count:int,seed:int):
    if split not in ("train","development") or not 1<=count<=64:
        raise ValueError("invalid root generation request")
    rng=random.Random(f"olr-metta-02/{seed}/{split}")
    seen=set()
    result=[]
    for i in range(count):
        mode=("tie","r0","r1","r2")[i%4]
        for retry in range(500):
            a,b=rng.randint(1,18),rng.randint(1,18)
            bad=[a+b+1,a+b+2,a+b+3]
            true=[a+b if mode=="tie" and n in (0,1) or mode==f"r{n}"
                  else bad[n] for n in range(3)]
            key=(a,b,tuple(true))
            if key not in seen:
                seen.add(key)
                result.append(Root(split,i,a,b,tuple(true),mode))
                break
        else:
            raise ValueError("root generator exhausted unique operands")
    return tuple(result)


def _one_action(lean:LeanRepl,root:Root,action:str):
    if action not in ACTIONS:
        raise ValueError("outside allowed proof tactic menu")
    lean.reset()
    state=lean.init(root.statement(),pp=True)
    visible=public_state(lean,state)
    menu=candidate_menu(lean,state)
    if menu!=ACTIONS:
        raise ValueError(f"native root candidate menu differs: {menu}")
    source_sha=digest(visible)
    first=lean.run(state["sid"],action)
    if first.get("ok") is not True:
        raise ValueError("native rule opening rejected")
    current=first["state"]
    if len(public_state(lean,current)["goals"])!=3:
        raise ValueError("native applied rule has wrong obligation count")
    guard=lean.run(current["sid"],"rfl")
    if guard.get("ok") is not True:
        return {"status":"guard_rejected","native_kernel":False,
                "standalone_verified":False,"axioms":[],
                "total_tactic_calls":2,"state_sha256":source_sha}
    current=guard["state"]
    for tactic in ("apply h0","apply h1"):
        result=lean.run(current["sid"],tactic)
        if result.get("ok") is not True:
            raise ValueError("native true guard failed deterministic proof tail")
        current=result["state"]
    if public_state(lean,current)["goals"]:
        raise ValueError("expected proof to be closed after two fact applications")
    checker=lean.kernel(current["sid"])
    if checker.get("ok") is not True or checker.get("axioms")!=[]:
        raise ValueError("native kernel rejected a proof tail")
    return {"status":"kernel_closed","native_kernel":True,
            "standalone_verified":True,"axioms":[],
            "total_tactic_calls":4,"state_sha256":source_sha}


def verify_root(lean:LeanRepl,root:Root):
    checks={action:_one_action(lean,root,action) for action in ACTIONS}
    hashes={item["state_sha256"] for item in checks.values()}
    if len(hashes)!=1:
        raise ValueError("independent native actions did not see the same root")
    gold=teacher_actions_from_verified_graph(
        ACTIONS,checks,remaining_calls=4,goal_is_kernel_closed=False)
    count=2 if root.mode=="tie" else 1
    if len(gold)!=count:
        raise ValueError("native theorem did not have registered witness count")
    # The full source is generated from known bounded typed public numerals.
    # The binary equality test is a private audit only, never a model feature.
    features=[[root.a/40.,root.b/40.,target/80.] for target in root.targets]
    return {"root_id":root.root_id,"statement":root.statement(),
            "state_sha256":next(iter(hashes)), "legal":ACTIONS,
            "optimal":gold,"features":features,"receipts":checks,
            "split":root.split,"mode":root.mode,"index":root.index}


def compile_proofs(rows:list[tuple[Root,str]], out:Path):
    """Compile each native-verified source root/action as a full Lean theorem."""
    source=["namespace OLRMeTTaTies"]
    for i,(root,action) in enumerate(rows):
        if action not in ACTIONS or any(x<0 or x>100 for x in (root.a,root.b,*root.targets)):
            raise ValueError("outside bounded independently checked statement grammar")
        source.append(f"theorem p{i} {root.statement()[2:-3]} : R := by")
        source.extend(("  "+action,"  rfl","  apply h0","  apply h1"))
    source.append("end OLRMeTTaTies")
    source.extend(f"#print axioms OLRMeTTaTies.p{i}" for i in range(len(rows)))
    text="\n".join(source)+"\n"
    out.write_text(text,encoding="utf-8")
    with tempfile.TemporaryDirectory() as d:
        file=Path(d)/"NativeTies.lean"
        file.write_text(text,encoding="utf-8")
        process=subprocess.run([str(config.LEAN_EXE),str(file)],check=False,
              env=config.lean_env(),capture_output=True,text=True,
              encoding="utf-8",timeout=120)
    data=process.stdout+process.stderr
    clean=sum(f"'OLRMeTTaTies.p{i}' does not depend on any axioms" in data
              for i in range(len(rows)))
    if process.returncode!=0 or clean!=len(rows):
        raise ValueError(f"native multi-witness standalone certifier failed: "
                         f"rc={process.returncode}; clean={clean}/{len(rows)}; {data[-1500:]}")
    return {"independent_proofs":len(rows),"axiom_free":clean,
            "sha256":sha256_file(out)}


class TiedSetScorer(nn.Module):
    """Tied 3-slot recurrent score cell; same architecture for all label arms."""
    def __init__(self,width:int=24):
        super().__init__()
        self.embedding=nn.Sequential(nn.Linear(3,width),nn.GELU(),nn.LayerNorm(width))
        self.role=nn.Parameter(torch.zeros(1,3,width))
        self.message=nn.Sequential(nn.Linear(3*width,2*width),nn.Tanh(),
                                   nn.Linear(2*width,width))
        self.gate=nn.Linear(3*width,width)
        self.norm=nn.LayerNorm(width)
        self.head=nn.Linear(width,1)

    def forward(self,public:torch.Tensor,depth:int=3):
        if public.ndim!=3 or public.shape[1:]!=(3,3) or not 1<=depth<=8:
            raise ValueError("expected Bx3x3 public numeric action features")
        original=self.embedding(public)+self.role
        state=original
        for _ in range(depth):
            shared=state.mean(1,keepdim=True).expand_as(state)
            packet=torch.cat((original,state,shared),dim=-1)
            proposal=self.message(packet)
            gate=self.gate(packet).sigmoid()
            state=self.norm(state+0.25*gate*proposal)
        return self.head(state).squeeze(-1)


def fit(roots:list[dict],seed:int,mode:str,steps:int=40):
    if mode not in TRAIN_MODES or not roots or not 1<=steps<=200:
        raise ValueError("unsupported model training request")
    torch.manual_seed(seed)
    model=TiedSetScorer()
    # Same seed initialization and sampled roots for all loss arms.
    initialization=digest([(k,v.detach().tolist()) for k,v in model.state_dict().items()])
    optimizer=torch.optim.AdamW(model.parameters(),lr=.003)
    rng=torch.Generator().manual_seed(seed+3000)
    choices=torch.randint(len(roots),(steps, min(len(roots),12)),generator=rng)
    losses=[]
    for indices in choices:
        optimizer.zero_grad()
        loss=0.
        for j in indices.tolist():
            row=roots[j]
            features=torch.tensor([row["features"]],dtype=torch.float32)
            logits=model(features)[0]
            loss=loss+objective(logits,row["legal"],row["optimal"],
                  mode=mode,root_id=row["root_id"],
                  state_sha256=row["state_sha256"],model_seed=seed)
        loss=loss/len(indices)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
        losses.append(float(loss.detach()))
    model.eval()
    return model,{"seed":seed,"objective":mode,"initial_sha256":initialization,
                  "steps":steps,"training_root_count":len(roots),
                  "parameter_count":sum(p.numel() for p in model.parameters()),
                  "final_loss":losses[-1]}


def run(out:Path, *, seed:int=20261128,train_roots:int=12,
        development_roots:int=8,steps:int=40):
    if out.exists():
        raise ValueError("out must be a new directory")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.mkldnn.enabled=False
    trains=generate_roots("train",train_roots,seed)
    developments=generate_roots("development",development_roots,seed+31)
    seen={(r.a,r.b,r.targets) for r in trains}
    if any((r.a,r.b,r.targets) in seen for r in developments):
        raise ValueError("train/development source overlap")
    with LeanRepl(timeout=60) as lean:
        if not str(lean.version).startswith("4.19.0"):
            raise ValueError("pinned native Lean 4.19.0 required")
        version=lean.version
        train=[verify_root(lean,r) for r in trains]
        dev=[verify_root(lean,r) for r in developments]
    root_objects={r.root_id:r for r in (*trains,*developments)}
    verified=[(root_objects[row["root_id"]],action) for row in (*train,*dev)
              for action in row["optimal"]]
    out.mkdir(parents=True)
    cert=compile_proofs(verified,out/"standalone.lean")
    # Native MeTTa admission runs once for each TRAIN source root. A rooted
    # action set with 2 equal successes stays one supervised presentation.
    offerings=[
        VerifiedRollout(f"root{i}",f"lineage{i}","train",True,True,True,
                        True,True,True,True,True,len(row["optimal"]),
                        len(ACTIONS),1,-4)
        for i,row in enumerate(train)
    ]
    selected=curate_verified(offerings)
    if len(selected["offer_ids"])!=len(trains):
        raise ValueError("native MeTTa curator unexpectedly dropped a certified root")
    byid={f"root{i}":r for i,r in enumerate(train)}
    curated=[byid[x] for x in selected["offer_ids"]]
    models={}
    model_data={}
    for model_seed in SEEDS:
        initial=None
        for mode in TRAIN_MODES:
            model,meta=fit(curated,model_seed,mode,steps)
            models[model_seed,mode]=model
            model_data[f"{model_seed}:{mode}"]=meta
            if initial is None:
                initial=meta["initial_sha256"]
            elif initial!=meta["initial_sha256"]:
                raise ValueError("unequal initialization in matched label-control arms")
    records=[]
    with LeanRepl(timeout=60) as lean:
        for row in dev:
            r=root_objects[row["root_id"]]
            for model_seed in SEEDS:
                for mode in TRAIN_MODES:
                    with torch.no_grad():
                        scores=models[model_seed,mode](
                            torch.tensor([row["features"]],dtype=torch.float32))[0]
                    action=ACTIONS[int(scores.argmax().item())]
                    # Query outcome is checked by a fresh native Lean session.
                    result=_one_action(lean,r,action)
                    if (result["status"]!=row["receipts"][action]["status"]
                            or result["state_sha256"]!=row["state_sha256"]):
                        raise ValueError("frozen model's new native verdict drifted")
                    records.append({"root_id":row["root_id"],"mode":mode,
                        "model_seed":model_seed,"action":action,
                        "kernel_closed":result["native_kernel"],
                        "tie":len(row["optimal"])>1,"checks":result["total_tactic_calls"]})
    scoreboard={}
    for mode in TRAIN_MODES:
        matched=[x for x in records if x["mode"]==mode]
        scoreboard[mode]={
            "trials":len(matched),
            "certified":sum(x["kernel_closed"] for x in matched),
            "tied_state_trials":sum(x["tie"] for x in matched),
            "tied_state_certified":sum(x["kernel_closed"] and x["tie"] for x in matched),
            "tactic_calls":sum(x["checks"] for x in matched),
        }
    root_manifest={"schema":SCHEMA,"root_seed":seed,
         "train_roots":len(trains),"development_roots":len(developments),
         "tied_train_roots":sum(len(r["optimal"])>1 for r in train),
         "tied_development_roots":sum(len(r["optimal"])>1 for r in dev),
         "lean_version":version,
         "lean_sha256":sha256_file(config.LEAN_EXE),
         "native_source_certificates":cert,
         "native_metta_curation":selected,
         "fitted_arms":model_data,"evaluation":scoreboard,
         "native_query_trials":len(records),
         "source_only_model_training":True,
         "checkpoint_promotion_eligible":False,
         "evidence_class":"native_lean_source_and_hyperon_curation_cpu_training_pilot"}
    (out/"train.jsonl").write_text(
        "".join(canonical(r)+"\n" for r in train),encoding="utf-8")
    (out/"development.jsonl").write_text(
        "".join(canonical(r)+"\n" for r in dev),encoding="utf-8")
    (out/"decisions.jsonl").write_text(
        "".join(canonical(r)+"\n" for r in records),encoding="utf-8")
    with (out/"models.pt").open("xb") as handle:
        torch.save({f"{seed}:{mode}":model.state_dict()
                    for (seed,mode),model in models.items()},handle)
    root_manifest["artifacts"]={
        name:sha256_file(out/name) for name in (
            "standalone.lean","train.jsonl","development.jsonl","decisions.jsonl","models.pt")}
    (out/"manifest.json").write_text(canonical(root_manifest)+"\n",encoding="utf-8")
    return root_manifest


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--seed",type=int,default=20261128)
    p.add_argument("--train",type=int,default=12)
    p.add_argument("--development",type=int,default=8)
    p.add_argument("--steps",type=int,default=40)
    a=p.parse_args(argv)
    result=run(a.out,seed=a.seed,train_roots=a.train,
               development_roots=a.development,steps=a.steps)
    print(canonical({
        k:result[k] for k in ("schema","train_roots","development_roots",
                             "tied_train_roots","tied_development_roots",
                             "native_source_certificates","evaluation")
    }))


if __name__=="__main__":
    main()
