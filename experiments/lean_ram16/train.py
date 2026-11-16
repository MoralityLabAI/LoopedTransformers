"""RAM-16 fixed-budget compact 23-family model training.

Conditional on independently THIRD certified source-macro positives and
registered gate. Every task row comes from original immutable RAM08 TRAIN,
with a whole-root task validation split disjoint from source-macro modules.
Task-only and structural arms use byte-identical initialization per seed.
No TEST source, goal, or reference proof is accessible to this job.
"""
from __future__ import annotations
import argparse,collections,copy,hashlib,io,json,random,sys,time,zipfile
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
import old_model_frozen as old
from native_replay import sha
from collect import jsonwrite,sha_file

HERE=Path(__file__).resolve().parent
CONFIG=json.loads((HERE/"config.json").read_text())
FAMILIES23=old.FAMILIES+("nat_induction","simp_prior","rewrite_prior")
PROXY_NAMES={"nat_induction":"induction_list",
             "simp_prior":"simp_all_only_visible",
             "rewrite_prior":"rw_local"}
STOP_NOTE="TRAIN-native-Lean triple cert macro; TEST source/goal never opened"
SOURCE_SALT="RAM16-OLDTASK-VALIDATION-20261116/"


def state_fingerprint(state:dict)->str:
    h=hashlib.sha256()
    for name,tensor in sorted(state.items()):
        h.update(name.encode()+b"\0")
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class StructuralPolicy(old.Policy):
    """Exact prior shared policy, widened from 20 to 23 family rows."""
    def __init__(self,config):
        super().__init__(config)
        w=config.width
        previous=self.family.weight.detach().clone()
        self.family=nn.Embedding(23,w)
        with torch.no_grad():
            self.family.weight[:20].copy_(previous)
            for index,family in enumerate(FAMILIES23[20:],start=20):
                proxy=old.FAMILIES.index(PROXY_NAMES[family])
                self.family.weight[index].copy_(
                    previous[proxy]+.01*torch.randn_like(previous[proxy]))
        if len(FAMILIES23)!=23:raise AssertionError("old family ID count changed")


def encode(rows:list[dict])->dict:
    contexts=[];goals=[];pending=[];actions=[];families=[];counts=[]
    for row in rows:
        state=row["before"]
        active=state["goals"]
        if not active or any(x.count("⊢")!=1 for x in active):
            raise ValueError("invalid original Lean-printed goal")
        context,goal=active[0].split("⊢",1)
        contexts.append(old.token_ids(context,96,True))
        goals.append(old.token_ids(goal,96))
        pending.append(old.token_ids(" || ".join(x.split("⊢",1)[1] for x in active[1:]),128))
        actions.append(old.token_ids(row["script"],96))
        families.append(FAMILIES23.index(row["family"]))
        counts.append([min(8,len(active))/8.,float(len(active)>1)])
    return dict(context=torch.tensor(contexts,dtype=torch.long),
        goal=torch.tensor(goals,dtype=torch.long),
        pending=torch.tensor(pending,dtype=torch.long),
        action=torch.tensor(actions,dtype=torch.long),
        family=torch.tensor(families,dtype=torch.long),
        counts=torch.tensor(counts,dtype=torch.float32))


def task_split(train:list[dict]):
    macro_mods=set(CONFIG["source_modules"]["train"])|set(CONFIG["source_modules"]["dev"])
    trainrows=[];devrows=[];rootset_train=set();rootset_dev=set()
    for row in train:
        if row.get("development"):continue
        file=row["source_file"]
        name=file.removeprefix("MathCan/").removesuffix(".lean")
        key=file+":"+row["theorem"]
        take_dev=name not in macro_mods and int(hashlib.sha256(
            (SOURCE_SALT+key).encode()).hexdigest(),16)%6==0
        (devrows if take_dev else trainrows).append(row)
        (rootset_dev if take_dev else rootset_train).add(key)
    if not trainrows or not devrows or rootset_train & rootset_dev:
        raise AssertionError("registered task-only old-TRAIN theorem split is empty or leaking")
    return trainrows,devrows,dict(original_old_train_rows=len(train),
            task_train_rows=len(trainrows),task_dev_rows=len(devrows),
            task_train_roots=len(rootset_train),task_dev_roots=len(rootset_dev),
            root_disjoint=True,dev_salt=SOURCE_SALT)


def load(ram08:Path,third:Path):
    if sha_file(ram08)!=old.ARCHIVE_SHA:
        raise AssertionError("original RAM08 training ZIP SHA does not match immutable pin")
    gate=json.loads((third/"independent_gate.json").read_text())
    if gate["status"]!="passed_compiler_replay_gate_admitted" or gate["gate_admitted"] is not True:
        raise AssertionError("full original TRAIN macro gate was not admitted")
    if gate["source_heldout_test_accessed"] is not False:
        raise AssertionError("unregistered TEST source access in original macro training artifact")
    positive_file=third/"independently_certified_positive_actions.jsonl"
    if sha_file(positive_file)!=gate["validated_positive_data_sha256"]:
        raise AssertionError("independent native positive curriculum SHA changed")
    certified=[json.loads(s) for s in positive_file.read_text().splitlines() if s.strip()]
    if len(certified)!=gate["all_triple_compiled_positives"]:
        raise AssertionError("independently replayed macro positivity count differs")
    source_rows=[]
    with zipfile.ZipFile(ram08) as archive:
        content=archive.read("data/transitions.jsonl")
        if sha(content)!=old.DATA_SHA:raise AssertionError("original old task compiler effects changed")
        audit=json.loads(archive.read("data/independent_audit.json"))
        if audit["status"]!="passed" or audit["checked"]!=20930:
            raise AssertionError("original RAM08 native effect audit not passed")
        source_rows=[json.loads(t) for t in content.decode().splitlines() if t.strip()]
    if len(source_rows)!=20930:raise AssertionError("original mathcan compiler dataset incomplete")
    task,taskdev,task_receipt=task_split([r for r in source_rows if not r["development"]])
    modules_train=set(CONFIG["source_modules"]["train"])
    modules_dev=set(CONFIG["source_modules"]["dev"])
    macro_train=[];macro_dev=[]
    for positive in certified:
        role=positive["role"]
        module=positive["source_file"].removeprefix("MathCan/").removesuffix(".lean")
        if (role=="train" and module not in modules_train) or (role=="dev" and module not in modules_dev):
            raise AssertionError("macro positive missing registered source module")
        if role not in ("train","dev") or positive["label"]!=3 or positive["source_compiles"]!=3:
            raise AssertionError("uncertified or nonclosed source was used as a training positive")
        candidate=dict(before=positive["before"],script=positive["script"],
                       family=positive["family"],label=3,delta=1,development=(role=="dev"),
                       source_file=positive["source_file"],theorem=positive["theorem"])
        (macro_train if role=="train" else macro_dev).append(candidate)
    if not macro_train:raise AssertionError("TRAIN macro curriculum empty despite admitted gate")
    taskroots={(r["source_file"],r["theorem"]) for r in taskdev}
    macros={(r["source_file"],r["theorem"]) for r in macro_train}
    if taskroots&macros:raise AssertionError("old task DEV shares root with macro TRAIN")
    task_receipt.update(train_verified_macros=len(macro_train),dev_verified_macros=len(macro_dev),
                        macros_triple_compiled=True,eligible_macro_root_set_size=len(macros),
                        old_dev_used_for_selection=False,
                        compiler_effects_sha256=old.DATA_SHA,
                        source_train_positive_sha256=gate["validated_positive_data_sha256"])
    return task,taskdev,macro_train,macro_dev,task_receipt


def single_loss(model,rows):
    x=encode(rows)
    true=torch.tensor([r["label"] for r in rows],dtype=torch.long)
    delta=torch.tensor([float(r["delta"]) for r in rows],dtype=torch.float32)
    logits,pred=model(x)
    return F.cross_entropy(logits,true)+.2*F.smooth_l1_loss(pred,delta)


@torch.no_grad()
def validation(model,rows:list[dict]):
    model.eval()
    total=0.;n=0
    for start in range(0,len(rows),64):
        portion=rows[start:start+64]
        loss=single_loss(model,portion)
        total+=float(loss)*len(portion);n+=len(portion)
    return total/n


def make_model(name,seed):
    t=CONFIG["learning"]["models"]
    setting=next(x for x in t if x["name"]==name)
    torch.manual_seed(seed)
    m=StructuralPolicy(old.Config(name=name,width=setting["width"],loops=setting["loops"]))
    return m


def draw_batch(task,macro,seed,step,mode,rng):
    # 6 macro positives/64 each optimizer update plus a 7th in 2/5
    # updates: exactly 2560/25600 structural positives in 400 updates.
    n_macro=(6+int(step%5 in (0,2))) if mode=="task_plus_structural" else 0
    n_task=64-n_macro
    batch=[task[rng.randrange(len(task))] for _ in range(n_task)]
    batch.extend(macro[rng.randrange(len(macro))] for _ in range(n_macro))
    rng.shuffle(batch)
    return batch,n_macro


def run(ram08:Path,third:Path,out:Path):
    if out.exists():raise FileExistsError("RAM16 training outputs are single use")
    if CONFIG["learning"]["updates_per_seed"]!=400:raise AssertionError("optimizer budget changed")
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(2)
    task,dev,macro,macro_dev,data_receipt=load(ram08,third)
    out.mkdir(parents=True)
    jsonwrite(out/"source_split_receipt.json",data_receipt)
    result=dict(schema="RAM16-v1-fixed-twelve-model-matched-objective-train",
                status="training",date="2026-11-16",
                original_training_zip_sha256=old.ARCHIVE_SHA,
                old_task_effects_sha256=old.DATA_SHA,
                train_verified_structural_positives=len(macro),
                dev_verified_structural_positives=len(macro_dev),
                original_RAM08_development_used_in_selection=False,
                objective_arms=list(CONFIG["learning"]["objectives"]),models={},
                seeds=list(CONFIG["learning"]["train_seeds"]),whole_module_TEST_touched=False,
                optimizer_updates_each=400,model_fits=[],
                source_split=data_receipt,hidden_original_source_TEST_scores=0)
    jsonwrite(out/"training_report.json",result)
    selected={}
    t0=time.perf_counter()
    for modeldef in CONFIG["learning"]["models"]:
        arch=modeldef["name"]
        for seed in CONFIG["learning"]["train_seeds"]:
            # Instantiate once for each seed and retain the identical initial
            # parameters in both objective arms.
            initial=make_model(arch,seed)
            starting=copy.deepcopy(initial.state_dict())
            initial_sha=state_fingerprint(starting)
            del initial
            params=None
            for objective in CONFIG["learning"]["objectives"]:
                model=make_model(arch,seed)
                if state_fingerprint(model.state_dict())!=initial_sha:
                    raise AssertionError("objective arms did not start with same model weights")
                model.load_state_dict(starting)
                count=sum(x.numel() for x in model.parameters())
                if params is not None and count!=params:raise AssertionError("objective changed parameter budget")
                params=count
                where=out/"fits"/f"{arch}_{objective}_s{seed}"
                where.mkdir(parents=True,exist_ok=False)
                optim=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.01)
                rng=random.Random(seed)
                bestloss=float("inf");beststep=None;bestsha=None;history=[];macro_used=0
                for step in range(1,401):
                    batch,extra=draw_batch(task,macro,seed,step-1,objective,rng)
                    macro_used+=extra
                    model.train();optim.zero_grad(set_to_none=True)
                    loss=single_loss(model,batch)
                    if not torch.isfinite(loss):raise FloatingPointError("nonfinite RAM16 supervised loss")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                    optim.step()
                    if step%100==0:
                        v=validation(model,dev)
                        history.append(dict(step=step,dev_loss=round(v,8),
                            last_training_loss=round(float(loss.detach()),8)))
                        if v<bestloss:
                            bestloss=v;beststep=step
                            checkpoint=dict(schema=result["schema"],architecture=arch,objective=objective,
                                model_config=vars(model.config),state_dict=model.state_dict(),
                                optimizer_state=optim.state_dict(),rng_state=rng.getstate(),
                                torch_rng_state=torch.get_rng_state(),train_data_sha256=old.DATA_SHA,
                                new_positive_sha256=data_receipt["source_train_positive_sha256"],
                                validation_source_split_sha256=sha_file(out/"source_split_receipt.json"),
                                source_test_accessed=False,seed=seed,selected_step=step,
                                initial_state_fingerprint=initial_sha,validation_loss=v,
                                macro_train_count=len(macro))
                            dest=where/"best.pt";torch.save(checkpoint,dest)
                            bestsha=sha_file(dest)
                            print("RAM16_BEST",arch,objective,seed,step,round(v,6),flush=True)
                last=dict(schema=result["schema"],architecture=arch,objective=objective,
                          model_config=vars(model.config),state_dict=model.state_dict(),
                          optimizer_state=optim.state_dict(),rng_state=rng.getstate(),
                          torch_rng_state=torch.get_rng_state(),seed=seed,step=400,
                          new_positive_sha256=data_receipt["source_train_positive_sha256"],
                          train_data_sha256=old.DATA_SHA,source_test_accessed=False)
                torch.save(last,where/"last.pt")
                if beststep is None:raise AssertionError("no original DEV checkpoint selected")
                expected_macro=(2560 if objective=="task_plus_structural" else 0)
                if macro_used!=expected_macro:
                    raise AssertionError("registered 10-percent novelty sample fraction altered")
                row=dict(architecture=arch,objective=objective,seed=seed,parameters=count,
                   training_steps=400,macro_examples_sampled=macro_used,
                   starting_state_sha256=initial_sha,best_step=beststep,
                   best_validation_loss=bestloss,
                   best_checkpoint=str((where/"best.pt").relative_to(out)),best_checkpoint_sha256=bestsha,
                   last_checkpoint_sha256=sha_file(where/"last.pt"),dev_history=history,
                   torch_cpu_version=torch.__version__,test_cases_scored=0)
                result["model_fits"].append(row)
                jsonwrite(out/"training_report.json",result)
                print("RAM16_FIT",json.dumps({k:row[k] for k in ("architecture","objective","seed","parameters","best_step","best_validation_loss","macro_examples_sampled")},sort_keys=True),flush=True)
    for modeldef in CONFIG["learning"]["models"]:
        arch=modeldef["name"]
        for objective in CONFIG["learning"]["objectives"]:
            options=[r for r in result["model_fits"] if r["architecture"]==arch and r["objective"]==objective]
            best=min(options,key=lambda z:(z["best_validation_loss"],z["seed"],z["best_step"]))
            selected[f"{arch}_{objective}"]=dict(best)
    a=next(r["parameters"] for r in result["model_fits"] if r["architecture"]=="loop64")
    b=next(r["parameters"] for r in result["model_fits"] if r["architecture"]=="flat71")
    ratio=abs(b-a)/a
    if ratio>.015:
        raise AssertionError("registered flat and recurrent parameter match exceeds 1.5%")
    result.update(status="complete",fits=len(result["model_fits"]),
        selected=selected,relative_parameter_difference=ratio,
        elapsed_cpu_train_seconds=round(time.perf_counter()-t0,3),heldout_proof_cases_scored=0)
    jsonwrite(out/"training_report.json",result)
    print("RAM16_TRAIN_COMPLETE",json.dumps({k:result[k] for k in ("status","fits","relative_parameter_difference","heldout_proof_cases_scored")}),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ram08",type=Path,required=True)
    p.add_argument("--third",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    run(a.ram08.resolve(),a.third.resolve(),a.out.resolve())
