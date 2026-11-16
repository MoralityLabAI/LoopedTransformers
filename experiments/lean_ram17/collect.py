"""RAM-17 TRAIN/DEV-only native Lean4.35 source-macro collection.

Only original selected source paths and theorem names may be read.
No protected RAM17 TEST theorem source is materialized in this stage.
A completed original theorem creates a *positive*, never an implicit
negative for competing macros whose independent full proof was absent.
"""
from __future__ import annotations
import argparse,collections,hashlib,json,os,re,subprocess,sys,time
from pathlib import Path

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"lean_ram08"))
sys.path.insert(0,str(HERE.parent/"lean_ram16"))
sys.path.insert(0,str(HERE))
from runtime import ResourceGuard
from goal_probe import observe_original_goal
import grammar
from native_replay import render_source,compile_once,sha,axiom_inventory

ROLE_ORDER=("train","dev")
PROTECTED=tuple(json.loads((HERE/"config.json").read_text())["protected_test_modules_older"])
SCHEMA="RAM17-first-authentic-macro-train-source"
MAX_COMPILATION=15.0

def jsonwrite(path:Path,item):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(item,sort_keys=True,ensure_ascii=False,indent=2)+"\n",encoding="utf8")

def jsonappend(path:Path,item):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("a",encoding="utf8") as f:
        f.write(json.dumps(item,sort_keys=True,ensure_ascii=False)+"\n")

def sha_file(path:Path):return hashlib.sha256(path.read_bytes()).hexdigest()

def protected_training_checkout(mathcan:Path):
    version=subprocess.check_output(["git","rev-parse","HEAD"],cwd=mathcan,text=True).strip()
    config=json.loads((HERE/"config.json").read_text())
    if version!=config["train_source_commit"]:
        raise ValueError("protected TRAIN source checkout SHA mismatch")
    if (mathcan/"lean-toolchain").read_text().strip() != config["lean_toolchain"]:
        raise ValueError("original Lean4.35 toolchain changed")
    for name in PROTECTED:
        if (mathcan/"MathCan"/(name+".lean")).exists():
            raise ValueError("pre-training protected TEST module is still materialized: "+name)
    for name in config["chronologically_new_test_modules"]:
        if (mathcan/"MathCan"/(name+".lean")).exists():
            raise ValueError("chronologically new TEST module in TRAIN checkout: "+name)
    active=set(config["source_modules"]["train"]) | set(config["source_modules"]["dev"])
    if active & (set(PROTECTED) | set(config["chronologically_new_test_modules"])):
        raise AssertionError("TRAIN/DEV registry overlaps protected TEST modules")
    for module in active:
        path=mathcan/"MathCan"/(module+".lean")
        if not path.is_file():raise FileNotFoundError(module)
        imports=re.findall(r"(?m)^import\s+MathCan\.([A-Za-z_][A-Za-z0-9_]*)",path.read_text(encoding="utf8"))
        if set(imports) & set(PROTECTED):
            raise ValueError("protected TEST module is a direct TRAIN/DEV import: "+module)
    return config

def select_original_theorems(source:str,howmany:int)->list[str]:
    # Source order, top-level declarations only, with comments erased.
    masked=grammar.mask_comments(source)
    names=[m.group(1) for m in re.finditer(
        r"^(?:theorem|lemma)\s+([A-Za-z_][A-Za-z0-9_']*)",masked,re.M)]
    return list(dict.fromkeys(names))[:howmany]

def actual_root_goal(source:str,theorem:str,path:Path,guard,lean:str):
    return observe_original_goal(lean,source,theorem,path,guard,timeout=15.)

def collect(mathcan:Path,lean:str,out:Path):
    if out.exists():raise FileExistsError("all experiment output paths are single-use")
    config=protected_training_checkout(mathcan)
    if subprocess.run([lean,"--version"],capture_output=True,text=True,check=False).returncode:
        raise ValueError("Lean compiler binary absent")
    out.mkdir(parents=True)
    scratch=out/"native_source_compiler_artifacts"
    scratch.mkdir(exist_ok=False)
    summary=dict(schema=SCHEMA,status="collecting",source=config["train_source_commit"],
                 toolchain=config["lean_toolchain"],source_roles={},roots=[],
                 source_heldout_test_accessed=False, original_RAM08_PEF_scores=0,
                 RAM10_first_test_scores=0, subsequent_native_third_audit=False)
    start=time.monotonic()
    deadline=start+config["collector"]["run_wall_minutes"]*60
    rows_count=0
    positives=[]
    with ResourceGuard(out/"resource_samples.jsonl") as guard:
        for role in ROLE_ORDER:
            roots=[]
            manifest=config["source_modules"][role]
            for module,expected_blob in manifest.items():
                if time.monotonic()>deadline:raise TimeoutError("registered source-only collection wall limit")
                path=mathcan/"MathCan"/(module+".lean")
                raw=path.read_bytes()
                if grammar.git_blob(raw)!=expected_blob:
                    raise ValueError("TRAIN/DEV registered original source Git blob mismatch: "+module)
                source=raw.decode("utf8")
                theorem_names=select_original_theorems(source,config["selection"][role+"_first"])
                if not theorem_names:raise AssertionError("registered source module has zero selectable original theorems")
                for theorem in theorem_names:
                    if time.monotonic()>deadline:raise TimeoutError("registered 90-minute TRAIN-only source cap")
                    root_index=len(summary["roots"])
                    root_folder=out/"native_roots"/f"{root_index:03d}"
                    original_state=actual_root_goal(source,theorem,root_folder/"goal",guard,lean)
                    before=original_state["goals"]
                    actions=grammar.generate(source,theorem)
                    original_actions=grammar.ram14_frozen.generate(source,theorem) if hasattr(grammar,"ram14_frozen") else grammar.old.generate(source,theorem)
                    if actions[:len(original_actions)]!=original_actions:
                        raise AssertionError("frozen historical RAM14 candidate action prefix changed")
                    if len(actions)>config["collector"]["max_total_per_root"]:
                        raise AssertionError("registered 48 candidate cap exceeded")
                    actioncounts=collections.Counter(a.family for a in actions)
                    if any(v>config["collector"]["max_each_family"] for v in actioncounts.values()):
                        raise AssertionError("registered per-family 16 cap exceeded")
                    rowbase=dict(role=role,root_index=root_index,
                        source_file="MathCan/"+module+".lean",theorem=theorem,
                        original_source_git_blob_sha1=expected_blob,
                        original_source_sha256=sha(raw),
                        original_target_statement_sha256=grammar.describe(source,theorem)["statement_sha256"],
                        original_before_state=dict(accepted=True,goals=original_state["goals"],
                                                   definitions=original_state["definitions"]),
                        original_source_root_prefix=[],generated_count=len(actions))
                    success_count=0
                    for index,action in enumerate(actions):
                        if time.monotonic()>deadline:raise TimeoutError("TRAIN source collection full-run cap")
                        case= scratch/role/f"{root_index:03d}"/f"{index:03d}"
                        compiled_source=render_source(source,theorem,[action.script]).encode("utf8")
                        first=compile_once(mathcan,compiled_source,case/"first",MAX_COMPILATION)
                        second=None
                        complete=False
                        if first["status"]=="accepted":
                            second=compile_once(mathcan,compiled_source,case/"independent"/"second",MAX_COMPILATION)
                            complete=(second["status"]=="accepted" and
                                      first["source_sha256"]==second["source_sha256"] and
                                      first["axioms"]==second["axioms"])
                        row=dict(**rowbase,action_index=index,candidate=action.as_dict(),
                            first_compiler=first,second_compiler=second,
                            source_status="double_full_source_compiled" if complete else
                                "compiler_timeout_unknown" if first["status"]=="compiler_timeout_unknown"
                                or (second and second["status"]=="compiler_timeout_unknown")
                                else "unverified_nonclosure_censored",
                            full_source_positive=complete,unknown_is_censored=True,
                            source_candidate_sha256=sha(compiled_source))
                        jsonappend(out/"first_source_effects.jsonl",row)
                        rows_count+=1
                        if complete:
                            success_count+=1
                            positive=dict(source_file=rowbase["source_file"],theorem=theorem,
                                original_source_sha256=sha(raw),source_root_role=role,
                                before=rowbase["original_before_state"],action_family=action.family,
                                action_script=action.script,full_source_compiles=2,
                                candidate_provenance=list(action.witnesses),label=3,delta=1,
                                full_source_sha256=sha(compiled_source),
                                exact_original_source_target=True,third_replay_pending=True,
                                candidate_source="source header and previously declared statements")
                            positives.append(positive)
                            jsonappend(out/"two_compile_positive_candidates.jsonl",positive)
                        print("RAM17_NATIVE",role,root_index,theorem,index,
                              action.family,first["status"],"certified_twice" if complete else "censored",flush=True)
                    info=dict(role=role,module=module,theorem=theorem,original_git_blob=expected_blob,
                              original_state_goals=len(before),candidate_count=len(actions),
                              candidate_counts=dict(actioncounts),two_compile_positives=success_count)
                    summary["roots"].append(info);roots.append(info)
                    jsonwrite(out/"first_collection_receipt.json",summary)
                print("RAM17_MODULE",role,module,len(theorem_names),flush=True)
            summary["source_roles"][role]=len(roots)
        summary.update(status="complete_pending_independent_third_audit",
            candidate_compilations=rows_count,positive_source_twice_count=len(positives),
            positive_by_family=dict(collections.Counter(x["action_family"] for x in positives)),
            collected_theorem_roots=len(summary["roots"]),
            elapsed_seconds=round(time.monotonic()-start,3),
            source_effects_sha256=sha_file(out/"first_source_effects.jsonl"),
            positive_data_sha256=sha_file(out/"two_compile_positive_candidates.jsonl")
               if positives else None, resource_guard=guard.receipt())
        jsonwrite(out/"first_collection_receipt.json",summary)
    # Files on the original source checkout contain only this TRAIN experiment,
    # and the canonical evidence is copied to the private Actions artifact.
    # Compiler sources and logs were written in the original output tree as each
    # attempt ran, including when a later TRAIN-only root raises an error.
    print("RAM17_TRAIN_FIRST",json.dumps({k:v for k,v in summary.items() if k!="roots"},sort_keys=True),flush=True)
    return summary

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--mathcan",type=Path,required=True)
    parser.add_argument("--lean",default="lean")
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    collect(args.mathcan.resolve(),args.lean,args.out.resolve())
