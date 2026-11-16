"""Select original TEST theorem names only after the 12-model lock is committed.

This script is the first authorized TEST file reader in the RAM17 sequence.
It selects names from a deterministic source-order grammar and never starts Lean,
observes a goal, ranks tactics, or reads a reference proof as a feature.
"""
from __future__ import annotations
import argparse,hashlib,json,re,subprocess
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"lean_ram16"))
sys.path.insert(0,str(HERE))
import grammar

CONFIG=json.loads((HERE/"config.json").read_text())
LOCK_REPO_PATH="experiments/lean_ram17/LOCKED_CHECKPOINTS.json"

def git(*args,cwd:Path)->bytes:
    result=subprocess.run(["git",*args],cwd=cwd,stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE,check=False)
    if result.returncode:
        raise ValueError("Git source/lock history verification failed: "+result.stderr.decode(errors="replace")[-250:])
    return result.stdout.strip()

def source_test_manifest(repo:Path, mathcan:Path, checkpoint_lock:Path,
                         checkpoint_commit:str)->dict:
    if not re.fullmatch(r"[0-9a-f]{40}",checkpoint_commit):
        raise ValueError("checkpoint lock must reference a full committed Git SHA")
    committed=git("show",f"{checkpoint_commit}:{LOCK_REPO_PATH}",cwd=repo)
    raw=checkpoint_lock.read_bytes().strip()
    if committed!=raw:
        raise ValueError("pretest 12-model lock bytes differ from Git commit")
    lock=json.loads(raw)
    if lock["schema"]!="RAM17-pretest-checkpoint-candidate-v1":
        raise ValueError("unregistered prospective lock schema")
    if not (lock["test_cases_scored"]==0 and lock["original_test_goals_accessed"]==0):
        raise ValueError("scoring/TEST goal access preceded committed training freeze")
    if len(lock["all_twelve_models_sha256"])!=12 or lock["train_source_commit"]!=CONFIG["train_source_commit"]:
        raise ValueError("incomplete source-only model freeze")
    if lock["test_source_commit_sealed"]!=CONFIG["test_source_commit"]:
        raise ValueError("TEST source changed after model freeze")
    if git("rev-parse","HEAD",cwd=mathcan).decode()!=CONFIG["test_source_commit"]:
        raise ValueError("MathCan TEST checkout commit differs from preregistration")
    if (mathcan/"lean-toolchain").read_text().strip()!=CONFIG["lean_toolchain"]:
        raise ValueError("Lean TEST toolchain changed")
    cases=[]
    for module,blob in CONFIG["source_modules"]["test"].items():
        file=mathcan/"MathCan"/f"{module}.lean"
        source_bytes=file.read_bytes()
        if grammar.git_blob(source_bytes)!=blob:
            raise ValueError("frozen TEST Git blob changed: "+module)
        masked=grammar.mask_comments(source_bytes.decode("utf8"))
        names=[m.group(1) for m in re.finditer(
            r"^(?:theorem|lemma)\s+([A-Za-z_][A-Za-z0-9_']*)",masked,re.M)]
        names=list(dict.fromkeys(names))[:CONFIG["selection"]["test_first"]]
        if len(names)!=2:
            raise ValueError("TEST source cannot supply first two unambiguous top-level theorems: "+module)
        cases.append(dict(module=module,source_file=f"MathCan/{module}.lean",
                          git_blob_sha1=blob,theorems=names,
                          original_source_sha256=hashlib.sha256(source_bytes).hexdigest()))
    return dict(schema="RAM17-original-theorem-names-locked",
                test_source_commit=CONFIG["test_source_commit"],
                train_source_commit=CONFIG["train_source_commit"],
                checkpoint_lock_commit=checkpoint_commit,
                checkpoint_lock_bytes_sha256=hashlib.sha256(raw).hexdigest(),
                sealed_before_TEST_goal_access=True,
                selected_before_goal_state_access=True,
                observed_test_goal_count=0,
                source_selected_after_checkpoint_commit=True,
                time_new_source_modules=CONFIG["chronologically_new_test_modules"],
                source_selector="first two top-level theorem/lemma declarations, comments and strings masked",
                cases=cases,case_count=sum(len(x["theorems"]) for x in cases))

def run(repo:Path,mathcan:Path,lock:Path,commit:str,out:Path):
    if out.exists():raise FileExistsError("original TEST lock output must be new")
    item=source_test_manifest(repo,mathcan,lock,commit)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(item,sort_keys=True,indent=2)+"\n",encoding="utf8")
    return item

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--repo",required=True,type=Path)
    p.add_argument("--mathcan",required=True,type=Path)
    p.add_argument("--checkpoint-lock",required=True,type=Path)
    p.add_argument("--checkpoint-commit",required=True,type=str)
    p.add_argument("--out",required=True,type=Path)
    a=p.parse_args()
    o=run(a.repo.resolve(),a.mathcan.resolve(),a.checkpoint_lock.resolve(),
          a.checkpoint_commit,a.out.resolve())
    print("RAM17_TEST_NAMES_SELECTED_AFTER_COMMIT",o["case_count"],"Lean goal accesses",o["observed_test_goal_count"])
