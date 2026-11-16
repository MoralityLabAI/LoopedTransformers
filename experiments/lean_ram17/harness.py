"""RAM-17 validated original Lean proof search with 23-family trained rankers.

Reuses RAM15's previously triple-audited source-level search, mailbox and
in-budget paired certificate implementation. Overrides only its frozen policy
scores, source/goal hashes and registered per-theorem budgets. New source TEST
is opened only after all frozen trained checkpoint SHA checks have succeeded.
"""
from __future__ import annotations
import hashlib,importlib.util,io,json,os,subprocess,sys,zipfile
from pathlib import Path
sys.path.insert(1,str(Path(__file__).resolve().parent.parent/"lean_ram16"))
import torch
import old_model_frozen as old
import train as trained
import eval_grammar as grammar

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
SRC=Path(os.environ.get("RAM17_MATHCAN",str(ROOT/"_mathcan_test"))).resolve()
LEAN=Path(os.environ.get("RAM17_LEAN","/tmp/lean-4.35.0-linux/bin/lean")).resolve()
OUT=Path(os.environ.get("RAM17_EVALUATION_OUT",str(ROOT/"ram16_evaluation"))).resolve()
OLDZIP=Path(os.environ.get("RAM17_ORIGINAL_RAM08",str(ROOT/"ram08-frozen-original.zip"))).resolve()
FITS=Path(os.environ.get("RAM17_FROZEN_FITS",str(ROOT/"_ram16_training"))).resolve()
CONFIG=json.loads((HERE/"config.json").read_text())
FROZEN=json.loads((HERE/"LOCKED_CHECKPOINTS.json").read_text())
LOCKED_SOURCE=json.loads((HERE/"LOCKED_TEST_MANIFEST.json").read_text())
if LOCKED_SOURCE["selected_before_goal_state_access"] is not True or FROZEN["test_cases_scored"]!=0:
    raise AssertionError("training freeze/TEST source-order lock not pristine")
if LOCKED_SOURCE["test_source_commit"]!=CONFIG["test_source_commit"]:
    raise AssertionError("source TEST pin changed")
if len(FROZEN["all_twelve_models_sha256"])!=12 or len(FROZEN["all_best_and_last"])!=12:
    raise AssertionError("missing preregistered paired model checkpoint freeze")
if LOCKED_SOURCE["checkpoint_lock_bytes_sha256"]!=hashlib.sha256((HERE/"LOCKED_CHECKPOINTS.json").read_bytes().strip()).hexdigest():
    raise AssertionError("original TEST name lock references another checkpoint lock")
# The checkpoint freeze and source-name manifest must already exist in Git.
for path in ("LOCKED_CHECKPOINTS.json","LOCKED_TEST_MANIFEST.json"):
    committed=subprocess.check_output(["git","show",f"HEAD:experiments/lean_ram17/{path}"],cwd=ROOT).strip()
    if committed!=(HERE/path).read_bytes().strip():
        raise AssertionError("pretest freeze file was not committed before TEST search: "+path)
if subprocess.run(["git","merge-base","--is-ancestor",LOCKED_SOURCE["checkpoint_lock_commit"],"HEAD"],
                  cwd=ROOT,check=False).returncode:
    raise AssertionError("model checkpoint lock commit is not ancestor of TEST source manifest")
if LOCKED_SOURCE["test_source_commit"]!=CONFIG["test_source_commit"]:
    raise AssertionError("source TEST pin changed")
LOCKED={x["module"]:(x["git_blob_sha1"],tuple(x["theorems"]))
        for x in LOCKED_SOURCE["cases"]}
PRIMARY={k:v for k,v in CONFIG["primary"].items() if k not in ("policies","server_restarts")}
SECONDARY={**PRIMARY,**{k:v for k,v in CONFIG["secondary"].items() if k not in ("policies","server_restarts")}}
POLICIES=list(CONFIG["primary"]["policies"])
SECOND=list(CONFIG["secondary"]["policies"])
ABLATED=list(CONFIG["original_grammar_ablation"]["policies"])
ALIASES={
    f"{arch}_{'structural' if obj=='task_plus_structural' else obj}_s{seed}":
        f"{arch}_{obj}_s{seed}"
    for arch in ("loop64","flat71")
    for obj in ("task_only","task_plus_structural")
    for seed in (17,29,43)
}
assert len(POLICIES)==16 and len(SECOND)==8 and len(LOCKED)==5
assert set(ALIASES).issubset(set(POLICIES)) and len(ALIASES)==12

# Load the prior complete Lean proof search as a frozen source utility. Its
# candidate generator is explicitly substituted by the RAM17 shared grammar.
sys.path.insert(0,str(ROOT/"experiments/lean_ram15"))
sys.path.insert(0,str(ROOT/"experiments/lean_ram13"))
sys.modules["ram15_grammar"]=grammar
os.environ["RAM15_MATHCAN"]=str(SRC)
spec=importlib.util.spec_from_file_location("ram16_frozen_original_search",
       ROOT/"experiments/lean_ram15/harness.py")
if spec is None or spec.loader is None:
    raise RuntimeError("original triple-certified RAM15 Lean scorer unavailable")
base=importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
base.grammar=grammar
base.SRC=SRC
base.LEAN=LEAN
base.PROBE=HERE.parent/"lean_ram16"
base.OUT=OUT
base.LOCKED=LOCKED
base.PRIMARY=PRIMARY
base.SECONDARY=SECONDARY
base.POLICIES=POLICIES
base.SECOND=SECOND
make_env=base.make_env
module_sources=base.module_sources
sha=base.sha
git_blob=base.git_blob


class Rankers:
    """Read the same historical task corpus and all twelve sealed paired checkpoint policies."""
    def __init__(self):
        if sha(OLDZIP.read_bytes())!=old.ARCHIVE_SHA:
            raise AssertionError("original old RAM08 private training ZIP SHA mismatch")
        report=json.loads((FITS/"training_report.json").read_text())
        if report["status"]!="complete" or report["fits"]!=12 or report["heldout_proof_cases_scored"]!=0:
            raise AssertionError("RAM17 original twelve-model training freeze was incomplete")
        if report["original_RAM08_development_used_in_selection"] is not False:
            raise AssertionError("unregistered exposed old DEV checkpoint selection")
        selected={
            f"{r['architecture']}_{r['objective']}_s{r['seed']}":r
            for r in report["model_fits"]
        }
        if set(selected)!=set(FROZEN["all_twelve_models_sha256"]) or len(selected)!=12:
            raise AssertionError("twelve-pair checkpoint roster differs from committed pre-TEST manifest")
        protected={f"MathCan/{x}.lean" for x in CONFIG["source_modules"]["test"]}
        if set(report["source_split"]["prospective_test_modules_excluded"])!=protected:
            raise AssertionError("historical training rows include RAM17 prospective TEST modules")
        torch.set_num_threads(2)
        self.trained={}
        self.original=None
        with zipfile.ZipFile(OLDZIP) as z:
            raw=z.read("data/transitions.jsonl")
            if sha(raw)!=old.DATA_SHA:
                raise AssertionError("old RAM08 compiler labels changed")
            oldrows=[json.loads(x) for x in raw.decode().splitlines() if x.strip()]
            if len(oldrows)!=20930:
                raise AssertionError("incomplete original compiler train archive")
            self.frequency,self.shape=old.controls([r for r in oldrows
                if not r["development"] and r["source_file"] not in protected])
            seed,digest=old.EXPECTED["loop64"]
            buf=z.read(f"training/loop64_s{seed}/best.pt")
            if sha(buf)!=digest:
                raise AssertionError("frozen original loop64 proxy bytes changed")
            data=torch.load(io.BytesIO(buf),map_location="cpu",weights_only=True)
            self.original=old.Policy(old.Config(**data["model_config"])).eval()
            self.original.load_state_dict(data["state_dict"])
        for key,stored in sorted(FROZEN["all_best_and_last"].items()):
            info=selected[key]
            frozen=stored
            for field in ("seed","best_step","best_checkpoint","best_checkpoint_sha256","starting_state_sha256"):
                if info[field]!=frozen[field]:
                    raise AssertionError("pre-TEST selected architecture/seed/checkpoint changed")
            file=FITS/info["best_checkpoint"]
            raw=file.read_bytes()
            if sha(raw)!=info["best_checkpoint_sha256"]:
                raise AssertionError("original checkpoint file bytes changed")
            cp=torch.load(io.BytesIO(raw),map_location="cpu",weights_only=True)
            if cp["source_test_accessed"] is not False or cp["seed"]!=frozen["seed"]:
                raise AssertionError("original TEST source touched during paired optimizer fit")
            if cp["initial_state_fingerprint"]!=frozen["starting_state_sha256"] or cp["architecture"]!=frozen["architecture"] or cp["objective"]!=frozen["objective"]:
                raise AssertionError("paired optimizer checkpoint metadata differs from pretest freeze")
            if cp["source_test_accessed"] is not False or cp["seed"]!=frozen["seed"]:
                raise AssertionError("unregistered TEST exposure during original optimizer fit")
            if cp["schema"]!=report["schema"] or cp["selected_step"]!=frozen["best_step"]:
                raise AssertionError("original DEV-only model selection changed")
            model=trained.StructuralPolicy(old.Config(**cp["model_config"])).eval()
            model.load_state_dict(cp["state_dict"])
            self.trained[key]=model
        n_recurrent=sum(p.numel() for p in self.trained["loop64_task_only"].parameters())
        n_flat=sum(p.numel() for p in self.trained["flat71_task_only"].parameters())
        if abs(n_flat-n_recurrent)/n_recurrent>.015:
            raise AssertionError("tied recurrence and flat comparator parameter mismatch")

    @torch.no_grad()
    def score(self,policy,state,pool):
        if policy=="frozen_loop64_proxy":
            model=self.original
            rows=[dict(before=state,family=grammar.family_proxy(f,s),script=s)
                  for f,s in pool]
            convert=old.encode
        else:
            model=self.trained[ALIASES[policy]]
            rows=[dict(before=state,family=f,script=s) for f,s in pool]
            convert=trained.encode
        result=[]
        for i in range(0,len(rows),16):
            batch=convert(rows[i:i+16])
            logits,delta=model(batch);p=logits.softmax(-1)
            value=3*p[:,3]+1.5*p[:,1]+.3*p[:,2]-.6*p[:,0]+.4*delta.clamp(-3,3)
            result.extend(value.tolist())
        return result

    def rank(self,policy,state,pool):
        if policy=="native":return pool
        if policy in ("frequency","shape_frequency"):
            rows=[dict(before=state,family=grammar.family_proxy(f,s),script=s)
                  for f,s in pool]
            fn=self.frequency if policy=="frequency" else self.shape
            scores=[fn(r) for r in rows]
        else:
            scores=self.score(policy,state,pool)
        return [pool[i] for i in sorted(range(len(pool)),key=lambda i:(-scores[i],i))]


def search(source,theorem,ranker,policy,budget,out,*,ablation=False):
    grammar.set_context(source,theorem,ablation=ablation)
    result=base.search(source,theorem,ranker,policy,budget,out,ablation=ablation)
    if result["budget"]!=budget:raise AssertionError("registered budget mutated")
    if result["logical_evaluations"]>budget["evaluations"]:
        raise AssertionError("source search exceeded registered 48/96 tactic budget")
    if result["accepted"] and result["elapsed_seconds"]>budget["seconds"]:
        raise AssertionError("proof scored after registered deadline")
    return result
