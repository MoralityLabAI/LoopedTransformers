"""Exercise source isolation, a signed two-pass contrast and host failure custody."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
from shutil import copyfile,copytree
import sys

from experiments.nan_phase35.run import lease
from experiments.nan_readout34 import runtime
from experiments.nan_readout34.runtime_plan import write_new
from experiments.nan_readout34.tokenizer_check import sha
from experiments.nan_vocab82.heads import partition_sha
from experiments.nan_vocab83.batches import record_roots
from experiments.nan_vocab83.custody import HOST_BUDGET,vocabulary
from experiments.nan_vocab83.verify import SyntheticHead,baseline_rows

from .host import MODES,collect,score
from .lane import ROUTEMATCH,host,terminal,verified
from .source import config,load,worker_inputs


def sample(source,twins,heads,geometry):
    gold={r["id"]:r["answer"] for r in source["gold"]}
    output=[]
    for i,w in enumerate(("A","B")):
        target=gold[twins[w]]
        foil=gold[twins["B" if w=="A" else "A"]]
        third=next(c for c in "ABCDEFGHIJKLMNOP" if c not in (target,foil))
        pass1=heads.read(foil,target,third)
        pass2=heads.read(target,foil,third)
        miss=heads.read(target,foil,third,edited=True)
        finals={key:miss if key in ("cal_static","learned_k_rescue") else pass2 for key in MODES}
        output.append({"zero_hook_exact":True,"clean_repeat_exact":True,"rescue_k_exact":True,
             "reader_l2":{a:{"q":.1,"k":.2,"v":.3} for a in ("learned","cal_static","seeded")},
             "arms":{key:[pass1[0],value[0]] for key,value in finals.items()},
             "class_maxima":{key:[pass1[1],value[1]] for key,value in finals.items()}})
    return {"status":"closed","worlds":output,"plans":geometry}


def refuse(action,message):
    try:
        action()
    except ValueError as error:
        assert message in str(error),str(error)
    else:
        raise AssertionError("Tampered fixture was accepted")


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--source-dir",type=Path,required=True)
    cli.add_argument("--snapshot",type=Path,required=True)
    cli.add_argument("--work",type=Path,required=True)
    args=cli.parse_args()
    frozen=verified()
    args.work.mkdir(parents=True,exist_ok=False)
    geometry=[w["matched_shell"] for w in json.loads(ROUTEMATCH.read_bytes())["diagnostics"]]
    with lease(HOST_BUDGET) as (check,samples):
        worker=args.work/"worker"
        worker.mkdir()
        for name in ("public.json","tokens.json"):
            copyfile(args.source_dir/name,worker/name)
        public=worker_inputs(worker,args.snapshot,check)
        assert set(public)=={"public","tokens"} and not (worker/"gold.json").exists()
        source,pins=load(args.source_dir,("public","gold","tokens"))
        tok,classes=vocabulary(args.snapshot,check,pins["tokenizer_pins"])
        heads=SyntheticHead(classes,source["tokens"]["bare_label_token_ids"])
        roots={}
        for row in source["public"]:
            if row["arm"]=="focused":
                roots.setdefault(row["root"],{})[row["world"]]=row["id"]
        assert len(roots)==config()["source_roots"]
        work=args.work/"synthetic"
        work.mkdir()
        simulated="synthetic_fixture"
        commit="f"*40
        write_new(work/"admission.json",{"runtime_sha256":frozen,"mode":simulated,"published_commit":commit})
        write_new(work/"preflight.json",{"native_B2_heads_exact":True,"resident_and_streamed_exact":True,
                                        "partition_sha256":partition_sha(classes)})
        baseline=baseline_rows(source,roots,heads,gate_passes=True)
        write_new(work/"baseline.json",baseline)
        host(args,work,qualify_only=True,fixture=True)
        gate=json.loads((work/"qualification.json").read_bytes())["gate"]
        assert gate["passed"]
        first=min(roots)

        def trial(root,pair):
            return sample(source,pair,heads,geometry) if root==first else {"status":"geometry_unqualified"}

        count=record_roots(work,public,roots,gate,trial)
        target=runtime.TARGETS["nf4"]
        terminal(work,frozen,commit,classes,target,target,"fixture","fixture",(len(baseline)//2,count),0,mode=simulated)
        host(args,work,fixture=True)
        result=json.loads((work/"aggregate.json").read_bytes())
        primary=result["contrasts"]["learned_minus_cal_static"]
        assert result["mode"]==simulated and result["closed_roots"]==1 and result["unknown_worlds"]==30
        assert result["private_outcomes_sha256"]==sha(work/"private_outcomes.json")
        assert primary["pass2"]["root_macro_delta"]["full_margin"]==3
        assert primary["pass1"]["root_macro_delta"]["full_margin"]==0
        assert primary["pass2_minus_pass1_full_margin"]==3 and primary["pass2"]["repairs"]==2
        assert result["contrasts"]["learned_minus_clean"]["pass2"]["repairs"]==0
        assert result["contrasts"]["learned_minus_learned_k_rescue"]["pass2"]["repairs"]==2
        refuse(lambda:collect(work,source,tok,classes,frozen,mode="pretrained_cuda"),"does not close")
        copy=args.work/"tampered"
        copytree(work,copy)
        with (copy/"root_000.json").open("ab") as out:
            out.write(b" ")
        refuse(lambda:collect(copy,source,tok,classes,frozen,mode=simulated),"Source hash mismatch: root_000.json")
        episodes=[json.loads((work/f"root_{i:03d}.json").read_bytes()) for i in range(count)]
        bad=deepcopy(episodes)
        bad[0]["plans"][0]["measured"]["seeded"]["radial_l2"]+=1.
        refuse(lambda:score(source,baseline,bad,gate,classes),"Reached MLP displacement")
        # A blocked gate must give all unknowns, even when supplied a full roster.
        refused_gate=gate|{"passed":False,"qualified_focused_roots":[]}
        refused=[episode|{"status":"native_gate_failed"} for episode in episodes]
        empty,_=score(source,baseline,refused,refused_gate,classes)
        assert empty["closed_roots"]==0 and empty["unknown_worlds"]==32
        assert empty["primary_crosses_engineering_floor"] is None and "torch" not in sys.modules
    write_new(args.work/"verification.json",{"status":"passed","runtime_sha256":frozen,
       "host_two_pass_effect_checked":True,"registered_worlds":len(baseline),"worker_without_gold":True,
       "hash_tamper_refused":True,"geometry_tamper_refused":True,"fixture_refused_as_pretrained":True,
       "b2_batches":len(baseline)//2,"synthetic_unknown_worlds":result["unknown_worlds"],
       "refused_unknown_worlds":empty["unknown_worlds"],"synthetic_learned_minus_static":primary,
       "pretrained_forwards":0,"sealed_test_accessed":False,
       "max_sampled_private_bytes":max(s["private_bytes"] for s in samples),
       "minimum_available_RAM_bytes":min(s["available_physical_bytes"] for s in samples)})
    print("TANGENT87 fresh roster, scoped geometry and two-pass host custody passed",flush=True)


if __name__=="__main__":
    main()
