"""OLR-METTA-01 native Hyperon + Lean STOP repair and source curation smoke.

Runs *real* MeTTa 0.2.10 policy queries and a real Lean 4.19 proof server.
Provenance split: Hyperon checks advisory typed/lineage filters, Lean alone
checks tactic outcomes and proof closure. This is a plumbing/verifier test.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from lcgrt import config
from lcgrt.lean_repl import LeanRepl
from .data import canonical, digest, sha256_file
from .live_lean import public_state
from .metta_action_sets import objective, teacher_actions_from_verified_graph
from .metta_obligations import (
    RepairPlan, VerifiedRollout, curate_verified, offer_repairs,
    run_atomic_stop_repair, HYPERON_VERSION,
)
from .olr_delayed import DelayedWorld, compile_scripts

SCHEMA="ppt.olr-metta-native-bridge.v1"


def demonstration(out: Path, *, a: int=5, b: int=7) -> dict:
    """Create-only native test; input numerals do not come from a trained model."""
    if out.exists() or not 0<=a<=15 or not 0<=b<=15:
        raise ValueError("fresh output directory and bounded integers required")
    world=DelayedWorld("metta_smoke_source",0,a,b,(a+b,a+b+1))
    plans=(
        RepairPlan("good",("apply h0","apply h1"),"good_source",1),
        RepairPlan("wrong_order",("apply h1","apply h0"),"wrong_source",1),
        RepairPlan("incomplete",("apply h0",),"short_source",1),
        RepairPlan("conflicted",("apply h0","apply h1"),"conflict_source",1,
                   conflict_free=False),
    )
    with LeanRepl(timeout=60) as lean:
        version=lean.version
        if not str(version).startswith("4.19.0"):
            raise ValueError("pinned Lean runtime required")
        lean.reset()
        root=lean.init(world.statement(),pp=True)
        initial=public_state(lean,root)
        if len(initial["goals"])!=1:
            raise ValueError("unexpected Lean root")
        open_rule=lean.run(root["sid"],"apply r0")
        if open_rule.get("ok") is not True:
            raise ValueError("native rule did not open expected obligations")
        subgoals=public_state(lean,open_rule["state"])
        if len(subgoals["goals"])!=3:
            raise ValueError("expected guard plus two pending obligations")
        guard=lean.run(open_rule["state"]["sid"],"rfl")
        if guard.get("ok") is not True:
            raise ValueError("native equality guard did not close")
        pending=guard["state"]
        public=public_state(lean,pending)
        if [goal["target"] for goal in public["goals"]] != [{"a":"P"},{"a":"Q"}]:
            raise ValueError("native pending goal order drifted")

        stop_untouched_state=digest(public)
        # A standalone STOP on a proof with open goals cannot be certified.
        direct_stop_closed=not public["goals"]
        if direct_stop_closed:
            raise ValueError("STOP baseline prematurely reported closure")

        # A one-call macro must be refused by MeTTa's public budget test.
        short=run_atomic_stop_repair(lean,pending,plans=plans,budget=1,stopped=True)
        if short["committed"] or short["verifier_calls"]!=0:
            raise ValueError("insufficient-budget typed repair was admitted")

        # Permit a maximum of two native tactic checks; certify them before
        # committing the end state to the caller.
        repaired=run_atomic_stop_repair(lean,pending,plans=plans,budget=2,stopped=True)
        if (repaired["status"]!="kernel_closed" or not repaired["committed"]
                or repaired["chosen_plan"]!="good" or repaired["verifier_calls"]!=2):
            raise ValueError("native MeTTa offer did not close exactly two obligations")
        if digest(public_state(lean,pending))!=stop_untouched_state:
            raise ValueError("immutable Lean source state was mutated")

        # The same source branch must independently compile at full theorem
        # statement scope. MeTTa's offer itself earns no certificate credit.
        out.mkdir(parents=True,exist_ok=False)
        script=["apply r0","rfl",*plans[0].actions]
        compiled=compile_scripts(
            [(world,{"kernel_closed":True,"script":script})],
            out/"standalone.lean")
        if compiled["axiom_clean"]!=1:
            raise ValueError("independent source script verification failed")

        # Native-program curation applies exclusively to TRAIN source. The
        # accepted rollout has a clean Lean+standalone receipt. A copied
        # lineage, incomplete attempt, or stale revision fails the query.
        curation=curate_verified([
            VerifiedRollout("goodproof","verified_source","train",True,True,True,
                            True,True,True,True,True,2,2,1,-2),
            VerifiedRollout("unverified","unverified_source","train",False,False,False,
                            True,True,True,True,True,10,10,3,-1),
            VerifiedRollout("stale","stale_source","train",True,True,True,
                            True,True,False,True,True,5,5,2,-3),
            VerifiedRollout("dupone","shared_source","train",True,True,True,
                            True,True,True,True,True,4,4,2,-2),
            VerifiedRollout("duptwo","shared_source","train",True,True,True,
                            True,True,True,True,True,4,4,2,-2),
        ])
        if curation["offer_ids"]!=("goodproof",):
            raise ValueError("native MeTTa source curation selected invalid evidence")

        # Minimal action-set custody example from the exact Lean-certified
        # rule. A different multi-witness bank is needed to test tie benefits.
        verified_root=teacher_actions_from_verified_graph(
            ["apply r0","apply r1"],
            {"apply r0":{"status":"kernel_closed","native_kernel":True,
                         "standalone_verified":True,"axioms":[],
                         "total_tactic_calls":4},
             "apply r1":{"status":"rfl_rejected","native_kernel":False,
                         "standalone_verified":False,"axioms":[],
                         "total_tactic_calls":2}},
            remaining_calls=4,goal_is_kernel_closed=False)
        if verified_root!=("apply r0",):
            raise ValueError("source-only action set leaked uncertified branch")

        return_report={
            "schema":SCHEMA,"evidence":"real_hyperon_and_real_lean_engineering_smoke",
            "scientific_eligible":False,
            "lean_version":version,"lean_sha256":sha256_file(config.LEAN_EXE),
            "lean_repl_sha256":sha256_file(config.REPL_EXE),
            "metta_runtime":f"hyperon=={HYPERON_VERSION}",
            "parent_public_sha256":digest(initial),
            "stopped_public_sha256":stop_untouched_state,
            "standalone":compiled,
            "source_train_curator":curation,
            "repairs":{
                "stop_only_verified_closed":False,
                "short_budget_committed":short["committed"],
                "short_budget_checks":short["verifier_calls"],
                "native_macro_committed":repaired["committed"],
                "native_macro_checks":repaired["verifier_calls"],
                "native_macro_plan":repaired["chosen_plan"],
                "native_macro_actions":[e["tactic"] for e in repaired["events"]],
                "native_macro_metta_sha256":repaired["me_tta"]["program_sha256"],
                "native_tactic_receipts":repaired["events"],
            },
            "action_set":{
                "source_optimal_actions":list(verified_root),
                "training_schema":"DG_METTA12_optimal_action_set_v1",
                "tie_training_evidence":"not_tested_in_single_success_world",
            },
        }
        (out/"manifest.json").write_text(canonical(return_report)+"\n",encoding="utf-8")
        return_report["artifact_sha256"]={
            name:sha256_file(out/name) for name in ("manifest.json","standalone.lean")
        }
        return return_report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--a",type=int,default=5)
    p.add_argument("--b",type=int,default=7)
    args=p.parse_args(argv)
    report=demonstration(args.out,a=args.a,b=args.b)
    print(canonical({
        key:report[key] for key in ("schema","scientific_eligible","standalone","repairs","action_set")
    }))


if __name__=="__main__":
    main()
