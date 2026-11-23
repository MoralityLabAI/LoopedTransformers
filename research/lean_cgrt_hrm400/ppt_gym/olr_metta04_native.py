"""OLR-METTA-04B: real Lean staged native protocol and Hyperon repair smoke.

This preflight uses a deterministic reference controller. No neural weights
are loaded or scored; its purpose is kernel/focus/restore/MeTTa custody.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from lcgrt import config
from lcgrt.lean_repl import LeanRepl
from .data import canonical,digest,sha256_file
from .olr_dependent import compile_certified
from .olr_metta04_world import collect
from .olr_metta04_episode import run_episode,replay_episode


def _reference(winner:int,macro:bool,wrong:bool):
    initial=1-winner if wrong else winner
    def policy(stage,remaining,branch):
        if stage=="root":return "root_r"+str(initial),0
        if stage=="retry":return "root_r"+str(winner),0
        if stage=="open":return "probe_guard",0
        if stage=="failed":return ("restore" if macro else "stop"),0
        if stage=="guard_ok":return ("metta_repair" if macro else "manual"),0
        if stage=="closed":return "stop",0
        raise ValueError("unknown stage in source reference")
    return policy


def preflight(out:Path,*,seed:int=20261130,pairs:int=3):
    if out.exists() or pairs!=3:
        raise ValueError("registered create-only 3-pair gate")
    source=collect("train",seed=seed,count=pairs,low=17,high=30)
    traces=[]
    successes=[]
    calls=0
    with LeanRepl(timeout=60) as lean:
        if not str(lean.version).startswith("4.19.0"):
            raise ValueError("native Lean 4.19 required")
        version=lean.version
        for w in source:
            packet=w.public()
            for wrong in (False,True):
                ep=run_episode(lean,w.proof,claims=packet["model_features"],
                    macro_allowed=w.macro_allowed,
                    policy=_reference(w.proof.reference_branch(),w.macro_allowed,wrong))
                if not replay_episode(lean,w.proof,ep,
                                      macro_allowed=w.macro_allowed):
                    raise ValueError("independent native replay failed")
                expected=(not wrong or w.macro_allowed)
                if ep["kernel_closed"]!=expected:
                    raise ValueError("dependent pending obligation outcome changed")
                if ep["kernel_closed"]:
                    script=[f"apply r{ep['verified_branch']}","rfl"]
                    script.extend(("apply h0","exact h1 _ h0")
                                  if ep["tail"]=="fast" else
                                  ("apply h0","apply h1","apply h0"))
                    successes.append((w.proof,{"kernel_closed":True,"root_id":w.proof.root_id,
                                               "script":script}))
                calls+=ep["tactic_calls"]
                traces.append({
                    "root":w.proof.root_id,"pair":w.proof.pair_id,
                    "world":w.proof.world,"wrong_start":wrong,
                    "macro_allowed":w.macro_allowed,"episode":ep,
                    "public_evidence_sha256":digest(packet["model_features"]),
                    "metta_program_sha256":packet["metta_program_sha256"]})
    out.mkdir(parents=True)
    cert=compile_certified(successes,out/"standalone.lean")
    (out/"episodes.jsonl").write_text(
        "".join(canonical(row)+"\n" for row in traces),encoding="utf-8")
    expected_closed=10
    if cert["axiom_free"]!=6 or sum(t["episode"]["kernel_closed"] for t in traces)!=expected_closed:
        raise ValueError("native dependent preflight certificate counts changed")
    report={
        "schema":"ppt.olr-metta04-staged-native.v1",
        "seed":seed,"pairs":pairs,"worlds":len(source),"episodes":len(traces),
        "native_replays":len(traces),"native_proof_episodes":expected_closed,
        "standalone":cert,"total_tactic_calls":calls,
        "lean_version":version,"lean_exe_sha256":sha256_file(config.LEAN_EXE),
        "tool_evidence":"native_Lean_4_19_and_pinned_Hyperon",
        "model_training_eligible":False,
        "architecture_promotion_eligible":False,
        "artifacts":{n:sha256_file(out/n) for n in ("episodes.jsonl","standalone.lean")},
    }
    (out/"manifest.json").write_text(canonical(report)+"\n",encoding="utf-8")
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--seed",type=int,default=20261130)
    p.add_argument("--pairs",type=int,default=3)
    args=p.parse_args(argv)
    print(canonical(preflight(args.out,seed=args.seed,pairs=args.pairs)))


if __name__=="__main__":
    main()
