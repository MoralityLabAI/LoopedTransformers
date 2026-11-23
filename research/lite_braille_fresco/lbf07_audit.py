"""LBF-07 public-only candidate-rule association and rule-order audit.

Uses the train-only native teacher source. Strictly checks output byte counts,
Unicode histograms, included hypotheses, and rule binding changes without
accessing truth labels or future verifier outcomes for analysis.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys

LCGRT = Path(__file__).resolve().parents[1]/"lean_cgrt_hrm400"
if str(LCGRT) not in sys.path:
    sys.path.insert(0,str(LCGRT))

from ppt_gym.data import canonical, sha256_file
from ppt_gym.native_train import _load_teacher
from ppt_gym.typed_state import encode_public_packet

VIEWS=("hybrid_rule_intact","hybrid_rule_rebind","hybrid_rule_order")
RULE = re.compile(r"([hr][0-9]{1,4}):([^;]*);")
SCHEMA="lbf07.public_rule_association_audit.v1"


def rules(text: str) -> list[tuple[str,str]]:
    # PP is an inert x-filled public suffix in all three controls.
    return RULE.findall(text.split("P:",1)[0])


def analyze_frame(packet: str, candidates: list[str]) -> dict:
    entries={v:encode_public_packet(packet,candidates,view=v,max_bytes=384)
             for v in VIEWS}
    reference=entries[VIEWS[0]]
    for view in VIEWS:
        text=entries[view]
        if len(text.encode("utf-8")) != len(reference.encode("utf-8")):
            raise ValueError("rule intervention changed UTF-8 input byte count")
        if len(text.encode("utf-8")) > 384:
            raise ValueError("rule intervention exceeded 384 ByteEncoder bytes")
        if Counter(text) != Counter(reference):
            raise ValueError("rule intervention changed character histogram")
    legacy=encode_public_packet(packet,candidates,view="hybrid_nopp",max_bytes=384)
    if legacy != reference:
        raise ValueError("LBF-07 intact differs from the existing blank-PP hybrid")
    original=rules(reference)
    rebound=rules(entries[VIEWS[1]])
    ordered=rules(entries[VIEWS[2]])
    if len(original)!=len(rebound) or len(original)!=len(ordered):
        raise ValueError("rule count changed after representation intervention")
    if [n for n,_ in original] != [n for n,_ in rebound]:
        raise ValueError("rule rebinding changed hypothesis-name positions")
    if sorted(payload for _,payload in original) != sorted(payload for _,payload in rebound):
        raise ValueError("rule rebinding changed typed clause-payload multiset")
    if sorted(original) != sorted(ordered):
        raise ValueError("rule order-placebo broke a name-to-payload association")
    menu_names=[c[6:] for c in candidates if c.startswith("apply ")]
    known=dict(original)
    after=dict(rebound)
    candidate_change=sum(
        known.get(name)!=after.get(name) for name in menu_names if name in known)
    return {
        "bytes":len(reference.encode("utf-8")),
        "rule_count":len(original),
        "candidate_count":len(menu_names),
        "candidate_rules_visible":sum(name in known for name in menu_names),
        "changed_candidate_bindings":candidate_change,
        "candidate_binding_changed":candidate_change>0,
        "any_rule_binding_changed":original!=rebound,
        "rule_order_changed":original!=ordered,
        "intact_sha256":hashlib.sha256(reference.encode("utf-8")).hexdigest(),
        "rebound_sha256":hashlib.sha256(entries[VIEWS[1]].encode("utf-8")).hexdigest(),
        "ordered_sha256":hashlib.sha256(entries[VIEWS[2]].encode("utf-8")).hexdigest(),
    }


def audit(teacher:Path,out:Path)->dict:
    if out.exists():
        raise ValueError("audit output must be new")
    rows,manifest=_load_teacher(teacher)
    if (manifest.get("split")!="train" or
        manifest.get("backend",{}).get("actual_lean_process") is not True or
        manifest.get("source_compiler_passed") is not True or
        manifest.get("query_roots_exposed_to_fit")!=0 or
        manifest.get("independent_lineages")!=48):
        raise ValueError("LBF-07 requires the 48-root native train-only source")
    per_root=defaultdict(list)
    records=[]
    for row in rows:
        # The teacher importer verifies native source and labels internally;
        # the representation audit consumes strictly public fields here.
        result=analyze_frame(row["goal"],row["candidates"])
        record={"lineage":row["lineage"],"root":row["root"],
                "public_packet_sha256":hashlib.sha256(row["goal"].encode()).hexdigest(),
                "public_menu_sha256":hashlib.sha256(canonical(row["candidates"]).encode()).hexdigest(),
                **result}
        records.append(record)
        per_root[row["lineage"]].append(result)
    if len(per_root)!=48 or len(rows)!=manifest["train_examples"]:
        raise ValueError("public root census/source manifest changed")
    root_fraction=lambda key: sum(
        sum(int(record[key]) for record in records_in_root)/len(records_in_root)
        for records_in_root in per_root.values())/len(per_root)
    exposure=root_fraction("candidate_binding_changed")
    output={
        "schema":SCHEMA,
        "source_split":"train","source_native_certified":True,
        "teacher_manifest_sha256":sha256_file(teacher/"manifest.json"),
        "train_frames":len(rows),"unique_train_roots":len(per_root),
        "intervention_views":list(VIEWS),
        "intact_matches_existing_blank_pp":True,
        "exact_visible_bytes_and_unicode_histogram":True,
        "byte_limit":384,
        "mean_visible_input_bytes":sum(x["bytes"] for x in records)/len(records),
        "frame_candidate_binding_changes":sum(x["candidate_binding_changed"] for x in records),
        "frame_rule_order_changes":sum(x["rule_order_changed"] for x in records),
        "fraction_candidate_bindings_changed":sum(x["candidate_binding_changed"] for x in records)/len(records),
        "root_weighted_candidate_binding_change_fraction":exposure,
        "root_weighted_order_change_fraction":root_fraction("rule_order_changed"),
        "minimum_informative_source_exposure":0.5,
        "intervention_exposure_sufficient":exposure>=0.5,
        "labels_consumed_to_choose_representation":False,
        "locked_test_opened":False,"scientific_eligible":False,
        "interpretation":"Model-facing candidate-binding perturbation exposure; source native proof status remains external",
    }
    out.mkdir(parents=True)
    destination=out/"public_rule_audit.jsonl"
    destination.write_text("".join(canonical(row)+"\n" for row in records),encoding="utf-8")
    output["audit_rows_sha256"]=sha256_file(destination)
    (out/"manifest.json").write_text(canonical(output)+"\n",encoding="utf-8")
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--teacher",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    args=p.parse_args()
    print(canonical(audit(args.teacher,args.out)))


if __name__=="__main__":
    main()
