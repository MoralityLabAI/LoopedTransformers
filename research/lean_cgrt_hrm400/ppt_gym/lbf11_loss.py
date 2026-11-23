"""LBF-11 matched outcome/ranking loss intervention and train-only diagnostics."""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from .cli import checkpoint_load
from .core import Config, PlasticProofTransformer, supervised_loss
from .data import canonical, sha256_file
from .lbf08_repair import _load_visited, frame_schedule, BEHAVIOR_VIEW
from .native_train import _load_teacher, _model_fingerprint
from .lbf_hybrid import SCHEMAS
from .typed_state import encode_public_packet

SCHEMA="lbf11.matched_loss_fit.v1"
OBJECTIVE_SCHEMA="lbf11.outcome_ranking_factorial.v1"
ARMS=("source_only","b1_r1","bs_r1","b1_rs","bs_rs","bs_balanced")
TERMS={
    "b1_r1":("b1","r1"), "bs_r1":("bs","r1"),
    "b1_rs":("b1","rs"), "bs_rs":("bs","rs"),
    "bs_balanced":("bs","rb"),
}
ARCHIVE_HASHES={
    "107-visited_one":"022c030e7de944cff900eae4248bcae2da46363f0a3827f0199d6757bc42724e",
    "107-visited_set":"2691c4945a6b6860b06806bbc8ab2e396adf617fa962a7df94d1d737a185ce73",
    "109-visited_one":"4b10d66007f84ab97aaec69070d38f2828faad8724bd0b9dc55afc992ab6ebe4",
    "109-visited_set":"924276d1d6733f879a05cdd5c055bda87c478983c084dc7ff4f8be0e6382c161",
}


def _write(path, obj):
    path.write_text(canonical(obj)+"\n",encoding="utf-8")


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha_rows(rows):
    return hashlib.sha256("".join(canonical(x)+"\n" for x in rows).encode()).hexdigest()


def _validate_labels(full, single):
    if (not full or len(full)!=len(single)
            or any(type(x) is not int or x not in (0,1) for x in [*full,*single])
            or sum(full)<1 or sum(single)!=1
            or any(a>b for a,b in zip(single,full))
            or single.index(1)!=full.index(1)):
        raise ValueError("require native binary full shortest set and its menu-first member")


def pass_terms(output, full, single):
    _validate_labels(full,single)
    z,o=output["logits"],output["outcome"]
    if z.ndim!=1 or o.shape!=z.shape or len(z)!=len(full):
        raise ValueError("objective logits and known native labels mismatch")
    ys=torch.tensor(full,device=z.device,dtype=torch.float32)
    y1=torch.tensor(single,device=z.device,dtype=torch.float32)
    s=ys==1
    # Same known-label indexing and same operations as legacy endpoints.
    # Every native label here is known. No synthetic or censored target is used.
    known=ys>=0
    base=torch.logsumexp(z[known],0)
    positive=z[known][s[known]]
    return {
        "b1":F.binary_cross_entropy_with_logits(o[known],y1[known]),
        "bs":F.binary_cross_entropy_with_logits(o[known],ys[known]),
        "r1":base-torch.logsumexp(z[known][y1[known]==1],0),
        "rs":base-torch.logsumexp(positive,0),
        "rb":base-positive.mean()-math.log(int(s.sum())),
    }


def objective(outputs, full, single, arm):
    if arm not in TERMS or not outputs:
        raise ValueError("unknown registered visited loss")
    a,b=TERMS[arm]
    loss=torch.zeros((),device=outputs[0]["logits"].device)
    for output in outputs:
        t=pass_terms(output,full,single)
        loss=loss+t[a]+t[b]
    return loss/len(outputs)


def verified_source(suite,teacher,collection,behavior,replay,exposure):
    gate=_read(exposure/"manifest.json")
    source_rows,teacher_meta=_load_teacher(teacher)
    frames,collection_meta=_load_visited(collection,teacher,suite,behavior,replay)
    sm=_read(suite/"manifest.json")
    if not (
        sm["seed"]==20261203 and sm.get("structural_alt") is True
        and sm.get("source_family")=="lbf10_independent_equal_cost_diamonds.v1"
        and gate["schema"]=="lbf11.native_loss_source_exposure.v1"
        and gate["gate"]=="MULTI_OPTIMAL_SOURCE_SUFFICIENT"
        and gate["source_native_verified"] is True
        and gate["source_independent_replay_verified"] is True
        and gate["suite_manifest_sha256"]==sha256_file(suite/"manifest.json")
        and gate["native_teacher_manifest_sha256"]==sha256_file(teacher/"manifest.json")
        and gate["collection_manifest_sha256"]==sha256_file(collection/"manifest.json")
        and gate["independent_replay_sha256"]==sha256_file(replay)
        and gate["behavior_checkpoint_sha256"]==sha256_file(behavior)
        and gate["model_visited_frames"]==len(frames)
        and gate["model_visited_multi_optimal_frames"]==sum(sum(x["labels"])>1 for x in frames)
        and gate["model_visited_multi_fraction"]>=.25
        and gate["model_visited_roots_with_multi_optimal_witnesses"]>=12
        and not gate["ordinary_test_opened"] and not gate["ood_family_opened"]
    ):
        raise ValueError("LBF-11 fresh source gate failed")
    for f in frames:
        _validate_labels(f["labels"],f["single_labels"])
    return source_rows,teacher_meta,frames,collection_meta,gate


def fit(suite,teacher,collection,behavior,replay,exposure,out,*,seed,arm,max_seconds=600):
    if seed not in (107,109) or arm not in ARMS or out.exists() or not 0<max_seconds<=1200:
        raise ValueError("registered LBF-11 fit requires new output, fixed seed/arm/budget")
    ref,source,visits,repair,gate=verified_source(suite,teacher,collection,behavior,replay,exposure)
    torch.use_deterministic_algorithms(True); torch.set_num_threads(1); torch.manual_seed(seed)
    cfg=Config(plan="b",width=32,heads=4,max_bytes=384).validate()
    model=PlasticProofTransformer(cfg).cpu().float()
    if model.parameter_count()!=57475:
        raise ValueError("parameter envelope changed")
    initial=_model_fingerprint(model)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
    src,vis=defaultdict(list),defaultdict(list)
    for frame in ref: src[frame["lineage"]].append(frame)
    for frame in visits: vis[frame["lineage"]].append(frame)
    roster=frame_schedule(ref,visits,seed,720)
    schedule_sha=_sha_rows(roster)
    planned=gate["seed_draws"][str(seed)]
    if schedule_sha!=planned["draw_schedule_sha256"]:
        raise ValueError("frozen optimizer draw roster changed")
    out.mkdir(parents=True)
    begun=time.perf_counter()
    draws,packets=[],[]
    for row in roster:
        if time.perf_counter()-begun>max_seconds:
            raise TimeoutError("fixed 720-step loss fit exceeded CPU limit")
        lineage=row["lineage"]
        visited=bool(arm!="source_only" and row["use_model_visited"])
        example=vis[lineage][row["visit_index"]] if visited else src[lineage][row["source_index"]]
        full=example["labels"]
        single=example.get("single_labels")
        goal=encode_public_packet(example["goal"],example["candidates"],
                                  view=BEHAVIOR_VIEW,max_bytes=cfg.max_bytes)
        outputs=model(goal,example["candidates"],depth=2,
                      checks_remaining=example["checks_remaining"],return_all=True,middle_order="ab")
        if visited:
            loss=objective(outputs,full,single,arm)
            outcome_term,rank_term=TERMS[arm]
        else:
            loss=supervised_loss(outputs,tuple(full))
            outcome_term,rank_term="bs","rs"
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("nonfinite registered objective")
        optimizer.zero_grad(set_to_none=True); loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        if not bool(torch.isfinite(norm)):
            raise FloatingPointError("nonfinite registered gradient")
        optimizer.step()
        packet={**row,"chosen_source":"model_visited" if visited else "native_reference",
                "packet_sha256":hashlib.sha256(canonical([example["goal"],example["candidates"],
                                                          example["checks_remaining"]]).encode()).hexdigest(),
                "native_positive_count":sum(full),"candidate_count":len(full)}
        packets.append(packet)
        draws.append({**packet,"outcome_target":outcome_term,"rank_target":rank_term,
                      "loss":float(loss.detach()),"preclip_gradient_norm":float(norm)})
    nvis=sum(d["chosen_source"]=="model_visited" for d in draws)
    nmulti=sum(d["chosen_source"]=="model_visited" and d["native_positive_count"]>1 for d in draws)
    if (arm=="source_only" and (nvis or nmulti)) or (arm!="source_only" and
         (nvis!=planned["model_visited_updates"] or nmulti!=planned["multi_positive_visited_updates"] or nmulti<50)):
        raise ValueError("actual native treatment exposure changed")
    model.eval()
    target=out/"checkpoint.pt"
    teacher_sha=sha256_file(teacher/"manifest.json")
    repair_sha=sha256_file(collection/"manifest.json")
    replay_sha=sha256_file(replay)
    exposure_sha=sha256_file(exposure/"manifest.json")
    impl_sha=sha256_file(Path(__file__))
    packet_sha=_sha_rows(packets)
    torch.save({
        "schema":"ppt_gym.checkpoint.v1","config":asdict(cfg),"weights":model.state_dict(),
        "source_manifest_sha256":source["source"]["suite_manifest_sha256"],
        "source_evidence_class":source["evidence_class"],"native_teacher_sha256":teacher_sha,
        "lbf08_repair_source_manifest_sha256":repair_sha,"lbf08_independent_replay_sha256":replay_sha,
        "lbf11_loss_arm":arm,"lbf11_objective_schema":OBJECTIVE_SCHEMA,"lbf11_impl_sha256":impl_sha,
        "lbf11_exposure_sha256":exposure_sha,"lbf11_packet_roster_sha256":packet_sha,
        "train_steps":720,"seed":seed,"train_order":"ab",
        "input_view":BEHAVIOR_VIEW,"input_view_schema":SCHEMAS[BEHAVIOR_VIEW],
    },target)
    (out/"optimizer_steps.jsonl").write_text("".join(canonical(d)+"\n" for d in draws),encoding="utf-8")
    result={
        "schema":SCHEMA,"arm":arm,"seed":seed,"objective_schema":OBJECTIVE_SCHEMA,
        "objective_impl_sha256":impl_sha,"parameter_count":model.parameter_count(),
        "updates":len(draws),"source_roots":len(src),"train_order":"ab",
        "input_view":BEHAVIOR_VIEW,"depth":2,"learning_rate":.001,"threads":1,
        "source_suite_manifest_sha256":source["source"]["suite_manifest_sha256"],
        "native_teacher_manifest_sha256":teacher_sha,"repair_source_manifest_sha256":repair_sha,
        "independent_replay_sha256":replay_sha,"exposure_manifest_sha256":exposure_sha,
        "draw_schedule_sha256":schedule_sha,"packet_roster_sha256":packet_sha,
        "initial_weights_sha256":initial,"final_weights_sha256":_model_fingerprint(model),
        "checkpoint_sha256":sha256_file(target),"optimizer_steps_sha256":sha256_file(out/"optimizer_steps.jsonl"),
        "model_visited_updates":nvis,"native_multi_positive_visited_updates":nmulti,
        "outcome_multi_positive_visited_updates":sum(d["chosen_source"]=="model_visited"
            and d["native_positive_count"]>1 and d["outcome_target"]=="bs" for d in draws),
        "rank_multi_positive_visited_updates":sum(d["chosen_source"]=="model_visited"
            and d["native_positive_count"]>1 and d["rank_target"] in ("rs","rb") for d in draws),
        "mean_preclip_gradient_norm":sum(d["preclip_gradient_norm"] for d in draws)/720,
        "clipped_updates":sum(d["preclip_gradient_norm"]>1 for d in draws),
        "last_32_loss":sum(d["loss"] for d in draws[-32:])/32,
        "calibration_updates":0,"locked_test_opened":False,"scientific_eligible":False,
    }
    _write(out/"manifest.json",result)
    return result


def _cos(a,b):
    den=float(a.norm()*b.norm())
    return float(torch.dot(a,b)/den) if den>0 else None


def _audit_model(model,frames):
    model.eval()
    before=_model_fingerprint(model)
    shared=[p for name,p in model.named_parameters()
            if not name.startswith(("policy.","outcome.","continue_head."))]
    records=[]
    for frame in frames:
        full,single=frame["labels"],frame["single_labels"]
        if sum(full)<2:
            continue
        _validate_labels(full,single)
        text=encode_public_packet(frame["goal"],frame["candidates"],view=BEHAVIOR_VIEW,max_bytes=384)
        outputs=model(text,frame["candidates"],depth=2,checks_remaining=frame["checks_remaining"],
                      return_all=True,middle_order="ab")
        passes=[pass_terms(o,full,single) for o in outputs]
        terms={k:sum(t[k] for t in passes)/len(passes) for k in passes[0]}
        gradients={}
        scores={}
        for name,loss in terms.items():
            heads=[o["outcome" if name.startswith("b") else "logits"] for o in outputs]
            g=torch.autograd.grad(loss,[*shared,*heads],retain_graph=True,allow_unused=True)
            flat=torch.cat([v.detach().reshape(-1) if v is not None else torch.zeros_like(p).reshape(-1)
                            for v,p in zip(g[:len(shared)],shared)])
            head=torch.cat([v.detach().reshape(-1) if v is not None else torch.zeros_like(p).reshape(-1)
                            for v,p in zip(g[len(shared):],heads)])
            gradients[name]=flat
            scores[name+"_loss"]=float(loss.detach())
            scores[name+"_shared_grad_norm"]=float(flat.norm())
            scores[name+"_head_grad_norm"]=float(head.norm())
        z=outputs[-1]["logits"].detach()
        mask=torch.tensor(full,dtype=torch.bool)
        cond=z[mask].softmax(0)
        scores.update(
            all_positive_menu=bool(mask.all()),
            candidate_count=len(full),native_positive_count=sum(full),
            positive_mass=float(z.softmax(0)[mask].sum()),
            positive_entropy_normalized=float(-(cond*cond.clamp_min(1e-30).log()).sum()/math.log(sum(full))),
            best_positive_minus_best_negative=float(z[mask].max()-z[~mask].max()) if not bool(mask.all()) else None,
            weakest_positive_minus_best_negative=float(z[mask].min()-z[~mask].max()) if not bool(mask.all()) else None,
            shared_cos_b1_bs=_cos(gradients["b1"],gradients["bs"]),
            shared_cos_r1_rs=_cos(gradients["r1"],gradients["rs"]),
            shared_cos_rs_rb=_cos(gradients["rs"],gradients["rb"]),
            shared_cos_b1_r1=_cos(gradients["b1"],gradients["r1"]),
            shared_cos_bs_rs=_cos(gradients["bs"],gradients["rs"]),
        )
        if any(v is not None and isinstance(v,float) and not math.isfinite(v) for v in scores.values()):
            raise FloatingPointError("nonfinite frozen diagnostic")
        records.append({"root":frame["root"],"lineage":frame["lineage"],
                        "attempt":frame["attempt"],"packet_sha256":frame["source_public_packet_sha256"],
                        **scores})
    if not records or before!=_model_fingerprint(model):
        raise ValueError("empty diagnostic or frozen weights changed")
    by_root=defaultdict(list)
    for r in records: by_root[r["lineage"]].append(r)
    metric_names=set(records[0])-{"root","lineage","attempt","packet_sha256"}
    summary={}
    for k in sorted(metric_names):
        values=[]
        for group in by_root.values():
            valid=[float(r[k]) for r in group if r[k] is not None]
            if valid: values.append(sum(valid)/len(valid))
        summary[k]={"root_weighted_mean":sum(values)/len(values) if values else None,
                    "defined_roots":len(values),"defined_frames":sum(r[k] is not None for r in records)}
    return records,{"frames":len(records),"lineages":len(by_root),"frozen_weights_sha256":before,
                    "all_positive_frames":sum(r["all_positive_menu"] for r in records),
                    "root_weighted_metrics":summary}


def audit(suite,teacher,collection,behavior,replay,out,*,archive_cells=None):
    if out.exists():
        raise ValueError("diagnostic destination must be fresh")
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    frames,meta=_load_visited(collection,teacher,suite,behavior,replay)
    sm=_read(suite/"manifest.json")
    expected_seed=20261201 if archive_cells is not None else 20261203
    if sm["seed"]!=expected_seed or sm.get("structural_alt") is not True:
        raise ValueError("diagnostic source identity differs")
    teacher_sha=sha256_file(teacher/"manifest.json")
    repair_sha=sha256_file(collection/"manifest.json")
    replay_sha=sha256_file(replay)
    if archive_cells is not None:
        if (sha256_file(suite/"manifest.json")!="9dc7da38d2030ad4cd5f90ef99a729be63a1758a67ea0b4816423150b13f21d6"
            or teacher_sha!="ef2719f7ebf246d86838fe80c841ba4ebd1ba1ab9368befe69fc70822f2fa52c"
            or repair_sha!="2e0333a0081b1873aeb3ec265a5d74924e199b81717b1918cb7ac36c7b2669cd"
            or replay_sha!="f00e979b7a125cbc9293a57f7067b9793af47b06b89eaf13177e0cf44478bae6"
            or sha256_file(behavior)!="55163e698ba4788d1f6396213f8d6fc16f8dd5334b600fa2999e07b91211dab2"
            or sum(sum(f["labels"])>1 for f in frames)!=55):
            raise ValueError("archived LBF-10 source/replay bytes differ from original verified run")
        paths={name:archive_cells/name/"fit"/"checkpoint.pt" for name in ARCHIVE_HASHES}
    else:
        paths={"fresh_behavior":behavior}
    out.mkdir(parents=True)
    all_summaries={}
    for name,path in paths.items():
        sha=sha256_file(path)
        model,ck=checkpoint_load(path)
        if (ck.get("native_teacher_sha256")!=teacher_sha
                or ck.get("source_manifest_sha256")!=sha256_file(suite/"manifest.json")
                or ck.get("input_view")!=BEHAVIOR_VIEW or model.parameter_count()!=57475):
            raise ValueError("diagnostic checkpoint/source mismatch")
        if archive_cells is not None and (sha!=ARCHIVE_HASHES[name]
                or ck.get("lbf08_repair_source_manifest_sha256")!=repair_sha
                or ck.get("lbf08_independent_replay_sha256")!=replay_sha):
            raise ValueError("archived frozen checkpoint custody mismatch")
        records,summary=_audit_model(model,frames)
        f=out/(name+".jsonl")
        f.write_text("".join(canonical(row)+"\n" for row in records),encoding="utf-8")
        all_summaries[name]={**summary,"checkpoint_sha256":sha,"frame_metrics_sha256":sha256_file(f)}
    result={"schema":"lbf11.frozen_train_only_diagnostic.v1",
            "source_kind":"archived_lbf10" if archive_cells is not None else "fresh_behavior",
            "teacher_manifest_sha256":teacher_sha,"collection_manifest_sha256":repair_sha,
            "replay_sha256":replay_sha,"models":all_summaries,
            "train_only":True,"optimizer_updates":0,"lean_calls":0,"calibration_outcomes_read":0,
            "locked_test_opened":False,"scientific_eligible":False}
    _write(out/"manifest.json",result)
    print("LBF11_DIAGNOSTIC",canonical(result),flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("suite","teacher","collection","behavior","replay","out"):
        p.add_argument("--"+k,type=Path,required=True)
    p.add_argument("--archive-cells",type=Path)
    a=p.parse_args()
    audit(a.suite,a.teacher,a.collection,a.behavior,a.replay,a.out,archive_cells=a.archive_cells)


if __name__=="__main__":
    main()
