"""NEC-03J: read-only GELU Jacobians and recurrent-memory transport.

Use the exact frozen NEC-03H-v2 900-step models and NEC-03I-B source-ranked
versus four hash-active controls. At each memory-token recurrent visit,
differentiate a CLEAN-CHOSEN-class logit margin against both the 64-D GELU
MLP input and the 32-D post-mlp_down output. Compare the source-control
donor displacement, predicted margin change and actual nonlinear change
under their unchanged hubs-only lesion. Propagate memory-state drift across
all 12 visits, preserving per-input records and every sign disagreement.

No weight training or confirmation/reserved context scoring occurs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from mechinterp.neuron_communities.nec03ib_active_shams import (
    VARIANTS, pair_donors, load_frozen_source, check_logits,
    verify_panel_lock,
)
from mechinterp.neuron_communities.nec03g_worlds import target_bits
from research.pef07.pef07 import L, as_tensor, model_digest
from mechinterp.neuron_communities.source_sae import WIDTH

SCHEMA="NEC-03J/GELU-Jacobian-and-tied-recurrent-transport/v1"
PARENT_RUN=44878411694
CONTROL_RUN=44885003105
FD_EPS=0.003
FD_MAX_ABSOLUTE_ERROR=0.02
JACOBIAN_IDENTITY_MAX_ERROR=0.001
TOTAL_CONTEXTS=48


def verify_direction_lock():
    """Fail closed if any source code, cohort or physical measurement changed."""
    root=Path(__file__).resolve().parent
    lockpath=root/"NEC03J_LOCK.json"
    obj=json.loads(lockpath.read_text())
    if (obj.get("schema")!="NEC-03J/locked-direction-transport-v1"
        or obj.get("original_native_run")!=PARENT_RUN
        or obj.get("original_active_control_run")!=CONTROL_RUN
        or obj.get("world_indices")!=list(range(8,16))
        or obj.get("gradient_reference_modes")!=["clean","hubs"]
        or obj.get("gradient_target")!="clean-chosen-binary-margin"
        or obj.get("finite_difference_epsilon")!=FD_EPS
        or obj.get("gelu_to_output_tangent_identity_max_error")!=JACOBIAN_IDENTITY_MAX_ERROR
        or obj.get("confirmation_worlds_scored")!=0
        or obj.get("reserved_context_groups_scored")!=0):
        raise AssertionError("NEC03J source and physical-direction protocol not locked")
    for name,expected in obj["git_blob_sha1"].items():
        content=(root/name).read_bytes()
        digest=hashlib.sha1(
            b"blob "+str(len(content)).encode()+b"\x00"+content).hexdigest()
        if digest!=expected:
            raise AssertionError("NEC03J source file differs from preregistered bytes")
    obj["lock_sha256"]=hash_file(lockpath)
    return obj


def encoded(obj):
    return (json.dumps(obj,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()


def hash_file(path: Path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""):
            h.update(chunk)
    return h.hexdigest()


def oriented_margin(logits, orientation):
    if logits.shape!=(len(orientation),2):
        raise ValueError("incorrect binary logit/margin orientation")
    primary=logits.gather(1,orientation[:,None]).squeeze(1)
    other=logits.gather(1,(1-orientation)[:,None]).squeeze(1)
    return primary-other


def traced_forward(model,bits,*,mode,orientation=None,donors=None,gates=None,
                   derivative=False,perturb=None):
    """Trace both native GELU and 32-D MLP output with memory per visit.

    mode clean keeps every channel. Others retain live hub channels and
    add the *clean-input donor* for the current recurrent visit.
    The gradient is of the same originally clean-chosen logit margin,
    with the fixed original choice in every modified trajectory.
    """
    if mode not in ("clean","hubs","native","sham"):
        raise ValueError("invalid recurrent donor arm")
    if bits.ndim!=2 or bits.shape[1]!=12:
        raise ValueError("expected complete PEF 12-bit input")
    if mode!="clean" and (donors is None or gates is None):
        raise ValueError("modified arm needs frozen donor masks")
    inputs,outputs,states=[],[],[]
    count=0
    def prehook(module,args):
        nonlocal count
        visit=count
        count+=1
        x=args[0]
        if x.ndim!=3 or x.shape[1:]!=(2,64):
            raise AssertionError("native GELU MLP memory token has wrong coordinate")
        modified=x
        if mode!="clean" and visit in donors:
            modified=x.clone()
            added=donors[visit][mode]
            if added.shape!=(len(bits),64):
                raise AssertionError("wrong physically attenuated donor dimensions")
            modified[:,1,:]=x[:,1,:]*gates[visit][None,:]+added
        if perturb is not None and visit==perturb[0]:
            _,channel,size=perturb
            if channel not in range(64):
                raise ValueError("finite difference channel outside 64D native MLP")
            modified=modified.clone()
            modified[:,1,channel]=modified[:,1,channel]+size
        inputs.append(modified)
        return (modified,) if modified is not x else None
    def downhook(module,args,output):
        outputs.append(output)
    def blockhook(module,args,output):
        if output.shape!=(len(bits),2,32):
            raise AssertionError("hidden memory trajectory outside native 32D")
        states.append(output[:,1,:])
    handles=[
        model.block.mlp_down.register_forward_pre_hook(prehook),
        model.block.mlp_down.register_forward_hook(downhook),
        model.block.register_forward_hook(blockhook)
    ]
    try:
        logits=model.logits(bits)
        if len(inputs)!=L or len(outputs)!=L or len(states)!=L or count!=L:
            raise AssertionError("recurrent tied weight hooks saw incorrect visit count")
        if orientation is None:
            orientation=logits.argmax(1)
        margin=oriented_margin(logits,orientation)
        result={
            "logits":logits.detach().clone(),
            "margin":margin.detach().clone(),
            "memory":torch.stack(states,dim=1).detach().clone(),
        }
        if derivative:
            if not torch.is_grad_enabled():
                raise AssertionError("gradient tracing requires autograd enabled")
            gradients=torch.autograd.grad(
                margin.sum(),inputs+outputs,
                retain_graph=False,allow_unused=False)
            result["grad_gelu"]=[v[:,1,:].detach().clone() for v in gradients[:L]]
            result["grad_down"]=[v[:,1,:].detach().clone() for v in gradients[L:]]
        return result
    finally:
        for h in handles:
            h.remove()


def finite_difference_check(model,catalog,candidate,*,eps=FD_EPS):
    """Source-training-only verification of clean actual margin gradient."""
    if eps<=0 or not candidate["nodes"]:
        raise ValueError("invalid source-only gradient check")
    bits=as_tensor(catalog["train"][:16])
    grad=traced_forward(model,bits,mode="clean",derivative=True)
    orientation=grad["logits"].argmax(1)
    visit,channel=(int(x) for x in candidate["nodes"][0].split(":"))
    with torch.no_grad():
        plus=traced_forward(model,bits,mode="clean",orientation=orientation,
                            perturb=(visit,channel,eps))
        minus=traced_forward(model,bits,mode="clean",orientation=orientation,
                             perturb=(visit,channel,-eps))
    observed=(plus["margin"]-minus["margin"])/(2*eps)
    predicted=grad["grad_gelu"][visit][:,channel]
    error=(observed-predicted).abs()
    mx=float(error.max())
    if mx>FD_MAX_ABSOLUTE_ERROR:
        raise AssertionError("source-training finite differences disagree with native GELU Jacobian")
    return {
        "source_split":"train[:16] only",
        "recurrent_visit":visit,"native_neuron":channel,
        "epsilon":eps,
        "max_absolute_fd_error":mx,
        "mean_absolute_fd_error":float(error.mean()),
        "max_allowed_error":FD_MAX_ABSOLUTE_ERROR,
        "original_16_source_histories_only":True,
    }


def analyze_pair(model,trace_clean,trace_hubs,trace_source,trace_control,donors,
                 local_physical_doses):
    """Recompute 64D/32D local gradient identities and nonlinear transport."""
    weight=model.block.mlp_down.weight.detach()
    memory_source=trace_source["memory"]
    memory_sham=trace_control["memory"]
    memory_hubs=trace_hubs["memory"]
    B=memory_source.shape[0]
    lin_clean=torch.zeros(B)
    lin_hubs=torch.zeros(B)
    per_site={}
    identities=[]
    cosines=[]
    for visit in sorted(donors):
        delta=(donors[visit]["native"]-donors[visit]["sham"]).detach()
        delta_out=delta @ weight.T
        grad_gelu_hubs=trace_hubs["grad_gelu"][visit]
        grad_out_hubs=trace_hubs["grad_down"][visit]
        in_dot=(grad_gelu_hubs*delta).sum(-1)
        out_dot=(grad_out_hubs*delta_out).sum(-1)
        identity=(in_dot-out_dot).abs()
        max_err=float(identity.max())
        if max_err>JACOBIAN_IDENTITY_MAX_ERROR:
            raise AssertionError("GELU/output Jacobian physical coordinate identity failed")
        identities.append(max_err)
        local_clean=(trace_clean["grad_gelu"][visit]*delta).sum(-1)
        lin_clean+=local_clean
        lin_hubs+=in_dot
        cos=F.cosine_similarity(delta_out,grad_out_hubs,dim=-1,eps=1e-10)
        # The nonlinear comparison carries all downstream interactions:
        # these are only *local tangent* directional scores.
        per_site[str(visit)]={
            "clean_directional_margin_derivative":local_clean.double().tolist(),
            "hubs_directional_margin_derivative":in_dot.double().tolist(),
            "direct_32d_delta_norm":delta_out.norm(dim=1).double().tolist(),
            "hubs_margin_gradient_32d_norm":grad_out_hubs.norm(dim=1).double().tolist(),
            "hubs_direction_cosine":cos.double().tolist(),
            "max_local_64d_32d_jacobian_identity_error":max_err,
            "native_dose":local_physical_doses[visit]["full_left_dose"],
            "control_dose":local_physical_doses[visit]["full_right_dose"],
            "achieved_common_dose":local_physical_doses[visit]["shared_achievable_dose"],
            "left_gain":local_physical_doses[visit]["left_gain"],
            "right_gain":local_physical_doses[visit]["right_gain"],
        }
        cosines.extend(cos.detach().tolist())
    actual=trace_source["margin"]-trace_control["margin"]
    discrepancy=actual-lin_hubs
    norm_src=(memory_source-memory_hubs).norm(dim=-1)
    norm_control=(memory_sham-memory_hubs).norm(dim=-1)
    norm_delta=(memory_source-memory_sham).norm(dim=-1)
    # [batch,12] arrays, each visit is the same tied-weight block's
    # chronological output, never an independent statistical sample.
    return {
        "clean_linear_margin_delta":lin_clean.detach().double().tolist(),
        "hubs_linear_margin_delta":lin_hubs.detach().double().tolist(),
        "nonlinear_exact_margin_delta":actual.detach().double().tolist(),
        "nonlinear_minus_hubs_linear_error":discrepancy.detach().double().tolist(),
        "gradient_direction_sign_agreement":[
            int(float(a)*float(b)>0) if abs(float(a))>1e-7 and abs(float(b))>1e-7
            else None for a,b in zip(lin_hubs,actual)],
        "source_vs_control_memory_drift_by_visit":
            norm_delta.double().tolist(),
        "source_vs_hubs_memory_drift_by_visit":
            norm_src.double().tolist(),
        "control_vs_hubs_memory_drift_by_visit":
            norm_control.double().tolist(),
        "local_jacobian":per_site,
        "max_native_gelu_to_mlpdown_jacobian_discrepancy":max(identities),
        "mean_absolute_nonlinear_remainder":float(discrepancy.abs().mean()),
    }


def panel_source_records(folder: Path):
    """Load and verify every frozen NEC-03I-B raw source/sham transcript."""
    panel=json.loads((folder/"source_blind_control_panel.json").read_text())
    if (panel.get("schema")!="NEC-03I-B/4-hash-active-rank-shams/v1"
        or panel.get("hash_scramble_variants")!=list(VARIANTS)
        or panel.get("source_contexts_used_by_new_panel")!=TOTAL_CONTEXTS
        or panel.get("reserved_contexts_scored")!=0
        or panel.get("confirmation_worlds_trained_or_scored")!=0):
        raise AssertionError("frozen active-sham panel source differs")
    raw={}
    for variant in VARIANTS:
        row=panel["comparisons"][variant]
        if row["variant"]!=variant:
            raise AssertionError("changed hash-control permutation ID")
        path=folder/row["raw_filename"]
        if hash_file(path)!=row["raw_context_sha256"]:
            raise AssertionError("input hash-sham raw transcript corrupted")
        original_items=[json.loads(line) for line in path.read_text().splitlines()]
        records={r["context_index"]:r for r in original_items}
        if (len(original_items)!=TOTAL_CONTEXTS or
            len(records)!=TOTAL_CONTEXTS or
            set(records)!=set(range(TOTAL_CONTEXTS))):
            raise AssertionError("input hash-sham transcript has missing/duplicate contexts")
        raw[variant]=records
    return panel,raw


def run(root: Path, panel: Path, out: Path, lock_path: Path):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    lock=verify_direction_lock()
    model_source=load_frozen_source(root,lock_path)
    if model_source["original_root"]["v2_protocol_lock_sha256"]!=lock["original_v2_lock_sha256"]:
        raise AssertionError("original V2 source locked hash changed")
    src_model=model_source["model"]
    world=model_source["world"]
    if world["world_index"] not in range(8,16):
        raise PermissionError("NEC-03J only examines original opened development worlds")
    candidate=model_source["candidate"]
    full_catalog=model_source["catalog"]
    orig_frozen=model_source["frozen_source"]
    original_panel,original_rows=panel_source_records(panel)
    if (original_panel["world_sha256"]!=world["world_sha256"]
        or original_panel["native_model_sha256"]!=model_source["model_digest"]
        or original_panel["source_sae_checkpoint_sha256"]!=
                orig_frozen["source_sae"]["source_sae_sha256"]):
        raise AssertionError("active-sham panel/model native source identities differ")
    finite=finite_difference_check(src_model,full_catalog,candidate)
    out.mkdir(parents=True,exist_ok=True)
    raw=[]
    context_summaries=[]
    for context in range(TOTAL_CONTEXTS):
        bits=as_tensor(full_catalog["development"][16*context:16*(context+1)])
        y=target_bits(bits,world)
        clean=traced_forward(src_model,bits,mode="clean",derivative=True)
        orientation=clean["logits"].argmax(1)
        acts=clean_activations(src_model,bits)
        all_world_pairs=[]
        # Each hash variant uses the same source-selected visit set and hubs.
        for variant in VARIANTS:
            donors,gates,doses,route=pair_donors(
                src_model,acts,candidate,orig_frozen["graph_hubs"],
                model_source["sae"],model_source["mean"],model_source["scale"],
                orig_frozen["source_ranking_per_visit"],
                world["trunk_seed"],variant)
            frozen_record=original_rows[variant][context]
            with torch.no_grad():
                source=traced_forward(src_model,bits,mode="native",
                                      orientation=orientation,donors=donors,gates=gates)
                control=traced_forward(src_model,bits,mode="sham",
                                       orientation=orientation,donors=donors,gates=gates)
            check_logits(clean["logits"],frozen_record["clean_logits"])
            check_logits(source["logits"],frozen_record["source_logits"])
            check_logits(control["logits"],frozen_record["control_logits"])
            if variant==0:
                hubs=traced_forward(src_model,bits,mode="hubs",derivative=True,
                                    orientation=orientation,donors=donors,gates=gates)
                check_logits(hubs["logits"],frozen_record["hubs_logits"])
                fixed_hubs=hubs
            else:
                check_logits(fixed_hubs["logits"],frozen_record["hubs_logits"])
                hubs=fixed_hubs
            diagnosis=analyze_pair(src_model,clean,hubs,source,control,donors,doses)
            true_clean=(clean["logits"].argmax(1)==y)
            true_source=(source["logits"].argmax(1)==y)
            true_control=(control["logits"].argmax(1)==y)
            correct_case_delta=(true_source.int()-true_control.int())
            if correct_case_delta.shape!=(16,):
                raise AssertionError("incomplete focal quartet")
            raw.append({
                "schema":SCHEMA,"world_id":world["world_id"],
                "variant":variant,"context_index":context,
                "source_context_bits":frozen_record["source_context_bits"],
                "focal_mask_order":list(range(16)),
                "labels":y.tolist(),
                "clean_chosen_class":orientation.tolist(),
                "clean_logits":clean["logits"].double().tolist(),
                "hubs_logits":hubs["logits"].double().tolist(),
                "source_logits":source["logits"].double().tolist(),
                "control_logits":control["logits"].double().tolist(),
                "source_vs_control_correctness_delta":correct_case_delta.tolist(),
                "direction_and_transport":diagnosis,
            })
    raw_bytes=b"".join(encoded(item) for item in raw)
    (out/"direction_transport_raw.jsonl").write_bytes(raw_bytes)
    if model_digest(src_model)!=model_source["model_digest"]:
        raise AssertionError("native recurrent trunk weights changed during Jacobian trace")
    report={
        "schema":SCHEMA,
        "world_id":world["world_id"],"world_index":world["world_index"],
        "world_sha256":world["world_sha256"],
        "native_model_sha256":model_source["model_digest"],
        "source_steering_candidate_id":candidate["id"],
        "source_site_kind":model_source["original_root"]["source_site_kinds"]["composed"],
        "original_ITT_correctness":
            model_source["original_root"]["primary_effect"]["root_primary_effect"],
        "source_worlds_are_previously_opened":True,
        "source_training_only_fd_preflight":finite,
        "donor_variants":list(VARIANTS),
        "local_gradient_target":"clean-argmax class margin, frozen orientation; source labels not used for gradient",
        "local_gradient_coordinate":"64-D native GELU input and 32-D MLP down output at memory token",
        "recurrent_memory_trace":"12 x 32D memory after each tied-weight block",
        "raw_file":"direction_transport_raw.jsonl",
        "raw_sha256":hashlib.sha256(raw_bytes).hexdigest(),
        "raw_variant_context_rows":len(raw),
        "raw_focal_predictions":len(raw)*16,
        "original_nec03h_run_id":PARENT_RUN,
        "original_nec03ib_run_id":CONTROL_RUN,
        "original_source_control_panel_sha256":hash_file(panel/"source_blind_control_panel.json"),
        "original_v2_lock_sha256":hash_file(lock_path),
        "NEC03J_direction_lock_sha256":lock["lock_sha256"],
        "source_model_weights_unchanged":True,
        "confirmation_worlds_scored":0,"reserved_context_groups_scored":0,
        "evidence_class":"posthoc development, no prospective causal confirmation",
        "FSI_01_gate":"REDESIGN_OR_PILOT_ONLY",
    }
    (out/"direction_transport_report.json").write_bytes(encoded(report))
    print("NEC03J_DIRECTION_TRACE_SAVED")
    return report


def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--panel",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--lock",type=Path,
                   default=Path(__file__).with_name("NEC03H_V2_LOCK.json"))
    a=p.parse_args(argv)
    run(a.root,a.panel,a.out,a.lock)


if __name__=="__main__":
    main()
