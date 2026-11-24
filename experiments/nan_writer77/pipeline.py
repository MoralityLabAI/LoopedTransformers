"""Execute the three matched writer plans and retain label-blind head records."""
from experiments.nan_cidose74.geometry import directions
from experiments.nan_premlp64.capture import same_bf16
from experiments.nan_radial76.geometry import ARMS, plan
from experiments.nan_radial76.native import WRITER, WRITER_SITES, check_writer
from experiments.nan_reader71.native import SELF, SITES
from experiments.nan_relabel68.pipeline import compress


MODES = (*ARMS,"active_k_rescue")


def source_index(payload):
    data = {(row["root"],row["arm"],row["world"]):row for row in payload["public"]}
    if len(data)!=len(payload["public"]):
        raise ValueError("Duplicate registered source slot")
    return data,payload["tokens"]["row_bindings"],payload["tokens"]["bare_label_token_ids"]


def head_rows(packet,labels):
    phases = [compress(x,labels) for x in packet["logits"]]
    return [list(row) for row in zip(*phases,strict=True)]


def pair_plan(clean,fit,root,masks,np):
    bank,ci,static,_ = fit
    x,y = (np.stack([clean[k][1,row].float().numpy() for row in (0,1)]).astype(np.float64)
           for k in ("writer_input","writer_raw"))
    gate = masks(x,bank,ci)
    proposals,_ = directions(x,y,gate,np.broadcast_to(static[0],gate.shape),bank,1,root,np)
    if not proposals:
        return [({}, {"status":"zero_native_write"}) for _ in (0,1)]
    return [plan(y[row],proposals["dynamic_tangent"][row],proposals["random_tangent"][row],np) for row in (0,1)]


def batch_baseline(capture,payload,arms,check):
    data,bindings,labels = source_index(payload)
    records,focus = [],{}
    for root in sorted({key[0] for key in data}):
        for arm in arms:
            check()
            pair = [data[root,arm,world]["id"] for world in ("A","B")]
            tokens = [bindings[name]["ids"] for name in pair]
            packet = capture.capture(tokens)
            heads = head_rows(packet,labels)
            for name,scores in zip(pair,heads,strict=True):
                records.append({"id":name,"token_sha256":bindings[name]["sha256"],"phases":scores})
            if arm=="focused":
                focus[root] = packet,tokens
    return records,focus


def matched_duplicate(original,duplicate,row):
    import torch

    return (torch.equal(original["logits"][:,row],duplicate["logits"][:,1])
            and all(same_bf16(original[site][phase,row],duplicate[site][phase,1])
                    for site in (*SITES,*WRITER_SITES) for phase in (1,2)))


def scoped_trial(capture,ids,clean,writes,labels,check):
    import torch

    base = clean["writer_raw"][WRITER]
    prompts = [ids,ids]
    check()
    zero = capture.capture(prompts,edit=(base,base))
    check_writer(clean,zero,identity=True)
    packets,reader = {},{}
    for arm in ARMS:
        check()
        after = torch.as_tensor(writes[arm],dtype=torch.bfloat16)
        edit = base,after
        observed = capture.capture(prompts,edit=edit)
        check_writer(clean,observed)
        if not same_bf16(observed["writer_reached"][WRITER],after):
            raise RuntimeError("The physical write differs from the certified plan")
        packets[arm] = head_rows(observed,labels)[1]
        reader[arm] = {name: float((observed[f"{name}_reached"][SELF].float()
                                     - clean[f"{name}_reached"][SELF].float()).norm()) for name in ("q","k","v")}
        if arm=="active":
            check()
            rescue = observed["k_raw"][SELF],clean["k_raw"][SELF]
            reached = capture.capture(prompts,edit=edit,rescue=rescue)
            check_writer(clean,reached)
            if (not same_bf16(reached["k_reached"][SELF],rescue[1])
                    or not same_bf16(reached["writer_reached"][WRITER],after)
                    or not torch.equal(reached["logits"][0],observed["logits"][0])
                    or any(not same_bf16(reached[s][SELF],observed[s][SELF]) for s in SITES if s!="k_reached")):
                raise RuntimeError("The K rescue changed an unintended reader or writer")
            packets["active_k_rescue"] = head_rows(reached,labels)[1]
    check()
    check_writer(clean,capture.capture(prompts),identity=True)
    return {"arms":packets,"reader_l2":reader,"rescue_k_exact":True,"zero_hook_exact":True,"clean_repeat_exact":True}


def root_trial(capture,root,original,tokens,labels,fit,masks,np,check):
    planned = pair_plan(original,fit,root,masks,np)
    receipts = [r for _,r in planned]
    if not all(d for d,_ in planned):
        return {"status":"geometry_unqualified","plans":receipts}
    duplicates = []
    for i,ids in enumerate(tokens):
        check()
        packet = capture.capture([ids,ids])
        if not matched_duplicate(original,packet,i):
            return {"status":"mixed_duplicate_unqualified","plans":receipts}
        duplicates.append(packet)
    results = [scoped_trial(capture,ids,clean,edits,labels,check)
               for ids,clean,(edits,_) in zip(tokens,duplicates,planned,strict=True)]
    return {"status":"closed","plans":receipts,"worlds":results}
