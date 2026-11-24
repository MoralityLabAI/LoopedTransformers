"""Compare serial and grouped BF16 execution at the registered CUDA capacity."""

import json
from contextlib import ExitStack
from time import monotonic

from experiments.nan_boundrescue109.core import certified_planner
from experiments.nan_boundrescue109.native import BoundaryCapture
from experiments.nan_delay39.instrument import prepare
from experiments.nan_gated46.gated import preservation
from experiments.nan_premlp64.capture import same_bf16
from experiments.nan_radial76.native import WRITER
from experiments.nan_reader71.native import SELF
from experiments.nan_readout34 import runtime
from experiments.nan_readout34.tokenizer_check import sha
from experiments.nan_readpath37.run import FixedHead, SHARD, write_new
from experiments.nan_relabel68.pipeline import paired_heads
from experiments.nan_rescuegroup110.group import ALL_SITES, PairGroup
from experiments.nan_stream36.device_preflight import preflight, repeated
from experiments.nan_vocab82.heads import partition_sha

from .worker import OFFLOADED_LAYERS, VARIANT, drive


DOSE = .004
FIXTURE_SEEDS = (11101,11102)


def audit_packet(actual, expected, torch):
    if (not torch.equal(actual["logits"],expected["logits"]) or actual["writer_edits"]!=expected["writer_edits"]
            or actual["rescues"]!=expected["rescues"]):
        raise RuntimeError("Grouped CUDA head or scoped counts differ from serial B2")
    for site in ALL_SITES:
        if (actual[site].keys()!=expected[site].keys()
                or any(not same_bf16(t,expected[site][k]) for k,t in actual[site].items())):
            raise RuntimeError("Grouped CUDA reader or writer bytes differ from serial B2")


def capacity(serial, group, loop, pair, labels, position, check, sync, np, torch):
    receipts = []
    offloaded = loop.n-loop.n_gpu_layers
    if offloaded <= 0:
        raise ValueError("Grouped staged preflight requires offloaded layers")
    for letter, ids, seed in zip(("A","B"),pair,FIXTURE_SEEDS,strict=True):
        prompts = [repeated(ids,position)]*2
        started = monotonic()

        def guard():
            check()
            if monotonic()-started > runtime.LIMITS["maximum_slot_seconds"]:
                raise RuntimeError("Grouped capacity probe exceeded the registered duration")

        guard()
        clean = serial.capture(prompts)
        base = clean["writer_raw"][WRITER]
        before = base.float().numpy().astype(np.float64)
        direction = np.random.default_rng(seed).standard_normal(len(before))
        patch = prepare(before,direction,DOSE,.10,.85,np)
        if not patch.eligible:
            raise ValueError("The fixed capacity edit misses its attained BF16 dose")
        after = torch.as_tensor(patch.after,dtype=torch.bfloat16)
        planned = serial.predict(clean,after)
        cfg = {"maximum_head_delta":.125,"maximum_margin_delta":.125}
        if not preservation(clean["logits"][0],planned.logits,labels,cfg)["passed"]:
            raise ValueError("The fixed grouped engineering preflight exceeded the immediate head gate")
        sync()
        began, copies = monotonic(),loop.stage_copies
        edited = serial.capture(prompts,edit=(base,after),planned=planned)
        guard()
        observed_k, clean_k = edited["k_raw"][SELF],clean["k_raw"][SELF]
        if same_bf16(observed_k,clean_k):
            raise ValueError("The fixed capacity edit has no native K displacement")
        rescued = serial.capture(prompts,edit=(base,after),planned=planned,rescue=(observed_k,clean_k))
        sync()
        serial_seconds = monotonic()-began
        serial_copies = loop.stage_copies-copies
        guard()
        begin_prime, prior_prime = monotonic(), loop.stage_copies
        primed = group.prime(prompts,clean)
        sync()
        prime_seconds = monotonic()-begin_prime
        prime_copies = loop.stage_copies-prior_prime
        began, copies = monotonic(),loop.stage_copies
        grouped, restored = group.run(primed,prompts,(base,after),planned,clean,guard)
        sync()
        grouped_seconds,grouped_copies = monotonic()-began,loop.stage_copies-copies
        audit_packet(grouped,edited,torch)
        audit_packet(restored,rescued,torch)
        if ((serial_copies,prime_copies,grouped_copies) != (4*offloaded,offloaded,offloaded)
                or not same_bf16(restored["k_reached"][SELF],clean_k)):
            raise RuntimeError("Grouped parity probe changed stage counts or rescue delivery")
        receipts.append({"probe":letter,"position":position,"full_heads_exact":True,"native_vectors_exact":True,
                         "exact_K_restored":True,"phase2_branches":2,"serial_stage_copies":serial_copies,
                         "prime_stage_copies":prime_copies,"grouped_stage_copies":grouped_copies,
                         "serial_wall_seconds":serial_seconds,"prime_wall_seconds":prime_seconds,
                         "grouped_wall_seconds":grouped_seconds,"achieved_fraction":patch.realized_l2/
                         float(np.linalg.norm(before))})
        guard()
    return receipts


def device(args, work, guard, payload, classes, admission, np, torch, settings, host_gate):
    from experiments.nan_bf31.loader import verify_checkpoint
    from experiments.nan_cache32.worker import quant_state_digest
    from experiments.nan_stream36.batch_engine import BatchedDepthEngine
    from mechinterp.nanbeige_bpd_gym.lazy_bf16_loop import digest_dispatch
    from mechinterp.nanbeige_bpd_gym.streamed_vocab_head import StreamedVocabularyHead

    position = settings["max_position"]
    card = torch.cuda.get_device_properties(0)
    if "3050" not in card.name or not 3.5*2**30<=card.total_memory<=4.5*2**30:
        raise RuntimeError("The admitted grouped device must be the 4 GiB RTX 3050")
    torch.set_num_threads(1)
    torch.cuda.set_per_process_memory_fraction(.9)
    torch.cuda.reset_peak_memory_stats()

    def check():
        guard(device=True)

    def slot(start,rows,n):
        if rows not in (1,2) or not 1<=n<=position or monotonic()-start>runtime.LIMITS["maximum_slot_seconds"]:
            raise RuntimeError("The B2 preflight exceeded its registered shape or duration")
        check()

    focus = [p for p in payload["public"] if p["arm"]=="focused"]
    bindings = payload["tokens"]["row_bindings"]
    world_ids = {(p["root"],p["world"]):bindings[p["id"]]["ids"] for p in focus}
    root = max(sorted({p["root"] for p in focus}),key=lambda r:len(world_ids[r,"A"]))
    pair = [world_ids[root,w] for w in ("A","B")]
    labels = tuple(payload["tokens"]["bare_label_token_ids"][key] for key in ("A","B"))
    base = json.loads((settings["repo"]/"experiments/nan_readpath37/registration.json").read_bytes())
    with ExitStack() as stack:
        target_digest = stack.enter_context(digest_dispatch())
        loop,checkpoint,stage = runtime.load_model("bf16",args.snapshot,args.adapter)
        before, quant = target_digest(loop),quant_state_digest(loop)
        if (before != runtime.TARGETS["bf16"] or loop.n != 22 or loop.n-loop.n_gpu_layers != OFFLOADED_LAYERS):
            raise ValueError("The grouped staged layout or target identity changed")
        guard(device=True,force_gpu_sample=True)
        model_head = loop.model.lm_head
        if model_head.bias is not None or len(model_head.weight)!=len(classes):
            raise ValueError("The pinned unbiased full-vocabulary head differs")
        head = StreamedVocabularyHead(model_head.weight,loop.device,8192)
        stack.callback(head.close)
        engine = BatchedDepthEngine(loop,head)
        try:
            depths = preflight(engine,pair,position,slot)
            expected = torch.stack(paired_heads(engine,pair,slot))
        finally:
            engine.close()
        serial = BoundaryCapture(loop,head)
        group = PairGroup(loop,head)
        stack.callback(serial.close)
        stack.callback(group.close)
        if not torch.equal(expected,serial.capture(pair)["logits"]):
            raise RuntimeError("The clean boundary hook changed the uninstrumented B2 head")
        probes = capacity(serial,group,loop,pair,labels,position,check,torch.cuda.synchronize,np,torch)
        guard(device=True,force_gpu_sample=True)
        write_new(work/"preflight.json",{"execution_variant":VARIANT,"depths":depths,"native_heads_exact":True,
                  "head_placement_exact":True,"grouped_probes":probes,"head_resident":False,
                  "partition_sha256":partition_sha(classes),"stage":stage,
                  "peak_allocated_bytes":torch.cuda.max_memory_allocated(),"peak_reserved_bytes":torch.cuda.max_memory_reserved()})
        file = stack.enter_context((args.snapshot/SHARD).open("rb"))
        planner, design = certified_planner("bf16",FixedHead(file,args.snapshot,base,np,check),np)
        counts = drive(work,serial,group,payload,admission,classes,planner,design,host_gate,check,np)
        guard(device=True,force_gpu_sample=True)
        verify_checkpoint(args.snapshot,checkpoint)
        after,q2 = target_digest(loop),quant_state_digest(loop)
        if after!=before or quant!=q2:
            raise RuntimeError("The staged device run changed the pinned decoder or quant state")
    if (counts["baseline_batches"],counts["root_records"]) != (settings["baseline_batches"],settings["roots"]):
        raise RuntimeError("The grouped device did not close its fixed roster")
    pins = {f"{k}_sha256":sha(work/f) for k,f in (("admission","admission.json"),("preflight","preflight.json"),
            ("baseline","baseline.json"),("qualification","qualification.json"),("journal","closed.jsonl"))}
    tags = {key:admission[key] for key in ("experiment","backend","mode","execution_variant","runtime_sha256","published_commit")}
    write_new(work/"terminal.json",tags|counts|pins|{"status":"completed","checkpoint_verified_after":True,
              "target_before":before,"target_after":after,"quant_before":quant,"quant_after":q2,
              "partition_sha256":partition_sha(classes),"peak_allocated_bytes":torch.cuda.max_memory_allocated(),
              "peak_reserved_bytes":torch.cuda.max_memory_reserved()})
