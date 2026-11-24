"""Deliver a pass-1 attention write edit and a pass-2 K rescue."""
from contextlib import contextmanager

from experiments.nan_premlp64.capture import PAIRED_PHASES, same_bf16
from experiments.nan_reader71.native import ReaderInstrument, SELF, SITES, summarize
from mechinterp.nanbeige_bpd_gym.chunked_cache import ChunkedTwoLoopCache


WRITER = (1,1)
WRITER_SITES = ("writer_input","writer_raw","writer_reached")


class WriterReaderInstrument(ReaderInstrument):
    def __init__(self,loop,replay):
        super().__init__(loop,0,replay)
        self.writer = loop.n - 1
        module = loop.layers[self.writer] if self.writer < loop.n_gpu_layers else loop.stage_layer
        if module is None:
            raise ValueError("The final attention writer has no execution layer")
        self.handles.append(module.self_attn.o_proj.register_forward_hook(self.write))
        self.writer_edit = None

    @contextmanager
    def context(self,modes):
        with super().context(modes):
            self.values.update({site:{} for site in WRITER_SITES})
            self.writer_edits = 0
            yield
            if self.writer_edits != int(self.writer_edit is not None):
                raise RuntimeError("Expected one scoped pass-1 attention write edit")

    def write(self,_module,args,output):
        if not self.active or self.loop.active_layer!=self.writer:
            return None
        self.record("writer_input",args[0])
        self.record("writer_raw",output)
        if self.writer_edit is not None and self.frame.phase==1 and self.frame.capture_offsets[1] is not None:
            reached = self.replace(output,self.writer_edit)
            self.writer_edits += 1
            self.record("writer_reached",reached)
            return reached
        self.record("writer_reached",output)
        return None


class WriterReaderCapture:
    def __init__(self,loop,head,*,chunk=128,replay=None):
        self.cache = ChunkedTwoLoopCache(loop,prefill_chunk=chunk,observe_layers=(),
                                         instrument_factory=lambda model:WriterReaderInstrument(model,replay))
        self.instrument = self.cache.instrument
        self.head = head

    def capture(self,prompts,*,edit=None,rescue=None):
        import torch

        hook = self.instrument
        if (len(prompts)!=2 or hook.active or (rescue is not None and edit is None)
                or (edit is not None and prompts[0]!=prompts[1])):
            raise ValueError("Writer edit requires its paired duplicate clean row")
        hook.writer_edit,hook.rescue = edit,rescue
        try:
            result = self.cache.prefill(prompts,capture=False)
            logits = torch.stack([self.head(out).detach().cpu() for out in result["boundaries"]])
            if logits.shape!=(2,2,len(self.head.weight)) or not torch.isfinite(logits).all():
                raise RuntimeError("Writer-reader prefill lacks a full vocabulary head")
            return hook.values | {"logits":logits,"writer_edits":hook.writer_edits,"rescues":hook.rescues}
        finally:
            hook.writer_edit = hook.rescue = None
            self.cache.reset()

    def close(self):
        self.cache.close()


def check_writer(clean,reached,*,identity=False):
    import torch

    for site in (*SITES,*WRITER_SITES):
        for phase,row in PAIRED_PHASES:
            required = identity or row==0 or (phase==1 and site!="writer_reached")
            if required and not same_bf16(clean[site][phase,row],reached[site][phase,row]):
                raise RuntimeError("The clean companion, pre-writer state or identity arm changed")
    if (not torch.equal(clean["logits"][:,0],reached["logits"][:,0])
            or (identity and not torch.equal(clean["logits"],reached["logits"]))):
        raise RuntimeError("Unexpected full-vocabulary change in a clean row")


def run_pair(capture,prompt,labels,choose,check):
    """Deliver the supplied matched plans on their verified native writer bytes."""
    import torch

    prompts = [prompt,prompt]
    check()
    clean = capture.capture(prompts)
    base = clean["writer_raw"][WRITER]
    writes, receipt = choose(clean)
    check()
    zero = capture.capture(prompts,edit=(base,base))
    check_writer(clean,zero,identity=True)
    if not writes:
        return receipt | {"prefills":2,"zero_hook_closed":True}
    if len(labels)!=2 or len(set(labels))!=2 or any(
            type(i) is not int or not 0<=i<clean["logits"].shape[-1] for i in labels):
        raise ValueError("Expected two distinct in-vocabulary answer IDs")
    effects = {}
    for arm,proposed in writes.items():
        check()
        target = torch.as_tensor(proposed,dtype=torch.bfloat16)
        edit = (base,target)
        changed = capture.capture(prompts,edit=edit)
        check_writer(clean,changed)
        if not same_bf16(changed["writer_reached"][WRITER],target):
            raise RuntimeError("Native BF16 attention writer did not deliver its plan")
        effects[arm] = summarize(clean,changed,labels)
        effects[arm]["pass1_max_full_head_delta"] = float((changed["logits"][0,1]-clean["logits"][0,1]).abs().max())
        if arm=="active":
            check()
            pair = (changed["k_raw"][SELF],clean["k_raw"][SELF])
            rescued = capture.capture(prompts,edit=edit,rescue=pair)
            check_writer(clean,rescued)
            if (not same_bf16(rescued["k_reached"][SELF],pair[1])
                    or any(not same_bf16(changed[site][SELF],rescued[site][SELF]) for site in set(SITES)-{"k_reached"})
                    or not same_bf16(changed["writer_reached"][WRITER],rescued["writer_reached"][WRITER])
                    or not torch.equal(changed["logits"][0],rescued["logits"][0])):
                raise RuntimeError("The K rescue changed a site other than its intended pass-2 reader")
            effects["active_k_rescue"] = summarize(clean,rescued,labels)
    check()
    check_writer(clean,capture.capture(prompts),identity=True)
    return receipt | {"status":"closed","prefills":len(writes)+4,"zero_hook_closed":True,
                      "clean_repeat_closed":True,"effects":effects}
