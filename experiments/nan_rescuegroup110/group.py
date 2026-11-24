"""Share one primed B2 residual and a staged layer across edited and clean-K branches."""

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from types import SimpleNamespace

from experiments.nan_boundrescue109.core import run_root
from experiments.nan_boundrescue109.native import ResidualReader
from experiments.nan_boundrescue109.worker import RootCapture
from experiments.nan_premlp64.capture import same_bf16
from experiments.nan_radial76.native import WRITER, WRITER_SITES, check_writer
from experiments.nan_reader71.native import SELF, SITES
from mechinterp.nanbeige_bpd_gym.chunked_cache import ChunkedTwoLoopCache


ALL_SITES = SITES + WRITER_SITES


class PhaseReader(ResidualReader):
    @contextmanager
    def phase(self, phase, sites, rescue=None):
        if self.active or self.writer_edit is not None or self.edit is not None or self.rescue is not None:
            raise RuntimeError("A grouped observer cannot share an active hook")
        self.active = True
        self.values = {site: {} for site in sites}
        self.rescue = rescue
        self.edits = self.rescues = self.writer_edits = 0
        expected = {(phase, 0), (phase, 1)}
        try:
            yield self.values
            if (any(rows.keys() != expected for rows in self.values.values())
                    or self.rescues != int(rescue is not None) or self.edits or self.writer_edits):
                raise RuntimeError("The grouped observer did not close its scoped native calls")
        finally:
            self.active = False
            self.frame = self.rescue = None
            self.values = {}


@dataclass(frozen=True)
class Prime:
    owner: object
    tokens: tuple
    raw: object
    attention: object
    targets: tuple
    lengths: tuple
    observations: dict


class PrimeCache(ChunkedTwoLoopCache):
    def __init__(self, loop, chunk):
        if loop.n < 2:
            raise ValueError("The phase-one writer and phase-two reader must be distinct layers")
        super().__init__(loop, prefill_chunk=chunk, observe_layers=(), instrument_factory=PhaseReader)
        self.first = None

    def _capture(self, h, targets, prefill, capture):
        if not prefill or capture or not self.inplace:
            raise ValueError("Grouped priming requires a read-only duplicate prefill")
        with self.instrument.phase(1, ALL_SITES) as measurements:
            self._walk(h, 1, targets, True)
            if any(self.caches[0].get_seq_length(i) != h.shape[1] for i in range(self.loop.n)):
                raise RuntimeError("The shared phase-one canvas did not close")
            self.first = h.detach(), tuple(targets), measurements
        return {}


class PairGroup:
    def __init__(self, loop, head, *, chunk=128):
        self.cache = PrimeCache(loop, chunk)
        self.loop, self.head = loop, head
        self.owner = object()

    def prime(self, prompts, clean):
        import torch

        if len(prompts) != 2 or prompts[0] != prompts[1] or not prompts[0]:
            raise ValueError("Grouped rescue requires the exact clean duplicate prompts")
        previous = self.loop.phase, self.loop.active_layer
        try:
            with torch.inference_mode():
                self.cache.prefill(prompts, capture=False)
                raw, targets, observations = self.cache.first
                for site in ALL_SITES:
                    if any(not same_bf16(value, clean[site][key]) for key, value in observations[site].items()):
                        raise RuntimeError("Primed phase one differs from the closed clean duplicate")
                return Prime(self.owner, tuple(tuple(p) for p in prompts), raw, self.cache.attention, targets,
                             tuple(self.cache.semantic_lengths), observations)
        finally:
            self.cache.first = None
            self.cache.reset()
            self.loop.phase, self.loop.active_layer = previous

    def layer(self, branch, p, layer, index, rescue, torch):
        hook, h, cache = self.cache.instrument, branch.h, branch.cache
        sites = SITES if index == 0 else WRITER_SITES if index == self.loop.n - 1 else ()
        context = hook.phase(2, sites, rescue) if sites else nullcontext()
        with context as measured:
            for start in range(0, h.shape[1], self.cache.chunk):
                stop = min(start+self.cache.chunk, h.shape[1])
                physical = torch.arange(start, stop, device=self.loop.device)
                if sites:
                    hook.set_frame(SimpleNamespace(
                        phase=2, prefill=True, start=start, stop=stop,
                        valid_lengths=[max(0,min(stop,n)-start) for n in p.lengths],
                        capture_offsets=[i-start if start<=i<stop else None for i in p.targets]))
                mask = self.loop.base._update_causal_mask(p.attention, h[:, start:stop], physical, cache, False)
                values = layer(h[:,start:stop], attention_mask=mask, position_ids=physical.unsqueeze(0).expand(2,-1),
                               past_key_value=cache, output_attentions=False, use_cache=True,
                               cache_position=physical, loop_idx=0)[0]
                if any(x.untyped_storage().data_ptr() == h.untyped_storage().data_ptr()
                       for x in (cache.key_cache[index],cache.value_cache[index])):
                    raise RuntimeError("A grouped K/V cache aliases its residual canvas")
                h[:,start:stop].copy_(values)
        if sites:
            branch.observations.update(measured)

    def run(self, p, prompts, edit, planned, clean, guard):
        import torch

        if p.owner is not self.owner or tuple(tuple(row) for row in prompts) != p.tokens or planned is None:
            raise ValueError("The grouped request lacks its bound duplicate and first-pass plan")
        before, after = edit
        token = p.raw[1,p.targets[1]].detach().cpu()
        if (not same_bf16(token,before) or before.shape != after.shape or after.dtype != torch.bfloat16
                or after.device.type != "cpu" or not torch.isfinite(after).all() or same_bf16(before,after)):
            raise ValueError("The grouped boundary edit is stale or degenerate")
        previous = self.loop.phase,self.loop.active_layer
        branches = []
        try:
            with torch.inference_mode():
                guard()
                raw = p.raw.clone()
                raw[1,p.targets[1]].copy_(after)
                normalized = self.loop.base.norm(raw)
                current = torch.stack([normalized[i,j] for i,j in enumerate(p.targets)])
                if not same_bf16(current.detach().cpu(),planned.normalized):
                    raise RuntimeError("Grouped normalized boundary differs before pass 2")
                current_logits = self.head(current).detach().cpu()
                if not torch.equal(current_logits,planned.logits):
                    raise RuntimeError("Grouped first-pass head differs before pass 2")
                branches = [SimpleNamespace(h=x,cache=self.cache.factory(),observations={})
                            for x in (normalized,normalized.clone())]
                if branches[0].cache is branches[1].cache:
                    raise RuntimeError("Edited and rescued branches share a K/V cache")
                self.loop.phase = 2
                for index in range(self.loop.n):
                    guard()
                    self.loop.active_layer = index
                    layer = (self.loop.layers[index] if index<self.loop.n_gpu_layers
                             else self.loop._load_stage(self.loop.layers[index]))
                    if layer.self_attn.layer_idx != index:
                        raise RuntimeError("Grouped staging changed the physical layer identity")
                    self.layer(branches[0],p,layer,index,None,torch)
                    rescue = (branches[0].observations["k_raw"][SELF],clean["k_raw"][SELF]) if index==0 else None
                    self.layer(branches[1],p,layer,index,rescue,torch)
                outputs = []
                for i,branch in enumerate(branches):
                    if any(branch.cache.get_seq_length(k)!=len(p.tokens[0]) for k in range(self.loop.n)):
                        raise RuntimeError("The separate phase-two cache did not close")
                    final = self.loop.base.norm(branch.h)
                    phase2 = self.head(torch.stack([final[r,j] for r,j in enumerate(p.targets)])).detach().cpu()
                    logits = torch.stack([current_logits,phase2])
                    if logits.shape != (2,2,len(self.head.weight)) or not torch.isfinite(logits).all():
                        raise RuntimeError("Grouped reader rescue lacks a complete B2 head")
                    packet = {k:p.observations[k] | branch.observations[k] for k in ALL_SITES}
                    packet["writer_reached"][WRITER] = after.clone()
                    packet.update(logits=logits,writer_edits=1,rescues=i)
                    check_writer(clean,packet)
                    outputs.append(packet)
                return tuple(outputs)
        finally:
            for branch in branches:
                branch.cache = branch.h = None
            self.loop.phase,self.loop.active_layer = previous

    def close(self):
        self.cache.close()


class RootBranches:
    """Adapt the grouped pair to the ordinary observed-then-rescued root interface."""

    def __init__(self, serial, group, guard):
        self.serial, self.group, self.guard = serial, group, guard
        self.clean = {}
        self.prepared = None
        self.pending = None
        self.approved = False
        self.grouped_pairs = 0
        self.primed_worlds = 0

    def predict(self, clean, after):
        return self.serial.predict(clean, after)

    def capture(self, prompts, *, edit=None, planned=None, rescue=None):
        tokens = tuple(tuple(row) for row in prompts)
        if self.pending is not None:
            previous, before, head, expected, saved = self.pending
            self.pending = None
            if (tokens != previous or edit is None or planned is not head or rescue is None
                    or any(not same_bf16(a,b) for a,b in zip(edit,before,strict=True))
                    or any(not same_bf16(a,b) for a,b in zip(rescue,expected,strict=True))):
                raise ValueError("The clean-K request differs from the already bound rescue branch")
            return saved
        if edit is None or same_bf16(*edit):
            result = self.serial.capture(prompts,edit=edit,planned=planned,rescue=rescue)
            if edit is None and len(prompts)==2 and prompts[0]==prompts[1]:
                self.clean[tokens] = result
            return result
        if not self.approved or rescue is not None or tokens not in self.clean:
            raise ValueError("Grouped continuation requires acknowledged first-pass root qualification")
        clean = self.clean[tokens]
        if self.prepared is None or self.prepared.tokens != tokens:
            self.prepared = self.group.prime(prompts,clean)
            self.primed_worlds += 1
        observed, saved = self.group.run(self.prepared,prompts,edit,planned,clean,self.guard)
        self.pending = tokens, edit, planned, (observed["k_raw"][SELF],clean["k_raw"][SELF]), saved
        self.grouped_pairs += 1
        return observed


def grouped_root(capture, group, root, prompts, labels, plan, design, commit_plan, check, np, *, anchor, bare):
    adapter = RootBranches(capture,group,check)

    def commit(receipt):
        commit_plan(receipt)
        adapter.approved = receipt["status"] == "qualified"

    observed = RootCapture(adapter,anchor,prompts,bare)
    result = run_root(observed,root,prompts,labels,plan,design,commit,check,np)
    if adapter.pending is not None or 2*adapter.grouped_pairs != result["edited_pass2_runs"] or len(observed.rows) != 2*adapter.grouped_pairs:
        raise RuntimeError("An edited or rescued branch remained unconsumed")
    work = {"grouped_endpoint_pairs":adapter.grouped_pairs,"primed_worlds":adapter.primed_worlds}
    return result, observed.rows, work
