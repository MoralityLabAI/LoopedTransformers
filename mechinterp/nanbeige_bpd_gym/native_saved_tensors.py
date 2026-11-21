"""Bounded saved tensors with immutable native-weight aliasing for BPD.

Staged BF16 kernels already have frozen CPU source parameters. Reuse those
exact views instead of making a dense CPU copy for every recurrent graph.
Other activations receive independent CPU snapshots, with a shared live cap.
"""
import torch


def storage_key(tensor):return tensor.untyped_storage().data_ptr()


class SavedTensor:
    def __init__(self,owner,tensor,device,owned=0,parameter=None):
        self.owner=owner;self.tensor=tensor;self.device=device;self.owned=owned
        self.parameter=parameter;self.version=None if parameter is None else parameter._version
    def __del__(self):
        if self.owned:self.owner.live_bytes-=self.owned


class NativeSavedTensors:
    def __init__(self,loop,maximum_added_cpu_bytes=2*1024**3):
        self.loop=loop;self.maximum=int(maximum_added_cpu_bytes);self.live_bytes=0;self.peak_bytes=0
        self.copied_bytes=0;self.staged_aliases=0;self.resident_cache_hits=0;self.activation_snapshots=0
        self.cache={};self.verified=set()
        self.stage={storage_key(p):(name,p) for name,p in loop.stage_layer.named_parameters()} if loop.stage_layer is not None else {}
        self.resident={storage_key(p):p for p in loop.model.parameters() if p.device.type!='cpu' and not p.requires_grad}

    def charge(self,count):
        if self.live_bytes+count>self.maximum:raise MemoryError('Native BPD saved-tensor CPU budget exceeded; preserve this attempt')
        self.live_bytes+=count;self.peak_bytes=max(self.peak_bytes,self.live_bytes);self.copied_bytes+=count

    @staticmethod
    def view(tensor,original,source):
        offset=source.storage_offset()+tensor.storage_offset()-original.storage_offset()
        return source.detach().as_strided(tensor.shape,tensor.stride(),offset)

    def pack(self,tensor):
        device=tensor.device
        if device.type=='cpu':return SavedTensor(self,tensor.detach(),device)
        key=storage_key(tensor)
        if key in self.stage:
            name,original=self.stage[key]
            source=self.loop.layers[self.loop.active_layer].get_parameter(name)
            if source.device.type!='cpu' or source.requires_grad or tensor.dtype!=source.dtype:raise ValueError('Staged source must be the frozen matching CPU operator')
            view=self.view(tensor,original,source)
            binding=(self.loop.active_layer,name,tuple(tensor.shape),tuple(tensor.stride()),tensor.storage_offset())
            if binding not in self.verified:
                # Verify each physical binding once; later invocations reuse the
                # same immutable CPU parameter, not the mutable GPU stage.
                if not torch.equal(tensor.detach().cpu(),view):raise ValueError('Staged CPU weight alias disagrees with the executed kernel')
                self.verified.add(binding)
            self.staged_aliases+=1
            return SavedTensor(self,view,device,parameter=source)
        if key in self.resident:
            original=self.resident[key]
            if key not in self.cache:
                count=original.numel()*original.element_size();self.charge(count)
                try:self.cache[key]=original.detach().cpu()
                except Exception:self.live_bytes-=count;raise
            self.resident_cache_hits+=1
            return SavedTensor(self,self.view(tensor,original,self.cache[key]),device,parameter=original)
        count=tensor.numel()*tensor.element_size();self.charge(count)
        try:snapshot=tensor.detach().to('cpu',copy=True)
        except Exception:self.live_bytes-=count;raise
        self.activation_snapshots+=1
        return SavedTensor(self,snapshot,device,owned=count)

    @staticmethod
    def unpack(saved):
        if saved.parameter is not None and saved.parameter._version!=saved.version:raise RuntimeError('A frozen source weight changed before backward')
        return saved.tensor.to(saved.device)

    def context(self):return torch.autograd.graph.saved_tensors_hooks(self.pack,self.unpack)

    def statistics(self):
        return {'maximum_added_cpu_bytes':self.maximum,'live_added_cpu_bytes':self.live_bytes,'peak_added_cpu_bytes':self.peak_bytes,
            'total_copied_bytes':self.copied_bytes,'staged_weight_aliases':self.staged_aliases,'verified_staged_bindings':len(self.verified),
            'resident_cache_hits':self.resident_cache_hits,'activation_snapshots':self.activation_snapshots,
            'scope':'CPU source weights aliased; activation snapshots and resident weight cache charged to live budget'}
