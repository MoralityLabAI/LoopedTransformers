"""Unregistered full-vocabulary reader with one reusable weight-row stage.

The unchanged head stays on CPU (read-only mapping allowed). No model loading,
quantization or cached-worker registration occurs here. CPU execution supports
small engineering checks; real CUDA parity/memory requires its own admission.
"""


class StreamedVocabularyHead:
    def __init__(self,weight,device,chunk_rows=8192):
        import torch
        if weight.ndim!=2 or weight.device.type!='cpu' or weight.requires_grad:raise ValueError('Frozen CPU vocabulary matrix required')
        if type(chunk_rows) is not int or chunk_rows<1:raise ValueError('Positive vocabulary row chunk required')
        self.weight=weight;self.device=torch.device(device);self.chunk=chunk_rows
        self.stage=torch.empty((min(chunk_rows,weight.shape[0]),weight.shape[1]),device=self.device,dtype=weight.dtype)
        if self.stage.data_ptr()==weight.data_ptr():raise RuntimeError('The writable stage must not alias immutable source weights')
        self.calls=0;self.last_chunks=0;self.last_copied_bytes=0
    @property
    def staging_bytes(self):return 0 if self.stage is None else self.stage.numel()*self.stage.element_size()
    def __call__(self,hidden):
        import torch
        import torch.nn.functional as F
        if self.stage is None:raise RuntimeError('Vocabulary stage is closed')
        if hidden.ndim!=2 or hidden.shape[1]!=self.weight.shape[1] or hidden.device!=self.device or hidden.dtype!=self.weight.dtype or hidden.requires_grad:
            raise ValueError('Frozen boundary matrix with matching device/width/dtype required')
        self.last_chunks=0;self.last_copied_bytes=0;pieces=[]
        with torch.inference_mode():
            for start in range(0,self.weight.shape[0],self.chunk):
                stop=min(start+self.chunk,self.weight.shape[0]);n=stop-start
                self.stage[:n].copy_(self.weight[start:stop],non_blocking=False)
                pieces.append(F.linear(hidden,self.stage[:n]).float())
                self.last_chunks+=1;self.last_copied_bytes+=n*self.weight.shape[1]*self.weight.element_size()
            self.calls+=1
            return torch.cat(pieces,dim=-1)
    def close(self):self.stage=None
