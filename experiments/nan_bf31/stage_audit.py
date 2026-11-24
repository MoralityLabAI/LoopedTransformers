"""Verify actual reusable stage bytes before any pretrained forward."""
import hashlib


def audit(loop,guard=lambda:None,chunk_bytes=1024**2):
    import torch
    store=loop.checkpoint_store;rows=[]
    for i,layer in enumerate(loop.layers):
        if i<loop.n_gpu_layers:continue
        guard();stage=loop._load_stage(layer);source=hashlib.sha256();actual=hashlib.sha256();count=total=0
        for name,p in stage.named_parameters():
            full=f'model.layers.{i}.'+name
            source.update(name.encode());actual.update(name.encode())
            for block in store.chunks(full,chunk_bytes):source.update(block)
            raw=p.detach().reshape(-1).view(torch.uint8)
            for offset in range(0,len(raw),chunk_bytes):
                block=raw[offset:offset+chunk_bytes].cpu().numpy();actual.update(memoryview(block));del block
            count+=1;total+=store.spans[full].length;guard()
        assert source.hexdigest()==actual.hexdigest(),'Actual CUDA/CPU stage bytes differ from checkpoint source'
        rows.append(dict(layer=i,parameters=count,bytes=total,source_sha256=source.hexdigest(),actual_stage_sha256=actual.hexdigest()))
    return dict(status='passed',all_offloaded_layers=len(rows),checks=rows,pretrained_forwards=0,
        byte_scope='Every offloaded-layer parameter after actual synchronous stage copy; not a kernel or buffer hash')
