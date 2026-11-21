"""Bounded parameter hashing and shared evidence I/O; no source oracle."""
import hashlib
import torch
from experiments.nan_fit07.io import read,rows,sha,write


def target_digest(loop,chunk_bytes=16*1024**2):
    digest=hashlib.sha256()
    for name,parameter in loop.model.named_parameters():
        digest.update(name.encode())
        if not parameter.is_contiguous():raise ValueError('Pinned parameters must retain their contiguous layout')
        data=parameter.detach().reshape(-1).view(torch.uint8)
        for start in range(0,len(data),chunk_bytes):
            values=data[start:start+chunk_bytes].cpu().numpy();digest.update(memoryview(values))
    return digest.hexdigest()
