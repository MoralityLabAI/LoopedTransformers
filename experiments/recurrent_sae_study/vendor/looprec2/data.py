"""Natural-text data from the local HF cache (offline): pile-10k (train / held-out) and GSM8K text."""
from __future__ import annotations

import glob
import os

import numpy as np
import torch

EOS = 151643
CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache")


def _hf_file(pattern: str) -> str:
    fs = sorted(glob.glob(os.path.expanduser(pattern)))
    if not fs:
        raise FileNotFoundError(pattern)
    return fs[0]


def _tokenize_docs(tok_file: str, docs: list[str], max_tokens: int) -> np.ndarray:
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(tok_file)
    out = []
    for i in range(0, len(docs), 256):
        for e in tok.encode_batch(docs[i:i + 256], add_special_tokens=False):
            out.extend(e.ids[:max_tokens] + [EOS])
    return np.array(out, dtype=np.int32)


class TextData:
    def __init__(self, tok_file: str, n_train_docs: int = 4000, n_eval_docs: int = 300, max_doc_tokens: int = 2048):
        os.makedirs(CACHE, exist_ok=True)
        f = os.path.join(CACHE, f"pile_{n_train_docs}_{n_eval_docs}_{max_doc_tokens}.npz")
        if os.path.exists(f):
            z = np.load(f)
            self.train, self.heldout = z["train"], z["heldout"]
        else:
            import pandas as pd
            df = pd.read_parquet(_hf_file("~/.cache/huggingface/hub/datasets--NeelNanda--pile-10k/snapshots/*/data/*.parquet"))
            docs = df["text"].tolist()
            self.train = _tokenize_docs(tok_file, docs[:n_train_docs], max_doc_tokens)
            self.heldout = _tokenize_docs(tok_file, docs[-n_eval_docs:], max_doc_tokens)
            np.savez(f, train=self.train, heldout=self.heldout)
        g = os.path.join(CACHE, "gsm8k_test.npy")
        if os.path.exists(g):
            self.gsm = list(np.load(g, allow_pickle=True))
        else:
            import pandas as pd
            df = pd.read_parquet(_hf_file("~/.cache/huggingface/hub/datasets--openai--gsm8k/snapshots/*/main/test-*.parquet"))
            texts = [f"Question: {q}\nAnswer: {a}" for q, a in zip(df["question"], df["answer"])][:256]
            from tokenizers import Tokenizer
            tok = Tokenizer.from_file(tok_file)
            self.gsm = [np.array(e.ids[:256], dtype=np.int32) for e in tok.encode_batch(texts, add_special_tokens=False)]
            np.save(g, np.array(self.gsm, dtype=object), allow_pickle=True)

    def train_batch(self, rng: np.random.Generator, bs: int, ctx: int) -> torch.Tensor:
        starts = rng.integers(0, len(self.train) - ctx - 1, size=bs)
        return torch.from_numpy(np.stack([self.train[s:s + ctx] for s in starts]).astype(np.int64))

    def heldout_chunks(self, n: int, ctx: int) -> torch.Tensor:
        k = len(self.heldout) // ctx
        k = min(k, n)
        return torch.from_numpy(self.heldout[:k * ctx].reshape(k, ctx).astype(np.int64))

    def gsm_batches(self, n: int = 64):
        """list of (ids [1,T]) - evaluated one by one or padded by the caller"""
        return [torch.from_numpy(g.astype(np.int64))[None] for g in self.gsm[:n]]
