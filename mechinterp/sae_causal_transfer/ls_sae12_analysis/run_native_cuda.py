"""Run the registered LS-SAE12 model stages unmodified on a CUDA device.

Lives outside active_reasoning_v1/ so committing it does not trigger the
registered workflow. The registered functions in active_reasoning_v1.run_native
run as written; four shims move them to the GPU:
  * TRM.load_trm_checkpoint and TRM.encode_questions receive "cuda" where the
    registered code passes "cpu";
  * the default tensor device is cuda, so tensors built without a device
    (power-iteration seeds, edit vectors) land on the GPU;
  * Tensor.numpy copies to host first (a no-op for host tensors).
Float32 matmuls run at full IEEE precision (no TF32); attention stays on the
MATH SDPA backend the registered code selects; deterministic algorithms are
requested with warn_only. The platform fingerprint in every receipt adds the
GPU name, CUDA version and the shim list. Arguments are those of run_native.
"""
from __future__ import annotations
import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from mechinterp.loopscope02.scripts import trm_sudoku as TRM
from mechinterp.sae_causal_transfer.active_reasoning_v1 import run_native as RN

DEVICE = "cuda"
SHIMS = ["load_trm_checkpoint->cuda", "encode_questions->cuda", "default_device=cuda",
         "Tensor.numpy copies to host"]


def install():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device not available")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_default_device(DEVICE)

    load, encode, to_numpy, fingerprint = (TRM.load_trm_checkpoint, TRM.encode_questions,
                                           torch.Tensor.numpy, RN.platform_fingerprint)
    TRM.load_trm_checkpoint = lambda tag, device, dtype=torch.bfloat16: load(tag, DEVICE, dtype=dtype)
    TRM.encode_questions = lambda questions, device: encode(questions, DEVICE)
    torch.Tensor.numpy = lambda self, *a, **k: to_numpy(self.cpu(), *a, **k)

    def platform_fingerprint():
        fp = fingerprint()
        fp.update({"device": DEVICE, "gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda,
                   "cudnn": torch.backends.cudnn.version(), "float32_matmul_precision": torch.get_float32_matmul_precision(),
                   "allow_tf32": torch.backends.cuda.matmul.allow_tf32, "shims": SHIMS,
                   "runner": "ls_sae12_analysis.run_native_cuda"})
        return fp
    RN.platform_fingerprint = platform_fingerprint


if __name__ == "__main__":
    install()
    sys.argv[0] = RN.__file__
    RN.main()
