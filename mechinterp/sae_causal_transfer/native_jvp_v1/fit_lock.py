"""Cross-run LS-SAE03 frozen calibration weight handoff.

The exact previously registered LS-SAE02 PCA/SAE arrays are serialized only
after the 32-fit-puzzle calibration fingerprints match their original SHA256.
A separate CI job can load and verify the original arrays byte-identically.
"""
from __future__ import annotations
import hashlib
from pathlib import Path

import numpy as np
import torch

from mechinterp.sae_causal_transfer.adapter import (
    fingerprint_arrays,to_physical_dictionary,
)
from mechinterp.sae_causal_transfer.native_trm_v1 import fit as FIT

SCHEMA="LS-SAE03/exact-frozen-LS-SAE02-calibration-v1"
FIT_SHA="f217453dc17e078b055e99f3cebc3cf736af7e7cfae2ab120353c91bf2554c9d"
PHYSICAL_SHA="decf90fd73137bd232ba40104cf387e894c4a3d1f4f007036c591d173f3e238c"
KEYS={"schema","mean512","basis","mu","scale",
      "encoder_weight","encoder_bias","decoder_weight","decoder_bias",
      "training_final_loss"}


def checked_fingerprint(fit):
    if fit.fingerprint!=FIT_SHA:
        raise ValueError("full source LS-SAE02 calibration fit fingerprint mismatched")
    physical=fingerprint_arrays(fit.physical_rows,fit.physical_bias,
                                fit.physical_decoder)
    if physical!=PHYSICAL_SHA:
        raise ValueError("LS-SAE02 physical dictionary fingerprint mismatched")
    return physical


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_exact(fit,path):
    checked_fingerprint(fit)
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    model=fit.sae.eval()
    data={
     "schema":np.asarray(SCHEMA),
     "mean512":fit.mean512,
     "basis":fit.basis,
     "mu":fit.mu,
     "scale":fit.scale,
     "encoder_weight":model.encoder.weight.detach().cpu().numpy().copy(),
     "encoder_bias":model.encoder.bias.detach().cpu().numpy().copy(),
     "decoder_weight":model.decoder.weight.detach().cpu().numpy().copy(),
     "decoder_bias":model.decoder.bias.detach().cpu().numpy().copy(),
     "training_final_loss":np.asarray(fit.final_loss,dtype=np.float64)
    }
    np.savez_compressed(path,**data)
    return {"file_sha256":sha_file(path),
            "fit_sha256":fit.fingerprint,"physical_sha256":PHYSICAL_SHA,
            "file_bytes":path.stat().st_size}


def load_exact(path):
    path=Path(path)
    if not path.is_file():
        raise FileNotFoundError("full job requires archived pilot-frozen PCA+SAE")
    with np.load(path,allow_pickle=False) as loaded:
        if set(loaded.files)!=KEYS or str(loaded["schema"])!=SCHEMA:
            raise ValueError("unrecognized source calibration serialization")
        arrays={k:loaded[k].copy() for k in KEYS if k!="schema"}
    if (arrays["mean512"].shape!=(512,) or arrays["basis"].shape!=(512,48)
        or arrays["mu"].shape!=(48,) or arrays["scale"].shape!=(48,)
        or arrays["encoder_weight"].shape!=(96,48)
        or arrays["encoder_bias"].shape!=(96,)
        or arrays["decoder_weight"].shape!=(48,96)
        or arrays["decoder_bias"].shape!=(48,)):
        raise ValueError("altered frozen SAE/PCA array shapes")
    if not all(np.all(np.isfinite(a)) for a in arrays.values()):
        raise ValueError("nonfinite source-fit array in archive")
    sae=FIT.SparseTopFour(48,96)
    with torch.no_grad():
        sae.encoder.weight.copy_(torch.tensor(arrays["encoder_weight"],dtype=torch.float32))
        sae.encoder.bias.copy_(torch.tensor(arrays["encoder_bias"],dtype=torch.float32))
        sae.decoder.weight.copy_(torch.tensor(arrays["decoder_weight"],dtype=torch.float32))
        sae.decoder.bias.copy_(torch.tensor(arrays["decoder_bias"],dtype=torch.float32))
    sae.eval()
    E,b,D=to_physical_dictionary(arrays["encoder_weight"],arrays["encoder_bias"],
         arrays["decoder_weight"],arrays["mu"],arrays["scale"])
    fp=fingerprint_arrays(
        arrays["mean512"],arrays["basis"],arrays["mu"],arrays["scale"],
        arrays["encoder_weight"],arrays["encoder_bias"],arrays["decoder_weight"])
    fit=FIT.Fitted(arrays["mean512"],arrays["basis"],arrays["mu"],
                   arrays["scale"],E,b,D,sae,fp,
                   float(arrays["training_final_loss"]))
    checked_fingerprint(fit)
    return fit,{"file_sha256":sha_file(path),
                "fit_sha256":fp,"physical_sha256":PHYSICAL_SHA,
                "file_bytes":path.stat().st_size}
