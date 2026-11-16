"""LS-SAE04 calibration-only job; no test puzzle generation or inference."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import torch
from mechinterp.loopscope02.scripts import trm_sudoku as TS
from mechinterp.sae_causal_transfer.native_trm_v1 import families as OLD
from mechinterp.sae_causal_transfer.native_trm_v1 import fit as FIT
from mechinterp.sae_causal_transfer.native_jvp_v2 import families as FAM
from mechinterp.sae_causal_transfer.native_jvp_v2 import fit_pack as PACK


def freeze(out_dir):
    torch.set_num_threads(4)
    torch.manual_seed(71)
    out_dir=Path(out_dir)
    rows=FAM.roster("fit_30")
    roster_sha=OLD.fingerprint(rows)
    model,checkpoint_sha=TS.load_trm_checkpoint("gaoxin492","cpu",dtype=torch.float32)
    if checkpoint_sha!=PACK.MODEL_SHA:
        raise ValueError("unregistered original model SHA")
    model.eval().requires_grad_(False)
    states,_=FIT.stage_one(model,rows)
    fit=FIT.fit_only(states)
    manifest=PACK.make_pack(fit,roster_sha,checkpoint_sha,
                            out_dir/"frozen_calibration.npz",
                            out_dir/"frozen_calibration_manifest.json")
    print("LS_SAE04_CALIBRATION_ONLY_SEALED",json.dumps(manifest,sort_keys=True),flush=True)
    return manifest


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out-dir",type=Path,required=True)
    args=p.parse_args()
    freeze(args.out_dir)


if __name__=="__main__":
    main()
