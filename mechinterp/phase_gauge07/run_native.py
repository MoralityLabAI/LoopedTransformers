"""COE-07: bounded native pretrained Attention-TRM activation extraction.

Run from LoopedTransformers checkout. Downloads the pinned 7M gaoxin492 TRM
through the repository's existing loader, which verifies the complete SHA256
before deserialization. Does not update the model or fit it to held-out data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys

import numpy as np
import torch

from .score import (FIT_STRATA, HOLDOUT_STRATA, MODEL_SHA256, STRATA,
                    VISITS, evaluate_heads, select_manifest)
from mechinterp.loopscope02.scripts import trm_sudoku as TS

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / 'mechinterp/loopscope02/results/benchmark_bundle.json'
REPS = ('visible', 'dark', 'full', 'scratch')


def validate_projection(model, zh, exact_lens):
    """Apply the same full 11-row-space decomposition as LoopScope-03."""
    w = model.lm_head.weight.float().detach()
    q,s,vh = torch.linalg.svd(w, full_matrices=False)
    rank = int((s>1e-8*s[0]).sum().item())
    v = vh[:rank].T.contiguous()
    dark = zh - ((zh @ v) @ v.T)
    parity = (dark @ w.T).abs().max().item()
    if parity > 2e-3:
        raise AssertionError(f'dark projector failed lens parity: {parity}')
    return dark, rank, parity


@torch.inference_mode()
def capture(model, manifest: dict) -> tuple[dict[str,np.ndarray],dict]:
    """Return model latent values indexed by strata, digit and outer visit."""
    raw={r:[] for r in REPS}
    correct={}
    projection_rank=None
    projection_max_err=0.0
    for group in STRATA:
        rows=manifest['strata'][group]
        q=TS.encode_questions([r['question'] for r in rows],device='cpu')
        emb=model.encode_inputs(q)
        zh,zl=model.init_states(len(rows))
        digit_idx=torch.arange(len(rows))
        target=torch.tensor([r['target_idx'] for r in rows],dtype=torch.long)+model.puzzle_emb_len
        rep_step={r:[] for r in REPS}
        accuracy=[]
        for step in range(VISITS):
            zh,zl=model.act_step(zh,zl,emb)
            part_h=zh[digit_idx,target,:].float()
            part_l=zl[digit_idx,target,:].float()
            logits=model.lm_head(part_h).float()
            dark,rank,parity=validate_projection(model,part_h,logits)
            if projection_rank is None: projection_rank=rank
            if rank!=projection_rank: raise AssertionError('rank changed')
            projection_max_err=max(projection_max_err,parity)
            rep_step['visible'].append(logits.cpu().numpy())
            rep_step['dark'].append(dark.cpu().numpy())
            rep_step['full'].append(part_h.cpu().numpy())
            rep_step['scratch'].append(part_l.cpu().numpy())
            predictions=logits[:,2:11].argmax(dim=-1)+1
            truth=torch.tensor([r['digit'] for r in rows],device=predictions.device)
            accuracy.append(int((predictions==truth).sum().item()))
        for rep in REPS:
            raw[rep].append(np.stack(rep_step[rep],axis=1)) # [9,T,D]
        correct[group]=accuracy
    arrays={rep:np.stack(raw[rep]) for rep in REPS} # [4,9,T,D]
    return arrays,{'per_stratum_correct_of_9_by_visit':correct,
                   'projector_rank':projection_rank,'max_removed_lens_logit':projection_max_err,
                   'dimensions':{k:int(v.shape[-1]) for k,v in arrays.items()}}


def evaluate_source(arrays: dict[str,np.ndarray], manifest:dict, checks:dict, model_sha:str):
    if model_sha!=MODEL_SHA256:
        raise AssertionError('checkpoint hash mismatch')
    out={}
    for rep in REPS:
        fit=arrays[rep][:2].astype(np.float64)
        test=arrays[rep][2:].astype(np.float64)
        selected=evaluate_heads(fit,test)
        # Keep full selected head vectors so independent replays can recompute.
        out[rep]=selected
    return {
        'schema':'COE-07-native-pretrained-TRM-phase-gauge-v1',
        'source_model':'gaoxin492/TinyRecursiveModels-Sudoku-Extreme-att:step_21700',
        'source_checkpoint_sha256':model_sha,
        'source_benchmark_sha256':manifest['manifest_sha256'],
        'fit_strata':list(FIT_STRATA),'heldout_strata':list(HOLDOUT_STRATA),
        'num_digits':9,'visits':VISITS,'pairs_per_stratum':36,
        'metadata':checks,
        'representations':out,
        'summary':{
            rep:{head:{metric:int(selected[head]['heldout_scores']['pooled'][metric])
                       for metric in ('ordered','cyclic','count','final','order_with_unknown_clock_loss','recoverable_order_only','denominator')}
                 for head in ('ordered','cyclic','cyclic_gain','count')}
            for rep,selected in out.items()},
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=Path('mechinterp/phase_gauge07/output'))
    parser.add_argument('--threads',type=int,default=4)
    p=parser.parse_args()
    if p.threads <1 or p.threads>16:raise ValueError('threads must be 1..16')
    torch.set_num_threads(p.threads)
    p.out.mkdir(parents=True,exist_ok=True)
    bundle=json.loads(BENCHMARK.read_text(encoding='utf8'))
    manifest=select_manifest(bundle)
    print('COE-07 manifest',manifest['manifest_sha256'],flush=True)
    (p.out/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    # Runs the existing SHA-pinned checkpoint loader, not a synthetic imitation.
    model,checkpoint_sha=TS.load_trm_checkpoint('gaoxin492','cpu',dtype=torch.float32)
    if checkpoint_sha!=MODEL_SHA256:raise ValueError('hash mismatch')
    print('COE-07 verified native model SHA256',checkpoint_sha,flush=True)
    arrays,checks=capture(model,manifest)
    checks['source_runner_sha256']=hashlib.sha256(Path(TS.__file__).read_bytes()).hexdigest()
    checks['torch_version']=str(torch.__version__)
    checks['numpy_version']=str(np.__version__)
    checks['platform']=platform.platform()
    checks['cpu_threads']=p.threads
    checks['activation_sha256']={rep:hashlib.sha256(np.ascontiguousarray(arrays[rep]).tobytes()).hexdigest()
                                 for rep in REPS}
    del model
    # Export native data with SHA-provenance for future independent replay; CI uploads this artifact.
    np.savez_compressed(p.out/'native_activations.npz',**arrays)
    result=evaluate_source(arrays,manifest,checks,checkpoint_sha)
    (p.out/'receipt.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'schema':result['schema'], 'model_sha256':checkpoint_sha,
                      'manifest_sha256':manifest['manifest_sha256'],
                      'native_accuracy':checks['per_stratum_correct_of_9_by_visit'],
                      'summary':result['summary']},sort_keys=True),flush=True)
    return 0

if __name__=='__main__':
    raise SystemExit(main())
