"""Independent COE-07 replay from the frozen native activation artifact."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .run_native import REPS, evaluate_source
from .score import VISITS,MODEL_SHA256,STRATA


def replay(path:Path, receipt:dict):
    if receipt['schema']!='COE-07-native-pretrained-TRM-phase-gauge-v1':
        raise ValueError('source receipt schema mismatch')
    if receipt['source_checkpoint_sha256']!=MODEL_SHA256:raise ValueError('wrong model')
    with np.load(path,allow_pickle=False) as z:
        arrays={r:z[r] for r in REPS}
    for rep in REPS:
        if arrays[rep].shape[:3]!=(4,9,VISITS):raise ValueError('source array shape wrong')
        sha=hashlib.sha256(np.ascontiguousarray(arrays[rep]).tobytes()).hexdigest()
        if sha!=receipt['metadata']['activation_sha256'][rep]:
            raise ValueError('native source artifact tampered')
    # Replay scoring exactly from frozen selected readouts. Selection itself may
    # independently be rerun from the frozen fit states using evaluate_source.
    dummy={'manifest_sha256':receipt['source_benchmark_sha256']}
    regenerated=evaluate_source(arrays,dummy,receipt['metadata'],MODEL_SHA256)
    for rep in REPS:
        for head in regenerated['representations'][rep]:
            fresh=regenerated['representations'][rep][head]
            old=receipt['representations'][rep][head]
            if fresh['heldout_codes']!=old['heldout_codes'] or fresh['heldout_scores']!=old['heldout_scores']:
                raise ValueError(f'head replay mismatch {rep}/{head}')
            if not np.allclose(fresh['direction'],old['direction'],rtol=0,atol=0):
                raise ValueError(f'head-selection mismatch {rep}/{head}')
    return {'schema':'COE-07-independent-replay-v1','source_checkpoint_sha256':MODEL_SHA256,
            'replayed_representations':list(REPS),'status':'pass'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True)
    p.add_argument('--receipt',type=Path,required=True)
    a=p.parse_args(); print(json.dumps(replay(a.input,json.loads(a.receipt.read_text())),sort_keys=True))
if __name__=='__main__':main()
