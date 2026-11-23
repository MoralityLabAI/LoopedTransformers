"""Read-only PPT Plan A/B replay-state probe -> LBF predecision traces.

This re-forwards a pinned checkpoint on stored PUBLIC inputs from native-order
or live-Lean episode ledgers. It does not replay Lean, change parameters,
perform interventions in the network, or certify any proof.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

from lbf_gym import (SCHEMA_OBS, canonical, digest, file_sha, fresh_dir,
                     read_jsonl, validate_observation, write_jsonl)

ROLE_INDEX = ("goal", "context", "lemma", "plan", "value", "message", "halt", "delegate")


def squash(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError("nonfinite latent scalar")
    return max(0.0, min(1.0, value / (1.0 + value)))


def bridge(checkpoint: Path, episodes: Path, out: Path, max_episodes: int = 8,
           depth: int = 3) -> dict:
    if not 1 <= depth <= 16 or not 1 <= max_episodes <= 128:
        raise ValueError("depth 1..16, episodes 1..128")
    if not checkpoint.is_file() or not episodes.is_file():
        raise ValueError("checkpoint and episodes file required")
    # This module lives next to lean_cgrt_hrm400 in this repository.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'lean_cgrt_hrm400'))
    import torch  # heavy dependency is entirely optional for the pure LBF codec
    from ppt_gym.cli import checkpoint_load
    from ppt_gym.data import canonical as ppt_canonical

    model, metadata = checkpoint_load(checkpoint)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    torch.set_num_threads(1)
    checks = file_sha(checkpoint)
    trace_hash = file_sha(episodes)
    records = []
    with torch.no_grad():
        for episode in read_jsonl(episodes)[:max_episodes]:
            if 'events' not in episode or not isinstance(episode['events'], list) or not episode['events']:
                raise ValueError('expected native Lean episode events')
            cs_seen = None
            visits = []
            # A changing menu is a new tactic decision. The fresco makes a new
            # "attempt" column; the candidate union keeps IDs stable.
            all_cs = []
            for idx, event in enumerate(episode['events'], 1):
                candidates = event.get('candidate_menu')
                public = event.get('input')
                if not candidates or not isinstance(public, dict) or 'state' not in public:
                    raise ValueError('episode missing public predecision packet')
                if len(candidates) > 64:
                    raise ValueError('candidate roster above 64')
                candidate_ids = [f"a{idx}:{n}" for n in range(len(candidates))]
                all_cs.extend(candidate_ids)
                if len(all_cs) > 64:
                    raise ValueError('bridge allows at most 64 unique candidate visits')
                order = episode.get('order', 'ab')
                if order not in ('ab','ba'):
                    order = 'ab'
                outputs = model(ppt_canonical(public), candidates, depth=depth,
                                checks_remaining=min(8, max(1, int(public.get('checks_remaining', 8)))),
                                return_all=True, middle_order=order)
                for k, result in enumerate(outputs, 1):
                    state = result['state'][0]
                    if state.shape[0] != 8:
                        raise ValueError('expected exactly eight typed PPT slots')
                    tiles = []
                    for slot_idx, role in enumerate(ROLE_INDEX):
                        norm = float(state[slot_idx].float().square().mean().sqrt())
                        tiles.append({'candidate_id':'shared', 'role':role,
                                      'frame':'state' if role not in ('message','halt') else 'memory' if role=='message' else 'gate',
                                      'activation':squash(norm), 'polarity':0})
                    for ci, candidate_id in enumerate(candidate_ids):
                        probability = float(torch.sigmoid(result['logits'][ci]))
                        tiles.append({'candidate_id':candidate_id, 'role':'value',
                                      'frame':'action','activation':probability,'polarity':0})
                    visits.append({'attempt':idx,'pass':k,'tiles':tiles})
            if len(visits) > 64:
                raise ValueError('episode exceeds 64 (attempt,pass) visits')
            root_id = str(episode['root']) + ':' + str(episode.get('order',episode.get('policy','frozen')))
            r = {'schema': SCHEMA_OBS, 'root_id':root_id,
                 'lineage':str(episode.get('lineage',episode['root'])),
                 'split':str(episode.get('split','calibration')),
                 'source':{'kind':'ppt_replay','source_sha256':trace_hash,
                           'checkpoint_sha256':checks},
                 'candidates':all_cs,'visits':visits}
            validate_observation(r)
            records.append(r)
    fresh_dir(out)
    write_jsonl(out/'observations.jsonl',records)
    manifest={'schema':'lbf.ppt_bridge.v1','source_episodes_sha256':trace_hash,
              'checkpoint_sha256':checks,'observation_sha256':file_sha(out/'observations.jsonl'),
              'n_episodes':len(records),'torch_version':torch.__version__,
              'model_plan':model.cfg.plan,'depth':depth,'lean_calls':0,
              'predecision_only':True,'labels_exported':False,'scientific_eligible':False,
              'state_is_coda_output':model.cfg.plan=='b',
              'role_names_are_nominal':True,'squash':'rms/(1+rms)'}
    (out/'manifest.json').write_text(canonical(manifest)+'\n',encoding='utf-8')
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--episodes',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--max-episodes',type=int,default=8)
    p.add_argument('--depth',type=int,default=3)
    args=p.parse_args()
    print(canonical(bridge(args.checkpoint,args.episodes,args.out,args.max_episodes,args.depth)))


if __name__=='__main__':
    main()
