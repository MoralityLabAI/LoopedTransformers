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
           depth: int = 3, session_manifest: Path | None = None,
           replay_receipt: Path | None = None) -> dict:
    if not 1 <= depth <= 16 or not 1 <= max_episodes <= 128:
        raise ValueError("depth 1..16, episodes 1..128")
    if not checkpoint.is_file() or not episodes.is_file():
        raise ValueError("checkpoint and episodes file required")
    # This module lives next to lean_cgrt_hrm400 in this repository.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'lean_cgrt_hrm400'))
    import torch  # heavy dependency is entirely optional for the pure LBF codec
    from ppt_gym.cli import checkpoint_load
    from ppt_gym.data import canonical as ppt_canonical
    try:
        from ppt_gym.typed_state import encode_public_packet, SCHEMA as TYPED_STATE_SCHEMA
    except ModuleNotFoundError as exc:
        if exc.name != 'ppt_gym.typed_state':
            raise
        # The historical native-order branch predates LSA-02C typed-state
        # encoding. Its raw view feeds the canonical predecision JSON directly.
        TYPED_STATE_SCHEMA = None
        def encode_public_packet(packet, candidates, *, view='raw', max_bytes=384):
            if view != 'raw':
                raise ValueError('typed view requires the LSA-02C typed_state module')
            return packet

    model, metadata = checkpoint_load(checkpoint)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    torch.set_num_threads(1)
    checks = file_sha(checkpoint)
    trace_hash = file_sha(episodes)
    view = metadata.get('input_view', 'raw')
    if view not in ('raw', 'typed', 'hybrid', 'hybrid_nopp', 'hybrid_shuffle',
                     'hybrid_rule_intact', 'hybrid_rule_rebind', 'hybrid_rule_order'):
        raise ValueError('invalid source checkpoint input view')
    if view == 'typed' and metadata.get('input_view_schema') != TYPED_STATE_SCHEMA:
        raise ValueError('typed source checkpoint schema mismatch')
    if view.startswith('hybrid'):
        from ppt_gym.lbf_hybrid import SCHEMAS as LBF_HYBRID_SCHEMAS
        if metadata.get('input_view_schema') != LBF_HYBRID_SCHEMAS[view]:
            raise ValueError('hybrid source checkpoint schema mismatch')
    provenance = {'manifest_validated': False, 'native_replay_verified': False}
    if (session_manifest is None) != (replay_receipt is None):
        raise ValueError('session manifest and native replay receipt must be supplied together')
    if session_manifest is not None:
        session = json.loads(session_manifest.read_text(encoding='utf-8'))
        replay = json.loads(replay_receipt.read_text(encoding='utf-8'))
        if session.get('schema') != 'lsa02b.native_block_order.v1':
            raise ValueError('expected LSA-02B native order manifest')
        if session.get('artifacts',{}).get('paired_episodes.jsonl') != trace_hash:
            raise ValueError('native episode bytes differ from frozen manifest')
        if session.get('checkpoint_sha256') != checks:
            raise ValueError('frozen checkpoint bytes differ from native session')
        if session.get('input_view','raw') != view:
            raise ValueError('native session view differs from checkpoint')
        if (session.get('backend',{}).get('actual_lean_process') is not True
            or session.get('status') != 'complete'
            or replay.get('replay_passed') is not True
            or replay.get('independent_replay_episodes') != 3 * session.get('paired_roots',0)):
            raise ValueError('incomplete or nonnative source/replay attestation')
        if replay.get('session_manifest_sha256') != file_sha(session_manifest):
            raise ValueError('independent replay receipt references another session manifest')
        provenance = {'manifest_validated': True, 'native_replay_verified': True,
                      'session_manifest_sha256': file_sha(session_manifest),
                      'replay_receipt_sha256': file_sha(replay_receipt)}
    records = []
    message_probes = []
    with torch.no_grad():
        # auto8 makes no model forward; its episodes are intentionally excluded.
        neural = [e for e in read_jsonl(episodes) if e.get('order', e.get('policy')) in ('ab','ba','frozen')]
        for episode in neural[:max_episodes]:
            if 'events' not in episode or not isinstance(episode['events'], list) or not episode['events']:
                raise ValueError('expected native Lean episode events')
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
                input_text = encode_public_packet(
                    ppt_canonical(public), candidates, view=view, max_bytes=model.cfg.max_bytes)
                remaining = min(8, max(1, int(public.get('checks_remaining', 8))))
                outputs = model(input_text, candidates, depth=depth,
                                checks_remaining=remaining,
                                return_all=True, middle_order=order, message_mode='normal')
                if model.cfg.plan == 'b':
                    controls = {
                        mode: model(input_text, candidates, depth=depth,
                                    checks_remaining=remaining, return_all=True,
                                    middle_order=order, message_mode=mode)
                        for mode in ('zero','roll')
                    }
                    for mode, altered in controls.items():
                        if len(altered) != len(outputs):
                            raise ValueError('message intervention changed forward-pass count')
                        for k, (baseline, changed) in enumerate(zip(outputs, altered), 1):
                            delta = (baseline['logits'] - changed['logits']).float().abs()
                            state_diff = (baseline['state'] - changed['state']).float()
                            max_logit = float(delta.max())
                            mean_state_l2 = float(state_diff.square().mean().sqrt())
                            swapped = int(baseline['logits'].argmax()) != int(changed['logits'].argmax())
                            if k == 1 and (max_logit > 1e-5 or mean_state_l2 > 1e-5):
                                raise ValueError('first-pass zero-memory equivalence violated')
                            message_probes.append({
                                'root_id': str(episode['root']) + ':' + order,
                                'attempt': idx, 'pass': k, 'message_mode': mode,
                                'max_abs_logit_delta': max_logit,
                                'mean_coda_state_l2_delta': mean_state_l2,
                                'changed_top_candidate': swapped,
                                'source_class': 'frozen_model_intervention',
                                'lean_calls': 0,
                            })
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
    write_jsonl(out/'message_probes.jsonl',message_probes)
    manifest={'schema':'lbf.ppt_bridge.v1','source_episodes_sha256':trace_hash,
              'checkpoint_sha256':checks,'observation_sha256':file_sha(out/'observations.jsonl'),
              'n_episodes':len(records),'torch_version':torch.__version__,
              'model_plan':model.cfg.plan,'input_view':view,'depth':depth,'lean_calls':0,
              'source_attestation':provenance,
              'message_probe_sha256':file_sha(out/'message_probes.jsonl'),
              'message_probe_rows':len(message_probes),
              'message_probe_modes':['zero','roll'] if model.cfg.plan=='b' else [],
              'message_probe_claim':'frozen-model output sensitivity, not proof completion',
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
    p.add_argument('--session-manifest',type=Path)
    p.add_argument('--replay-receipt',type=Path)
    args=p.parse_args()
    print(canonical(bridge(args.checkpoint,args.episodes,args.out,args.max_episodes,args.depth,
                           args.session_manifest,args.replay_receipt)))


if __name__=='__main__':
    main()
