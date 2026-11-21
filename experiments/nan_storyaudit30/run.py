"""One immutable episode block, streamed without Torch, SAE assets or refitting."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from types import SimpleNamespace
from .common import ROOT,RUNS,Guard,Handle,verify,no_torch,read,write,sha,select,closed_entries
from .audit import check_episode,check_pair_shapes

OUT=Path('reports/nanbeige_bpd_gym')
OUTPUTS=(str(OUT/'NAN_STORYAUDIT30_RESULTS.md'),str(OUT/'nan_storyaudit30_results.json'))
ENDPOINTS=('format_valid','final_correct','checkpoints_correct','path_correct','joint_correct')


def roster():
    from experiments.nan_full26.common import rows
    return [dict(r,stage='nan_story23',source_id=r['id'],id='nan_story23/'+r['id'])
            for r in rows('experiments/nan_story23/fixtures/public.jsonl') if r['mode']=='native']


def aggregate(public,records):
    byid={r['source_id']:r for r in records};assert len(byid)==len(records)
    strata=defaultdict(list)
    for r in public:strata[r['split'],r['query_type'],r['event_count'],r['arm']].append(r)
    result=[]
    for key,expected in sorted(strata.items()):
        observed=[byid[r['id']] for r in expected if r['id'] in byid]
        roots=defaultdict(list)
        for r in expected:roots[r['root']].append(byid.get(r['id']))
        assert all(len(v)==2 for v in roots.values())
        complete=[v for v in roots.values() if all(x is not None for x in v)]
        item=dict(split=key[0],query_type=key[1],event_count=key[2],arm=key[3],
            registered_worlds=len(expected),observed_worlds=len(observed),missing_worlds=len(expected)-len(observed),
            registered_roots=len(roots),complete_world_pairs=len(complete),endpoints={})
        for endpoint in ENDPOINTS:
            correct=sum(r['grade'][endpoint] for r in observed)
            both=sum(all(r['grade'][endpoint] for r in pair) for pair in complete)
            possible=sum(any(r is None for r in pair) and all(r is None or r['grade'][endpoint] for r in pair) for pair in roots.values())
            item['endpoints'][endpoint]=dict(correct_worlds=correct if observed else None,
                whole_roster_correct_bounds=[correct,correct+len(expected)-len(observed)],
                both_world_roots=both if complete else None,whole_roster_both_world_bounds=[both,both+possible])
        result.append(item)
    return result


def presentation(p):
    OUT.mkdir(exist_ok=True)
    Path(OUTPUTS[1]).write_text(json.dumps(p,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
    lines=['# RAM28 early storyworld behavior and state audit','',
        f"Verified **{p['audited_trajectories']}/72** registered NF4 story trajectories and **{p['physical_vectors_checked']}** physical vectors at **{p['captured_prefixes']}** generated prefixes. This is a separate early cache audit; it does not complete the full RAM28 gym or its later SAE/BPD audit.",'',
        'The native story block contains 12 source roots (3 CAL, 9 DEV), two counterfactual worlds, and main/copy/erased arms. Actual, first-order belief and nested recorded-belief stories retain all event-count strata. The original anchored 512-character packet grammar and final/checkpoint/path/joint endpoints are unchanged. No substring answer rescue, model replay or missing-trajectory imputation. Raw text/IDs/states/individual grades remain local.','',
        'All selected trajectories are native baseline rows. This audit makes no learned-route, SAE, BPD intervention, cross-precision geometry or faithful-reasoning claim. BF16 remains deferred under the host-RAM policy. The full frozen 1,549-baseline / 216 real A/B-pair scope and canonical post-GPU host analysis remain unchanged. Final target verification is pending unless a later canonical receipt supplies it.','',
        '## Every registered endpoint and coverage bound','',
        '| Split | Query | Events | Arm | Endpoint | Correct / observed | Missing | Whole-roster world bounds | Both / complete roots | Whole-roster root bounds |',
        '|---|---|---:|---|---|---:|---:|---|---:|---|']
    for g in p['groups']:
        for endpoint,v in g['endpoints'].items():
            lines.append(f"| {g['split']} | {g['query_type']} | {g['event_count']} | {g['arm']} | {endpoint} | {v['correct_worlds'] if v['correct_worlds'] is not None else '—'}/{g['observed_worlds']} | {g['missing_worlds']} | {v['whole_roster_correct_bounds']}/{g['registered_worlds']} | {v['both_world_roots'] if v['both_world_roots'] is not None else '—'}/{g['complete_world_pairs']} | {v['whole_roster_both_world_bounds']}/{g['registered_roots']} |")
    memory=p['audit_resources']
    lines+=['','## Integrity, resources and interpretation','',
        'The audit checks the frozen source oracle, pinned initial token IDs, exact generated-token decoding and stopping order, prefix/history hashes, all closed-record/archive hashes, all 40 uint16 BF16 vectors and native-output/boundary shapes per prefix, finite values, native local-write identity, and paired batch/padding bindings. Saved top-token score consistency is checked; full logits are not saved, so no independent argmax reconstruction is claimed. Only complete append-only journal entries are consumed; unclosed prefixes are never promoted into episodes.','',
        f"Maximum sampled private audit RAM: **{memory['maximum_sampled_private_bytes']/2**20:.1f} MiB**. Minimum sampled system available RAM: **{memory['minimum_sampled_available_physical_bytes']/2**30:.3f} GiB**. One numerical thread, Below Normal, no Torch import/model/CUDA/forward, 0.75 GiB private cap and 4 GiB system reserve (5 GiB admission). Samples every two seconds are not continuous peak guarantees.",'',
        'Roots are the evidence units; worlds, arms, generated tokens, checkpoints and path steps repeat within roots. These are exposed-development descriptions on one checkpoint. Output traces are not proof of internal reasoning. Effect/sham floors and power remain uncalibrated; zero formal comparisons and **REDESIGN_OR_PILOT_ONLY**. No original failure or gate changes, and no donor transfer. The original 168-case test stays sealed; the continuous goal remains active.','',
        f"RAM28 seal: `{p['RAM28_source_manifest_sha256']}`. Early-audit seal: `{p['source_manifest_sha256']}`.",'',
        '[Protocol](../../experiments/nan_storyaudit30/PROTOCOL.md) · [Every group and provenance](nan_storyaudit30_results.json) · [Full RAM28 condition](NAN_RAM28_PROGRESS.md)','']
    Path(OUTPUTS[0]).write_text('\n'.join(lines),encoding='utf-8',newline='\n')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--artifact-root',required=True);parser.add_argument('--snapshot',required=True)
    parser.add_argument('--model-pid',type=int);parser.add_argument('--model-creation',type=int);a=parser.parse_args()
    model=Handle(a.model_pid,a.model_creation) if a.model_pid else None
    with Guard(model) as guard:
        seal=verify();cfg=read(ROOT/'registration.json');public=roster();base=Path(a.artifact_root)/'nf4'
        assert len(public)==72 and [r['id'] for r in public]==cfg['ordered_source_ids']
        assert read(base/'manifest.json')['source_manifest_sha256']==seal['RAM28_source_manifest_sha256']
        preflight=read(base/'preflight.json');assert preflight['status']=='passed'
        entries=select(closed_entries(base),public);write(RUNS/'selected_closed_entries.json',entries)
        from importlib.metadata import version
        for name,expected in cfg['packages'].items():assert version(name)==expected
        from experiments.nan_story23.source_audit import validate
        from experiments.nan_full26.common import rows
        validate(rows('experiments/nan_story23/fixtures/public.jsonl'),rows('experiments/nan_story23/fixtures/host_labels.jsonl'))
        gold={'nan_story23/'+r['id']:r for r in rows('experiments/nan_story23/fixtures/host_labels.jsonl')}
        from transformers import PreTrainedTokenizerFast
        from experiments.nan_gym05.common import prompt_ids
        tok=PreTrainedTokenizerFast(tokenizer_file=str(Path(a.snapshot)/'tokenizer.json'));wrapped=SimpleNamespace(tok=tok)
        tokens=read('experiments/nan_full26/tokenizer_receipt.json')
        for name,digest in tokens['tokenizer_files_sha256'].items():assert sha(Path(a.snapshot)/name)==digest
        records=[]
        for entry in entries:
            r=public[entry['slot']];initial=prompt_ids(wrapped,a.snapshot,r['prompt'])
            records.append(check_episode(base,entry,r,gold[r['id']],tok,initial,tokens['contexts'][r['id']]))
            guard.check()
        check_pair_shapes(records);no_torch()
        write(RUNS/'private_episode_audits.json',records)
        payload=dict(experiment='NAN-STORYAUDIT-30',status='complete_story_block' if len(records)==72 else 'partial_closed_story_block',
            audited_trajectories=len(records),registered_trajectories=72,missing_trajectories=72-len(records),
            captured_prefixes=sum(r['prefixes'] for r in records),physical_vectors_checked=sum(r['physical_vectors'] for r in records),
            groups=aggregate(public,records),source_manifest_sha256=sha(ROOT/'SOURCE_MANIFEST.json'),
            RAM28_source_manifest_sha256=seal['RAM28_source_manifest_sha256'],selection_snapshot_sha256=sha(RUNS/'selected_closed_entries.json'),
            private_episode_audit_sha256=sha(RUNS/'private_episode_audits.json'),
            BF16_status='deferred_host_RAM_policy',final_target_verification='pending_canonical_RAM28_receipt',
            SAE_and_BPD_analysis='pending_canonical_serialized_host',new_pretrained_forwards=0,Torch_imported=False,
            formal_comparisons=0,resolution_verdict='REDESIGN_OR_PILOT_ONLY',scientific_causal_floor=None,matched_sham_floor=None,
            changes_original_gates=False,sealed_test_accessed=False,continuous_goal_remains_active=True)
    resource=read(RUNS/'resources.json');assert resource['status']=='passed';samples=resource['samples']
    payload['audit_resources']=dict(resources_sha256=sha(RUNS/'resources.json'),sample_count=len(samples),
        maximum_sampled_private_bytes=max(v['private_bytes'] for v in samples),
        minimum_sampled_available_physical_bytes=min(v['available_physical_bytes'] for v in samples))
    presentation(payload)
    write(RUNS/'completion.json',dict(status=payload['status'],outputs_sha256={name:sha(name) for name in OUTPUTS},new_pretrained_forwards=0))
    print(json.dumps({k:payload[k] for k in ('status','audited_trajectories','missing_trajectories','captured_prefixes','physical_vectors_checked','audit_resources')}),flush=True)


if __name__=='__main__':main()
