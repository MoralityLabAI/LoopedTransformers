"""Post-audit metadata reconciliation; no regrading, NumPy or model access.

Requires the frozen audit's normal completion and closed private episode
summaries. It cannot rescue a failed/missing original audit or change a grade.
"""
from collections import Counter,defaultdict
from pathlib import Path
import sys
import time
from unittest.mock import patch

ENDPOINTS=('format_valid','final_correct','checkpoints_correct','path_correct','joint_correct')


def main():
    from experiments.nan_storyaudit30 import common as bounded
    from experiments.nan_ram28.common import read,write,sha
    from experiments.nan_ranksite13.resources import below_normal
    below_normal();original_root=bounded.ROOT
    from experiments.nan_ram28.resources import memory
    profile=Path('runs/nan_storyaudit30_validation_metadata');out=profile/'runtime'
    model=bounded.Handle(28152,134397440962966790)
    admission_started=time.monotonic();admission=read(profile/'registration.json')
    while memory()['available_physical_bytes']<admission['minimum_start_available_RAM_bytes']:
        assert model.alive(),'Yield if model ends before validation admission'
        assert time.monotonic()-admission_started<600,'Validation admission window expired; no reserve reduction'
        time.sleep(10)
    out.mkdir(exist_ok=False)
    with patch.object(bounded,'ROOT',profile),patch.object(bounded,'RUNS',out),bounded.Guard(model):
        with patch.object(bounded,'ROOT',original_root):seal=bounded.verify()
        base=Path('runs/nan_storyaudit30');completion=read(base/'completion.json')
        assert completion['status']=='complete_story_block' and completion['new_pretrained_forwards']==0
        for name,value in completion['outputs_sha256'].items():assert sha(name)==value,name
        packet=read('reports/nanbeige_bpd_gym/nan_storyaudit30_results.json')
        assert packet['status']=='complete_story_block' and packet['audited_trajectories']==72 and packet['missing_trajectories']==0
        assert packet['source_manifest_sha256']==sha(original_root/'SOURCE_MANIFEST.json')
        assert packet['RAM28_source_manifest_sha256']==seal['RAM28_source_manifest_sha256']
        assert sha(base/'selected_closed_entries.json')==packet['selection_snapshot_sha256']
        assert sha(base/'private_episode_audits.json')==packet['private_episode_audit_sha256']
        source=[row for line in Path('experiments/nan_story23/fixtures/public.jsonl').read_text(encoding='utf-8').splitlines() if line.strip() for row in [__import__('json').loads(line)] if row['mode']=='native']
        byid={'nan_story23/'+row['id']:row for row in source}
        assert len(source)==len(byid)==72
        assert list(byid)==read(original_root/'registration.json')['ordered_source_ids']
        records=read(base/'private_episode_audits.json');observed={row['source_id']:row for row in records}
        assert len(records)==len(observed)==72 and set(observed)==set(byid)
        strata=defaultdict(list);root_splits=defaultdict(set)
        for case_id,row in byid.items():
            case=observed[case_id]
            for key in ('root','world','split','arm','query_type','event_count'):assert case[key]==row[key],key
            grade=case['grade'];assert set(grade)==set(ENDPOINTS) and all(type(v) is bool for v in grade.values())
            assert grade['joint_correct']==(grade['final_correct'] and grade['checkpoints_correct'] and grade['path_correct'])
            assert grade['format_valid'] or not any(grade.values())
            assert case['physical_vectors']==40*case['prefixes']
            strata[row['split'],row['query_type'],row['event_count'],row['arm']].append(case)
            root_splits[row['split']].add(row['root'])
        assert len(set.union(*root_splits.values()))==12 and len(root_splits['calibration'])==3 and len(root_splits['development'])==9
        assert packet['captured_prefixes']==sum(row['prefixes'] for row in records)==3990
        assert packet['physical_vectors_checked']==sum(row['physical_vectors'] for row in records)==159600
        assert len(packet['groups'])==len(strata)==36
        checked_groups=set();checked_endpoint_cells=0
        for group in packet['groups']:
            key=tuple(group[name] for name in ('split','query_type','event_count','arm'));assert key not in checked_groups
            checked_groups.add(key);expected=strata[key];roots=defaultdict(list)
            for row in expected:roots[row['root']].append(row)
            assert all(len(values)==2 and {row['world'] for row in values}=={'A','B'} for values in roots.values())
            assert group['registered_worlds']==group['observed_worlds']==len(expected) and group['missing_worlds']==0
            assert group['registered_roots']==group['complete_world_pairs']==len(roots)
            assert set(group['endpoints'])==set(ENDPOINTS)
            for endpoint in ENDPOINTS:
                values=group['endpoints'][endpoint]
                correct=sum(row['grade'][endpoint] for row in expected)
                both=sum(all(row['grade'][endpoint] for row in pair) for pair in roots.values())
                assert values['correct_worlds']==correct and values['whole_roster_correct_bounds']==[correct,correct]
                assert values['both_world_roots']==both and values['whole_roster_both_world_bounds']==[both,both]
                checked_endpoint_cells+=1
        resource=read(base/'resources.json');cfg=read(original_root/'registration.json');samples=resource['samples']
        assert resource['status']=='passed' and not resource['Torch_imported']
        assert sha(base/'resources.json')==packet['audit_resources']['resources_sha256']
        assert samples[0]['available_physical_bytes']>=cfg['minimum_start_available_RAM_bytes']
        assert all(v['private_bytes']<=cfg['maximum_private_bytes'] and v['available_physical_bytes']>=cfg['minimum_available_RAM_bytes'] for v in samples)
        assert max(v['private_bytes'] for v in samples)==packet['audit_resources']['maximum_sampled_private_bytes']
        assert min(v['available_physical_bytes'] for v in samples)==packet['audit_resources']['minimum_sampled_available_physical_bytes']
        assert packet['final_target_verification']=='pending_canonical_RAM28_receipt'
        assert packet['BF16_status']=='deferred_host_RAM_policy' and packet['formal_comparisons']==0 and not packet['changes_original_gates']
        assert packet['new_pretrained_forwards']==0 and not packet['sealed_test_accessed'] and not packet['Torch_imported']
        summary=[]
        for split in ('calibration','development'):
            for arm in ('main','copy','erased'):
                selected=[row for row in records if row['split']==split and row['arm']==arm]
                roots=defaultdict(list)
                for row in selected:roots[row['root']].append(row)
                summary.append(dict(split=split,arm=arm,worlds=len(selected),roots=len(roots),
                    correct_worlds={name:sum(row['grade'][name] for row in selected) for name in ENDPOINTS},
                    both_world_roots={name:sum(all(row['grade'][name] for row in values) for values in roots.values()) for name in ENDPOINTS}))
        assert 'torch' not in sys.modules and 'numpy' not in sys.modules
        payload=dict(status='passed',source_manifest_sha256=packet['source_manifest_sha256'],
            report_sha256=sha('reports/nanbeige_bpd_gym/nan_storyaudit30_results.json'),
            validator_sha256=sha(Path(__file__)),completion_sha256=sha(base/'completion.json'),
            audit_resources_sha256=sha(base/'resources.json'),groups=36,endpoint_cells=checked_endpoint_cells,
            all_registered_world_and_root_coverage_bounds_exact=True,trajectories=72,prefixes=3990,physical_vectors=159600,
            root_counts={key:len(value) for key,value in root_splits.items()},summary=summary,
            validation_kind='post-audit metadata consistency; original frozen grades unchanged',
            prior_admission_refusal_log_sha256=sha('runs/NAN_STORYAUDIT30_VALIDATION_20261122.log'),
            previous_metadata_wait_retirement_sha256=sha('runs/nan_storyaudit30_validation/metadata_wait_retirement.json'),
            new_model_calls=0,neural_or_grade_replays=0,Torch_imported=False,NumPy_imported=False,
            final_model_target_verification_still_pending=True,matched_BF16_and_SAE_BPD_results_still_pending=True,
            formal_comparisons=0,sealed_test_accessed=False)
    validation_resources=read(out/'resources.json');assert validation_resources['status']=='passed'
    payload['validation_resources_sha256']=sha(out/'resources.json')
    payload['validation_resource_summary']=dict(maximum_sampled_private_bytes=max(v['private_bytes'] for v in validation_resources['samples']),
        minimum_sampled_available_bytes=min(v['available_physical_bytes'] for v in validation_resources['samples']),
        resource_lease=admission)
    write('reports/nanbeige_bpd_gym/nan_storyaudit30_validation.json',payload)
    print({key:payload[key] for key in ('status','groups','endpoint_cells','root_counts','summary','validation_resource_summary')},flush=True)


if __name__=='__main__':main()
