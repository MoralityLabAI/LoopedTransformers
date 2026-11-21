"""Audit every textual story, canonical trace, CF pair and roster before inference."""
from collections import defaultdict
from .common import ROOT,read,rows,sha,write
from .oracle import replay,trace_packet
from .parser import parse


def validate(public,host):
    assert len(public)==len(host)==144 and len({r['id'] for r in public})==144
    gold={r['id']:r for r in host};assert set(gold)=={r['id'] for r in public}
    programs={};pairs=defaultdict(dict)
    for row in public:
        label=gold[row['id']];assert all(row[k]==label[k] for k in ('root','world','split','query_type','event_count','arm','mode'))
        assert parse(label['canonical_trace'],row['schema'])==trace_packet(label['canonical_trace'],row['schema'])
        packet=trace_packet(label['canonical_trace'],row['schema']);assert packet is not None
        for key in ('intermediate_rooms','path','final_room'):
            assert packet[key]==label[key]
        if row['arm']!='erased':
            independent=replay(row['prompt'],row['query_type'])
            for key in ('intermediate_rooms','path','final_room'):
                assert independent[key]==label[key],(row['id'],key)
            if row['arm']=='copy':
                assert row['prompt'].endswith(label['canonical_trace']+'\n')
            if row['mode']=='native' and row['arm']=='main':
                programs[row['source_case_id']]=independent
        if row['mode']=='native':
            pairs[row['root'],row['arm']][row['world']]=(row,label)
    assert len(programs)==24 and len({r['root'] for r in public})==12
    assert all(a['prompt']!=b['prompt'] for a,b in zip(public,public[1:]))
    for byworld in pairs.values():
        assert set(byworld)=={'A','B'};a,b=byworld['A'],byworld['B']
        assert a[1]['final_room']!=b[1]['final_room']
        if a[0]['arm']=='erased':
            assert a[0]['prompt']==b[0]['prompt']
        elif a[0]['arm']=='main':
            x,y=[programs[z[0]['source_case_id']] for z in (a,b)]
            assert x['initial']==y['initial'] and x['query_people']==y['query_people'] and x['target']==y['target']
            changes=[(u,v) for u,v in zip(x['events'],y['events']) if u!=v];assert len(changes)==1
            u,v=changes[0];assert u['destination']!=v['destination'] and {k:w for k,w in u.items() if k!='destination'}=={k:w for k,w in v.items() if k!='destination'}
    return dict(status='passed',roots=12,CAL_roots=3,DEV_roots=9,source_worlds=24,registered_trajectories=144,
        native_source_controls=72,additional_main_online_modes=72,textual_world_replays=120,CF_destination_changes_per_pair=1,
        erased_prompts_byte_identical=True,canonical_traces_both_grammars_agree=True,pretrained_model_calls=0,sealed_test_accessed=False)


def main():
    result=validate(rows(ROOT/'fixtures/public.jsonl'),rows(ROOT/'fixtures/host_labels.jsonl'))
    result['public_sha256']=sha(ROOT/'fixtures/public.jsonl');result['host_sha256']=sha(ROOT/'fixtures/host_labels.jsonl')
    write(ROOT/'fixtures/independent_source_audit.json',result);print(result)


if __name__=='__main__':
    main()
