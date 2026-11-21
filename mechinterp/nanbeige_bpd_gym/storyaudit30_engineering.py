"""MOCK_ONLY full-runner check: public oracle packets, never neural responses.

All output stays below ignored runs/. This exercises the unchanged frozen
runner with six synthetic episodes: four oracle-correct, one empty EOS and
one well-formed wrong-world packet. It is not model competence evidence;
host-gold packets for erased controls do not establish information sufficiency.
"""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
from types import SimpleNamespace


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--snapshot',required=True)
    parser.add_argument('--out-dir',required=True);args=parser.parse_args()
    out=Path(args.out_dir).resolve()
    out.relative_to(Path('runs').resolve())
    assert 'MOCK_ONLY' in out.name,'Make the non-neural scope visible in the directory name'
    out.mkdir(parents=True,exist_ok=False)
    (out/'README_MOCK_ONLY.txt').write_text('MOCK_ONLY. Synthetic state arrays and public host-oracle packets. No model inference or real generated-text inspection.\n',encoding='utf-8')
    # Rebind ONLY this child process's output namespace, never any source file.
    import experiments.nan_ctx21_resource29.common as budget
    budget.RUNS=out/'outer_resource_budget'
    import experiments.nan_storyaudit30.common as common
    import experiments.nan_storyaudit30.run as frozen
    common.RUNS=out/'audit';frozen.RUNS=common.RUNS
    frozen.OUT=out/'derived_MOCK_ONLY'
    frozen.OUTPUTS=(str(frozen.OUT/'NAN_STORYAUDIT30_RESULTS.md'),str(frozen.OUT/'nan_storyaudit30_results.json'))
    artifact=out/'artifact_MOCK_ONLY';base=artifact/'nf4';base.mkdir(parents=True)
    with budget.Guard():
        seal=common.verify()
        import numpy as np
        from transformers import PreTrainedTokenizerFast
        from experiments.nan_full26.common import rows,parse
        from experiments.nan_gym05.common import prompt_ids
        from experiments.nan_storyaudit30.audit import WIDTHS,digest,grade
        public=frozen.roster();source=public[:6]
        assert len({r['root'] for r in source})==1 and {r['world'] for r in source}=={'A','B'}
        assert {r['arm'] for r in source}=={'main','copy','erased'}
        labels={'nan_story23/'+r['id']:r for r in rows('experiments/nan_story23/fixtures/host_labels.jsonl')}
        tok=PreTrainedTokenizerFast(tokenizer_file=str(Path(args.snapshot)/'tokenizer.json'))
        wrapped=SimpleNamespace(tok=tok)
        initial=[prompt_ids(wrapped,args.snapshot,r['prompt']) for r in source]
        bindings=common.read('experiments/nan_full26/tokenizer_receipt.json')
        for name,expected in bindings['tokenizer_files_sha256'].items():assert common.sha(Path(args.snapshot)/name)==expected
        def packet_tokens(text,r):
            ids=tok.encode(text,add_special_tokens=False)
            for n in range(1,len(ids)+1):
                decoded=tok.decode(ids[:n],skip_special_tokens=False,clean_up_tokenization_spaces=False)
                if parse(decoded,r) is not None:return ids[:n],decoded
            raise AssertionError('Canonical reference did not round-trip through the fixed grammar')
        reference=[]
        for r in public:
            ids,text=packet_tokens(labels[r['id']]['canonical_trace'],r)
            assert len(ids)<=64 and grade(text,r,labels[r['id']])['joint_correct']
            reference.append((ids,text))
        generated=[reference[i][0] for i in range(6)]
        generated[1]=[tok.eos_token_id]  # Deliberately malformed, empty EOS.
        generated[4]=packet_tokens(labels[source[1]['id']]['canonical_trace'],source[4])[0]
        assert labels[source[1]['id']]['final_room']!=labels[source[4]['id']]['final_room']
        state={};index=[]
        for phase in (1,2):
            for layer in (5,11,17,21):
                for site,width in WIDTHS.items():
                    key='v'+str(len(index));state[key]=np.zeros(width,dtype=np.uint16)
                    index.append(dict(loop=phase,layer=layer,site=site,key=key))
        state.update(boundaries=np.zeros((2,3072),dtype=np.uint16),native_at_reached_input=np.zeros((2,3072),dtype=np.uint16))
        common.write(base/'manifest.json',dict(MOCK_ONLY=True,source_manifest_sha256=seal['RAM28_source_manifest_sha256']))
        common.write(base/'preflight.json',dict(MOCK_ONLY=True,status='passed',scope='Synthetic reader fixture; no CUDA preflight or model claim'))
        for ordinal in (0,2,4):
            fragments=[[],[]];stops=[None,None];texts=['','']
            for step in range(max(len(generated[ordinal]),len(generated[ordinal+1]))):
                active=[row for row in (0,1) if step<len(generated[ordinal+row])]
                lengths=[len(initial[ordinal+row])+step for row in active];padded=max(lengths)
                for batch_row,row in enumerate(active):
                    slot=ordinal+row;r=source[slot];ids=generated[slot]
                    stem=f'baseline_{ordinal:04d}_r{row}_p{step:03d}'
                    archive=base/(stem+'.npz')
                    with archive.open('xb') as stream:np.savez(stream,**state)
                    p=dict(MOCK_ONLY=True,step=step,source_id=r['id'],mode='native',token=ids[step],
                        history_sha256=digest(ids[:step]),prefix_ids_sha256=digest(initial[slot]+ids[:step]),
                        actual_length=len(initial[slot])+step,padded_length=padded,batch_size=len(active),batch_row=batch_row,
                        padding_tokens=padded*len(active)-sum(lengths),full_vocabulary_size=166144,
                        top_logit=0.,log_normalizer=1.,cuda_elapsed_ms_for_entire_batch=1.,wall_seconds_for_entire_batch=.001,
                        state_file=archive.name,state_sha256=common.sha(archive),state_index=index)
                    path=base/(stem+'.json');common.write(path,p)
                    fragments[row].append(dict(file=path.name,sha256=common.sha(path)))
                    eos=ids[step]==tok.eos_token_id
                    texts[row]=tok.decode(ids[:step] if eos else ids[:step+1],skip_special_tokens=False,clean_up_tokenization_spaces=False)
                    if step+1==len(ids):
                        stops[row]='eos' if eos else 'complete_packet'
                        assert eos or parse(texts[row],r) is not None
            for row in (0,1):
                slot=ordinal+row;r=source[slot]
                closed=base/f'baseline_{ordinal:04d}_r{row}_closed.json'
                common.write(closed,dict(MOCK_ONLY=True,source_id=r['id'],root=r['root'],stage=r['stage'],split=r['split'],
                    mode='native',lane='baseline',initial_ids=initial[slot],generated_ids=generated[slot],text=texts[row],
                    stop_reason=stops[row],prefix_count=len(fragments[row]),fragments=fragments[row],
                    wall_seconds=.01,attributed_batch_timing=dict(cuda_elapsed_ms=1.,forward_wall_seconds=.001)))
                common.append(base/'closed.jsonl',dict(file=closed.name,sha256=common.sha(closed)))
        common.no_torch()
        before=common.sha(common.ROOT/'SOURCE_MANIFEST.json')
        sys.argv=['MOCK_ONLY_storyaudit30','--artifact-root',str(artifact),'--snapshot',args.snapshot]
        print(json.dumps(dict(MOCK_ONLY=True,action='Executing frozen main on synthetic files; not model grades')),flush=True)
        frozen.main()  # Real, unchanged main/guard/tokenizer/auditor/presentation.
        assert common.sha(common.ROOT/'SOURCE_MANIFEST.json')==before
        result=common.read(frozen.OUTPUTS[1]);private=common.read(common.RUNS/'private_episode_audits.json')
        assert result['audited_trajectories']==6 and result['missing_trajectories']==66
        assert result['status']=='partial_closed_story_block'
        assert sum(r['grade']['format_valid'] for r in private)==5
        assert sum(r['grade']['final_correct'] for r in private)==4
        assert sum(r['grade']['joint_correct'] for r in private)==4
        assert not private[1]['grade']['format_valid'] and private[4]['grade']['format_valid'] and not private[4]['grade']['final_correct']
        assert sum(g['observed_worlds'] for g in result['groups'] if g['split']=='development')==0
        assert all(g['endpoints']['joint_correct']['correct_worlds'] is None for g in result['groups'] if g['split']=='development')
        for group in result['groups']:
            assert group['observed_worlds']+group['missing_worlds']==group['registered_worlds']
        completion=common.read(common.RUNS/'completion.json')
        assert all(common.sha(name)==digest for name,digest in completion['outputs_sha256'].items())
        common.no_torch()
        lengths=[len(ids) for ids,text in reference]
        summary=dict(MOCK_ONLY=True,status='passed',UTC=datetime.now(timezone.utc).isoformat(),
            actual_frozen_main_executed=True,source_manifest_unchanged=True,source_manifest_sha256=before,
            synthetic_episodes=6,synthetic_missing_slots=66,synthetic_format_valid=5,synthetic_final_correct=4,synthetic_joint_correct=4,
            synthetic_prefixes=result['captured_prefixes'],synthetic_vectors=result['physical_vectors_checked'],
            malformed_EOS_and_well_formed_wrong_world_both_preserved=True,DEV_outcomes_all_unknown=True,
            all_72_public_host_reference_packets_roundtrip_and_fit_64_tokens=True,
            reference_token_count_min=min(lengths),reference_token_count_max=max(lengths),
            reference_packet_scope='Syntax/budget check only, including host-gold packets for erased controls; not source-conditioned model competence.',
            inner_audit_resources=result['audit_resources'],new_pretrained_forwards=0,model_loaded=False,Torch_imported=False,
            real_generated_text_or_state_files_read=False,real_worker_modified=False,sealed_test_accessed=False,
            validation_scope='Synthetic full runner and private presentation only. All real episode grades/SAE/BPD/parameter verification remain pending.')
    resources=common.read(budget.RUNS/'resources.json');assert resources['status']=='passed'
    summary['whole_fixture_resources']=dict(sample_count=len(resources['samples']),
        maximum_sampled_private_bytes=max(v['private_bytes'] for v in resources['samples']),
        minimum_sampled_available_physical_bytes=min(v['available_physical_bytes'] for v in resources['samples']))
    common.write(out/'engineering_receipt_MOCK_ONLY.json',summary)
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
